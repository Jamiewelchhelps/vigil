"""M0 ingest: find OPERA DISP-S1 products for a region and save bounding-box subsets.

    python -m pipeline.ingest --region houston                # M0 acceptance command
    python -m pipeline.ingest --region houston --find-frames  # which frames cover the bbox
    python -m pipeline.ingest --region houston --dry-run      # what would be downloaded

What it does, per frame listed in ``config/regions.yaml``:
1. Lists every DISP-S1 granule for the frame from NASA CMR (no login needed). Stops if CMR
   returns fewer granules than it reports as hits.
2. For each granule not already on disk, resolves the Earthdata-protected URL to a short-lived
   signed URL, then reads only the HDF5 chunks covering the bbox (HTTP range requests), and
   writes a subset NetCDF under the original OPERA file name. Files are written to ``*.partial``
   and renamed when complete, so a rerun never mistakes a half-written file for a finished one.
3. Downloads the frame's DISP-S1-STATIC layers (LOS unit vectors, DEM, layover/shadow mask)
   and keeps a bbox crop of each.
4. Prints, per frame: epoch count, date range, % valid pixels, and writes
   ``data/raw/<region>/manifest.json`` plus ``reports/m0_ingest_<region>.md``.

The on-disk layout is the ingest -> stack contract in :mod:`pipeline.schemas`.

Why not ``opera-utils disp-s1-download``: its bbox path needs the OPERA frame database from
GitHub releases (not reachable from every environment), maps the bbox with two corners only
(a UTM-projected bbox is slightly rotated, so corners can be clipped), and over HTTPS downloads
each full product before cropping. See the Decision Log in BUILD_SPINE.md.

Credentials come only from env vars or ``~/.netrc`` (see :class:`EarthdataAuth`).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import logging
import multiprocessing
import netrc
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import yaml

from pipeline.schemas import (
    PARTIAL_SUFFIX,
    FrameSummary,
    IngestManifest,
    frame_dir,
    manifest_path,
    static_dir,
)

logger = logging.getLogger("vigil.ingest")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "regions.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
DISP_SHORT_NAME = "OPERA_L3_DISP-S1_V1"
STATIC_SHORT_NAME = "OPERA_L3_DISP-S1-STATIC_V1"

URS_HOST = "urs.earthdata.nasa.gov"
# Hosts that serve OPERA files behind Earthdata Login. A token is only ever sent to these,
# never to the signed S3/CloudFront URL they redirect to.
EARTHDATA_DATA_HOSTS = frozenset(
    {
        "cumulus.asf.earthdatacloud.nasa.gov",
        "cumulus.asf.alaska.edu",
        "datapool.asf.alaska.edu",
    }
)

# Layers the pipeline uses (BUILD_SPINE.md 3.1). A product missing any of them is an error.
REQUIRED_LAYERS = (
    "displacement",
    "short_wavelength_displacement",
    "temporal_coherence",
    "recommended_mask",
    "persistent_scatterer_mask",
    "estimated_phase_quality",
)
REQUIRED_CORRECTIONS = ("ionospheric_delay", "solid_earth_tide")
EXTRA_GROUPS = ("metadata", "identification")  # copied verbatim with h5py
REQUIRED_STATIC_LAYERS = ("line_of_sight_enu", "dem", "layover_shadow_mask")

VALID_RULE = "displacement is finite (not fill) and recommended_mask == 1"

_TS = r"\d{8}T\d{6}Z"
DISP_NAME_RE = re.compile(
    rf"OPERA_L3_DISP-S1_IW_F(?P<frame>\d{{5}})_(?P<pol>[A-Z]{{2}})_"
    rf"(?P<ref>{_TS})_(?P<sec>{_TS})_v(?P<version>[\d.]+)_(?P<prod>{_TS})"
)
STATIC_NAME_RE = re.compile(
    r"OPERA_L3_DISP-S1-STATIC_F(?P<frame>\d{5})_(?P<validity>\d{8})_(?P<sensor>S1[A-Z])"
    r"_v(?P<version>[\d.]+)(?:_(?P<layer>[a-z_]+))?"
)

CHUNK = 256  # output NetCDF chunk size (pixels)
MIB = 1024 * 1024


class IngestError(RuntimeError):
    """Anything that must stop the run and be reported (never papered over)."""


class EarthdataAuthError(IngestError):
    """The data host wants an Earthdata login we cannot provide."""


class MissingLayerError(IngestError):
    """A product lacks a layer the pipeline needs."""


# --------------------------------------------------------------------------------------
# Region config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSpec:
    frame_id: int
    direction: str  # "ascending" | "descending"
    path: int | None = None


@dataclass(frozen=True)
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)
    frames: tuple[FrameSpec, ...]
    product_version: str = "1.0"
    description: str = ""


def load_region(name: str, config_path: Path = DEFAULT_CONFIG) -> Region:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if name not in cfg:
        raise IngestError(f"Region {name!r} not in {config_path} (have: {sorted(cfg)})")
    r = cfg[name]
    bbox = tuple(float(v) for v in r["bbox"])
    if len(bbox) != 4:
        raise IngestError(f"{name}: bbox must be [west, south, east, north]")
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise IngestError(f"{name}: invalid bbox {bbox}, expected [west, south, east, north]")
    frames = []
    for fr in r.get("frames") or []:
        direction = str(fr["direction"]).lower()
        if direction not in ("ascending", "descending"):
            raise IngestError(f"{name}: frame {fr['frame_id']} direction must be ascending/descending")
        frames.append(FrameSpec(int(fr["frame_id"]), direction, fr.get("path")))
    return Region(
        name=name,
        bbox=bbox,  # type: ignore[arg-type]
        frames=tuple(frames),
        product_version=str(r.get("product_version", "1.0")),
        description=str(r.get("description", "")).strip(),
    )


# --------------------------------------------------------------------------------------
# CMR search
# --------------------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class DispGranule:
    granule_ur: str
    frame_id: int
    polarization: str
    reference_time: datetime
    secondary_time: datetime
    version: str
    production_time: datetime
    url: str  # HTTPS URL of the .nc file (or a local path, in tests)

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name if "://" in self.url else Path(self.url).name


def parse_disp_name(name: str) -> dict[str, Any]:
    m = DISP_NAME_RE.search(name)
    if not m:
        raise IngestError(f"Not an OPERA DISP-S1 name: {name}")
    return {
        "frame_id": int(m["frame"]),
        "polarization": m["pol"],
        "reference_time": _parse_ts(m["ref"]),
        "secondary_time": _parse_ts(m["sec"]),
        "version": m["version"],
        "production_time": _parse_ts(m["prod"]),
    }


def _https_urls(umm: dict) -> list[str]:
    return [
        u["URL"]
        for u in umm.get("RelatedUrls", [])
        if u.get("Type") == "GET DATA" and str(u.get("URL", "")).startswith("https://")
    ]


def disp_granule_from_umm(umm: dict) -> DispGranule:
    ur = umm["GranuleUR"]
    nc = [u for u in _https_urls(umm) if u.endswith(".nc")]
    if not nc:
        raise IngestError(f"No HTTPS .nc URL in CMR record {ur}")
    info = parse_disp_name(Path(urlparse(nc[0]).path).name)
    return DispGranule(granule_ur=ur, url=nc[0], **info)


@dataclass(frozen=True)
class StaticGranule:
    granule_ur: str
    frame_id: int
    validity_date: str  # YYYYMMDD
    version: str
    layer_urls: dict[str, str]  # layer name -> HTTPS URL of the .tif


def static_granule_from_umm(umm: dict) -> StaticGranule:
    ur = umm["GranuleUR"]
    m = STATIC_NAME_RE.search(ur)
    if not m:
        raise IngestError(f"Not an OPERA DISP-S1-STATIC name: {ur}")
    layers: dict[str, str] = {}
    for url in _https_urls(umm):
        if not url.endswith(".tif"):
            continue
        lm = STATIC_NAME_RE.search(Path(urlparse(url).path).stem)
        if lm and lm["layer"]:
            layers[lm["layer"]] = url
    return StaticGranule(ur, int(m["frame"]), m["validity"], m["version"], layers)


def cmr_search(params: dict, session: requests.Session, page_size: int = 2000) -> list[dict]:
    """All UMM items for a CMR granule query, following CMR-Search-After paging.

    Raises IngestError if the number of items differs from CMR's reported hit count, so a
    truncated listing can never silently drop epochs.
    """
    items: list[dict] = []
    headers: dict[str, str] = {}
    hits: int | None = None
    while True:
        r = session.get(
            CMR_GRANULES_URL, params={**params, "page_size": page_size}, headers=headers, timeout=120
        )
        r.raise_for_status()
        if hits is None and r.headers.get("CMR-Hits") is not None:
            hits = int(r.headers["CMR-Hits"])
        batch = r.json().get("items", [])
        items.extend(batch)
        after = r.headers.get("CMR-Search-After")
        if not batch or not after or (hits is not None and len(items) >= hits):
            break
        headers = {"CMR-Search-After": after}
    if hits is not None and len(items) != hits:
        raise IngestError(f"CMR returned {len(items)} items but reported {hits} hits for {params}")
    return items


def dedupe_latest(granules: Iterable[DispGranule]) -> list[DispGranule]:
    """One granule per (frame, reference, secondary) pair: the most recently produced one."""
    best: dict[tuple, DispGranule] = {}
    for g in granules:
        key = (g.frame_id, g.reference_time, g.secondary_time)
        if key not in best or g.production_time > best[key].production_time:
            best[key] = g
    return sorted(best.values(), key=lambda g: (g.frame_id, g.secondary_time))


def search_disp(
    frame_id: int, product_version: str, session: requests.Session
) -> list[DispGranule]:
    params = {
        "short_name": DISP_SHORT_NAME,
        "attribute[]": [f"int,FRAME_NUMBER,{frame_id}", f"float,PRODUCT_VERSION,{product_version}"],
    }
    granules = [disp_granule_from_umm(it["umm"]) for it in cmr_search(params, session)]
    wrong = [g.granule_ur for g in granules if g.frame_id != frame_id]
    if wrong:
        raise IngestError(f"CMR returned granules from another frame: {wrong[:3]}")
    return dedupe_latest(granules)


def search_static(frame_id: int, session: requests.Session) -> StaticGranule:
    params = {"short_name": STATIC_SHORT_NAME, "attribute[]": [f"int,FRAME_NUMBER,{frame_id}"]}
    found = [static_granule_from_umm(it["umm"]) for it in cmr_search(params, session)]
    found = [g for g in found if g.frame_id == frame_id]
    if not found:
        raise IngestError(f"No DISP-S1-STATIC granule found for frame {frame_id}")
    found.sort(key=lambda g: (g.validity_date, g.version))
    if len(found) > 1:
        logger.info(
            "Frame %s has %d STATIC granules; using the newest (%s)",
            frame_id, len(found), found[-1].granule_ur,
        )
    chosen = found[-1]
    missing = [layer for layer in REQUIRED_STATIC_LAYERS if layer not in chosen.layer_urls]
    if missing:
        raise MissingLayerError(f"{chosen.granule_ur} is missing STATIC layers {missing}")
    return chosen


def find_frames(bbox: Sequence[float], session: requests.Session) -> list[dict[str, Any]]:
    """Frames with DISP-S1 granules intersecting the bbox, with how much of the bbox each covers."""
    from shapely.geometry import Polygon, box

    items = cmr_search(
        {"short_name": DISP_SHORT_NAME, "bounding_box": ",".join(str(v) for v in bbox)}, session
    )
    target = box(*bbox)
    frames: dict[int, dict[str, Any]] = {}
    for it in items:
        umm = it["umm"]
        attrs = {a["Name"]: a["Values"] for a in umm.get("AdditionalAttributes", [])}
        fid = int(attrs["FRAME_NUMBER"][0])
        sec = parse_disp_name(umm["GranuleUR"])["secondary_time"].date().isoformat()
        f = frames.setdefault(
            fid,
            {
                "frame_id": fid,
                "direction": attrs.get("ASCENDING_DESCENDING", ["?"])[0].lower(),
                "path": int(attrs["PATH_NUMBER"][0]) if "PATH_NUMBER" in attrs else None,
                "n_granules": 0,
                "first": sec,
                "last": sec,
                "coverage": 0.0,
            },
        )
        f["n_granules"] += 1
        f["first"], f["last"] = min(f["first"], sec), max(f["last"], sec)
        for gp in umm.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get(
            "Geometry", {}
        ).get("GPolygons", []):
            pts = [(p["Longitude"], p["Latitude"]) for p in gp["Boundary"]["Points"]]
            cov = Polygon(pts).intersection(target).area / target.area
            f["coverage"] = max(f["coverage"], cov)
    return sorted(frames.values(), key=lambda f: (-f["coverage"], f["frame_id"]))


# --------------------------------------------------------------------------------------
# Earthdata auth and URL resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EarthdataAuth:
    """Earthdata credentials from env vars or ~/.netrc only. Never from the repo.

    Order: ``EARTHDATA_TOKEN`` (user token, sent as a bearer token to the ASF data hosts),
    then ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` or a ``~/.netrc`` entry for
    ``urs.earthdata.nasa.gov`` (sent to Earthdata Login only). With neither, requests go out
    without credentials, which works where a proxy injects them (Claude Code cloud sessions).
    """

    token: str | None = None
    username: str | None = None
    password: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "EarthdataAuth":
        token = os.environ.get("EARTHDATA_TOKEN") or None
        user = os.environ.get("EARTHDATA_USERNAME") or None
        pw = os.environ.get("EARTHDATA_PASSWORD") or None
        if not (user and pw):
            try:
                found = netrc.netrc().authenticators(URS_HOST)
            except (FileNotFoundError, netrc.NetrcParseError):
                found = None
            if found and found[0] and found[2]:
                user, pw = found[0], found[2]
            else:
                user = pw = None
        return cls(token=token, username=user, password=pw)

    @property
    def can_login(self) -> bool:
        return bool(self.username and self.password)

    def headers_for(self, url: str) -> dict[str, str]:
        host = urlparse(url).hostname or ""
        if self.token and host in EARTHDATA_DATA_HOSTS:
            return {"Authorization": f"Bearer {self.token}"}
        if self.can_login and host == URS_HOST:
            raw = f"{self.username}:{self.password}".encode()
            return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
        return {}


_AUTH_HELP = (
    "Set EARTHDATA_TOKEN, or EARTHDATA_USERNAME/EARTHDATA_PASSWORD, or add a ~/.netrc entry for "
    "urs.earthdata.nasa.gov. If you already did: make sure the Earthdata account has authorized "
    "ASF's download application (open one product URL in a browser while logged in and approve) "
    "and that the token has not expired."
)


@dataclass(frozen=True)
class ResolvedUrl:
    url: str
    headers: dict[str, str]  # headers to send when reading `url` (empty for signed URLs)


def resolve_url(
    url: str, session: requests.Session, auth: EarthdataAuth, max_hops: int = 10
) -> ResolvedUrl:
    """Follow the Earthdata redirect chain until a URL we can read directly.

    Usually ends at a short-lived signed S3/CloudFront URL, read with no auth headers.
    Credentials are only attached to Earthdata hosts (see :meth:`EarthdataAuth.headers_for`).
    """
    cur = url
    for _ in range(max_hops):
        host = urlparse(cur).hostname or ""
        headers = {**auth.headers_for(cur), "Range": "bytes=0-0"}
        resp = session.get(cur, headers=headers, allow_redirects=False, stream=True, timeout=60)
        resp.close()
        if resp.status_code in (301, 302, 303, 307, 308) and "Location" in resp.headers:
            nxt = urljoin(cur, resp.headers["Location"])
            nxt_host = urlparse(nxt).hostname or ""
            if nxt_host == URS_HOST and not auth.can_login and host != URS_HOST:
                raise EarthdataAuthError(
                    f"{host} redirected to Earthdata Login and no username/password is "
                    f"available (or the token was not accepted). {_AUTH_HELP}"
                )
            if nxt_host not in EARTHDATA_DATA_HOSTS and nxt_host != URS_HOST:
                return ResolvedUrl(nxt, {})
            cur = nxt
            continue
        if resp.status_code in (200, 206):
            if host == URS_HOST:
                raise EarthdataAuthError(
                    "Earthdata Login answered with a page instead of redirecting back to the "
                    f"data host: the account probably has to authorize the application. {_AUTH_HELP}"
                )
            return ResolvedUrl(cur, auth.headers_for(cur))
        if resp.status_code in (401, 403):
            raise EarthdataAuthError(f"HTTP {resp.status_code} from {host}. {_AUTH_HELP}")
        resp.raise_for_status()
        raise IngestError(f"Unexpected HTTP {resp.status_code} from {host}")
    raise IngestError(f"More than {max_hops} redirects resolving {url}")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "vigil-ingest/0.0.1 (+python-requests)"
    return s


# --------------------------------------------------------------------------------------
# Geometry: bbox -> pixel window on the product's own grid
# --------------------------------------------------------------------------------------


def bbox_ring(bbox: Sequence[float], n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Densified boundary of a lon/lat bbox (so its projected envelope is exact enough)."""
    w, s, e, nn = bbox
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    lon = np.concatenate([w + (e - w) * t, np.full(n, e), e - (e - w) * t, np.full(n, w)])
    lat = np.concatenate([np.full(n, s), s + (nn - s) * t, np.full(n, nn), nn - (nn - s) * t])
    return lon, lat


def bbox_envelope(bbox: Sequence[float], crs: Any) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of the lon/lat bbox projected into `crs`."""
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True)
    x, y = tr.transform(*bbox_ring(bbox))
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def bbox_polygon_projected(bbox: Sequence[float], crs: Any):
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon

    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True)
    x, y = tr.transform(*bbox_ring(bbox))
    return Polygon(zip(x, y))


def _axis_window(coord: np.ndarray, lo: float, hi: float, pad: int) -> slice | None:
    coord = np.asarray(coord, dtype=float)
    half = abs(coord[1] - coord[0]) / 2 if coord.size > 1 else 0.0
    idx = np.nonzero((coord + half >= lo) & (coord - half <= hi))[0]
    if idx.size == 0:
        return None
    return slice(max(int(idx.min()) - pad, 0), min(int(idx.max()) + 1 + pad, coord.size))


def index_window(
    x: np.ndarray, y: np.ndarray, envelope: Sequence[float], pad: int = 1
) -> tuple[slice, slice]:
    """(row_slice, col_slice) of all pixels touching the envelope, plus `pad` pixels."""
    xmin, ymin, xmax, ymax = envelope
    cols = _axis_window(x, xmin, xmax, pad)
    rows = _axis_window(y, ymin, ymax, pad)
    if cols is None or rows is None:
        raise IngestError(
            f"Region envelope {tuple(round(v) for v in envelope)} does not overlap the product grid "
            f"x=[{float(np.min(x)):.0f}, {float(np.max(x)):.0f}] "
            f"y=[{float(np.min(y)):.0f}, {float(np.max(y)):.0f}]"
        )
    return rows, cols


def dataset_crs(ds) -> Any:
    """CRS of a DISP-S1 dataset, from its CF grid-mapping variable."""
    from pyproj import CRS

    for name in ("spatial_ref", "crs"):
        if name in ds.variables:
            attrs = ds[name].attrs
            for key in ("crs_wkt", "spatial_ref"):
                if attrs.get(key):
                    return CRS.from_wkt(attrs[key])
            if attrs.get("epsg_code") or attrs.get("epsg"):
                return CRS.from_epsg(int(attrs.get("epsg_code") or attrs.get("epsg")))
    raise MissingLayerError("No CRS found (no spatial_ref/crs variable with crs_wkt)")


# --------------------------------------------------------------------------------------
# Subsetting one product
# --------------------------------------------------------------------------------------


def _subset_group(ds, envelope, fallback: tuple[slice, slice] | None, full_shape):
    has_dims = "y" in ds.dims and "x" in ds.dims
    if not has_dims:
        return ds
    if "x" in ds.coords and "y" in ds.coords:
        rows, cols = index_window(ds["x"].values, ds["y"].values, envelope)
    elif fallback is not None and (ds.sizes["y"], ds.sizes["x"]) == full_shape:
        rows, cols = fallback
    else:
        raise IngestError("Group has y/x dimensions but no coordinates to subset by")
    return ds.isel(y=rows, x=cols)


def _encoding(ds) -> dict[str, dict]:
    enc: dict[str, dict] = {}
    for name, var in ds.data_vars.items():
        if var.ndim >= 2 and var.dtype.kind in "fiub":
            chunks = tuple([1] * (var.ndim - 2) + [min(CHUNK, s) for s in var.shape[-2:]])
            enc[name] = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": chunks}
    return enc


def write_subset(source: Any, dst: Path, bbox: Sequence[float]) -> dict[str, Any]:
    """Crop one DISP-S1 product to the bbox and write it to `dst` atomically.

    `source` is a local path or an open binary file object (e.g. fsspec over HTTPS).
    Data are copied raw (no CF decoding), so values, fill values and attributes are unchanged.
    Raises MissingLayerError if a required layer is absent; nothing is written in that case.
    """
    import h5py
    import xarray as xr

    dst = Path(dst)
    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    tmp.unlink(missing_ok=True)

    def _rewind():
        if hasattr(source, "seek"):
            source.seek(0)

    try:
        _rewind()
        root = xr.open_dataset(source, engine="h5netcdf", decode_cf=False)
        missing = [v for v in REQUIRED_LAYERS if v not in root.variables]
        if missing:
            raise MissingLayerError(f"{dst.name}: missing layers {missing}")
        crs = dataset_crs(root)
        envelope = bbox_envelope(bbox, crs)
        rows, cols = index_window(root["x"].values, root["y"].values, envelope)
        full_shape = (root.sizes["y"], root.sizes["x"])
        sub = root.isel(y=rows, x=cols)
        sub.to_netcdf(tmp, engine="h5netcdf", encoding=_encoding(sub))
        shape = (sub.sizes["y"], sub.sizes["x"])
        root.close()

        _rewind()
        corr = xr.open_dataset(source, engine="h5netcdf", group="corrections", decode_cf=False)
        missing = [v for v in REQUIRED_CORRECTIONS if v not in corr.variables]
        if missing:
            raise MissingLayerError(f"{dst.name}: missing corrections {missing}")
        corr_sub = _subset_group(corr, envelope, (rows, cols), full_shape)
        corr_sub.to_netcdf(
            tmp, mode="a", engine="h5netcdf", group="corrections", encoding=_encoding(corr_sub)
        )
        corr.close()

        _rewind()
        with h5py.File(source, "r") as src, h5py.File(tmp, "a") as out:
            for group in EXTRA_GROUPS:
                if group in src:
                    src.copy(group, out, name=group)
            out.attrs["vigil_subset_bbox_lonlat"] = np.asarray(bbox, dtype=float)
            out.attrs["vigil_subset_rows"] = np.asarray([rows.start, rows.stop])
            out.attrs["vigil_subset_cols"] = np.asarray([cols.start, cols.stop])
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"shape": shape, "rows": (rows.start, rows.stop), "cols": (cols.start, cols.stop)}


def open_remote(resolved: ResolvedUrl, block_size: int = 4 * MIB):
    """Binary file object over HTTPS that fetches only the byte ranges h5py asks for."""
    import fsspec

    fs = fsspec.filesystem(
        "https", client_kwargs={"headers": resolved.headers, "trust_env": True}
    )
    return fs.open(resolved.url, mode="rb", block_size=block_size, cache_type="blockcache")


def _download(resolved: ResolvedUrl, dst: Path, session: requests.Session) -> int:
    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    n = 0
    try:
        with session.get(resolved.url, headers=resolved.headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * MIB):
                    f.write(chunk)
                    n += len(chunk)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


@dataclass
class FetchResult:
    filename: str
    ok: bool
    seconds: float = 0.0
    bytes_transferred: int | None = None
    error: str | None = None
    fatal: bool = False  # auth or missing-layer errors: stop the whole run


def fetch_subset(
    url: str, dst: str, bbox: Sequence[float], method: str = "range", retries: int = 3
) -> FetchResult:
    """Download one product's bbox subset. Runs in a worker process."""
    dst_p = Path(dst)
    t0 = time.monotonic()
    last_err = ""
    for attempt in range(retries):
        try:
            if not url.startswith(("https://", "http://")):  # local file (tests, reprocessing)
                write_subset(Path(url), dst_p, bbox)
                return FetchResult(dst_p.name, True, time.monotonic() - t0, 0)
            session = make_session()
            resolved = resolve_url(url, session, EarthdataAuth.from_env())
            if method == "range":
                with open_remote(resolved) as fobj:
                    write_subset(fobj, dst_p, bbox)
                    nbytes = getattr(getattr(fobj, "cache", None), "total_requested_bytes", None)
            elif method == "full":
                full = dst_p.with_name(dst_p.name + ".download")
                try:
                    nbytes = _download(resolved, full, session)
                    write_subset(full, dst_p, bbox)
                finally:
                    full.unlink(missing_ok=True)
            else:
                raise IngestError(f"Unknown method {method!r}")
            return FetchResult(dst_p.name, True, time.monotonic() - t0, nbytes)
        except (EarthdataAuthError, MissingLayerError) as e:
            return FetchResult(dst_p.name, False, time.monotonic() - t0, None, str(e), fatal=True)
        except Exception as e:  # network hiccups: retry with backoff
            last_err = f"{type(e).__name__}: {e}"
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    return FetchResult(dst_p.name, False, time.monotonic() - t0, None, last_err)


# --------------------------------------------------------------------------------------
# STATIC layers
# --------------------------------------------------------------------------------------


def crop_static(src: Path, dst: Path, bbox: Sequence[float]) -> tuple[int, int]:
    """Crop a STATIC GeoTIFF to the bbox (native grid, 1-pixel margin), written atomically."""
    import rasterio
    from rasterio.windows import Window, from_bounds

    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    try:
        with rasterio.open(src) as ds:
            env = bbox_envelope(bbox, ds.crs)
            win = from_bounds(*env, transform=ds.transform)
            c0 = max(int(np.floor(win.col_off)) - 1, 0)
            r0 = max(int(np.floor(win.row_off)) - 1, 0)
            c1 = min(int(np.ceil(win.col_off + win.width)) + 1, ds.width)
            r1 = min(int(np.ceil(win.row_off + win.height)) + 1, ds.height)
            if c1 <= c0 or r1 <= r0:
                raise IngestError(f"{src.name}: bbox does not overlap the STATIC raster")
            window = Window(c0, r0, c1 - c0, r1 - r0)
            data = ds.read(window=window)
            profile = ds.profile.copy()
            profile.update(
                driver="GTiff",
                width=window.width,
                height=window.height,
                transform=ds.window_transform(window),
                compress="deflate",
            )
            for key in ("blockxsize", "blockysize", "tiled"):
                profile.pop(key, None)
            with rasterio.open(tmp, "w", **profile) as out:
                out.write(data)
                out.update_tags(**ds.tags())
                for b in range(1, ds.count + 1):
                    out.update_tags(b, **ds.tags(b))
                    if ds.descriptions[b - 1]:
                        out.set_band_description(b, ds.descriptions[b - 1])
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return int(window.height), int(window.width)


def fetch_static(sg: StaticGranule, out_dir: Path, bbox: Sequence[float]) -> list[str]:
    """Crop every STATIC layer of the granule into out_dir. Skips layers already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    session = None
    auth = None
    for layer, url in sorted(sg.layer_urls.items()):
        name = Path(urlparse(url).path).name if "://" in url else Path(url).name
        dst = out_dir / name
        if dst.exists():
            done.append(layer)
            continue
        if not url.startswith(("https://", "http://")):
            crop_static(Path(url), dst, bbox)
        else:
            session = session or make_session()
            auth = auth or EarthdataAuth.from_env()
            full = out_dir / (name + ".download")
            try:
                _download(resolve_url(url, session, auth), full, session)
                crop_static(full, dst, bbox)
            finally:
                full.unlink(missing_ok=True)
        done.append(layer)
    return done


# --------------------------------------------------------------------------------------
# Planning, summary, report
# --------------------------------------------------------------------------------------


def clean_partials(directory: Path) -> int:
    """Remove leftovers of interrupted writes. Returns how many were removed."""
    n = 0
    if directory.exists():
        for p in list(directory.glob(f"*{PARTIAL_SUFFIX}")) + list(directory.glob("*.download")):
            p.unlink()
            n += 1
    return n


def plan_downloads(
    granules: Sequence[DispGranule], out_dir: Path
) -> tuple[list[DispGranule], list[DispGranule]]:
    """Split granules into (to_fetch, already_on_disk). Only complete files count."""
    to_fetch, existing = [], []
    for g in granules:
        (existing if (out_dir / g.filename).is_file() else to_fetch).append(g)
    return to_fetch, existing


def summarize_frame(
    out_dir: Path,
    bbox: Sequence[float],
    frame_id: int,
    direction: str,
    n_expected: int,
) -> FrameSummary:
    """Epochs, dates and % valid pixels for the subsets on disk (computed inside the bbox)."""
    import shapely
    import xarray as xr

    files = sorted(out_dir.glob("OPERA_L3_DISP-S1_*.nc"))
    infos = [parse_disp_name(p.name) for p in files]
    secs = sorted({i["secondary_time"] for i in infos})
    refs = {i["reference_time"] for i in infos}
    gap_days = gap_start = gap_end = None
    if len(secs) > 1:
        diffs = np.diff(np.array([s.timestamp() for s in secs])) / 86400.0
        k = int(np.argmax(diffs))
        gap_days, gap_start, gap_end = (
            round(float(diffs[k]), 1), secs[k].date().isoformat(), secs[k + 1].date().isoformat()
        )

    crs_str = shape = pixel = None
    n_in = None
    per_epoch: list[float] = []
    valid_total = 0
    inside_cache: dict[tuple, np.ndarray] = {}
    for p in files:
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            x, y = ds["x"].values, ds["y"].values
            key = (x[0], x[-1], x.size, y[0], y[-1], y.size)
            if key not in inside_cache:
                crs = dataset_crs(ds)
                poly = bbox_polygon_projected(bbox, crs)
                xx, yy = np.meshgrid(x, y)
                inside_cache[key] = shapely.contains_xy(poly, xx, yy)
                if crs_str is None:
                    epsg = crs.to_epsg()
                    crs_str = f"EPSG:{epsg}" if epsg else crs.to_string()
                    shape = [int(y.size), int(x.size)]
                    pixel = [float(abs(x[1] - x[0])), float(abs(y[1] - y[0]))]
            inside = inside_cache[key]
            disp = ds["displacement"].values
            mask = ds["recommended_mask"].values
            valid = np.isfinite(disp) & (mask == 1) & inside
            n_inside = int(inside.sum())
            n_in = n_inside if n_in is None else n_in
            valid_total += int(valid.sum())
            per_epoch.append(100.0 * valid.sum() / n_inside if n_inside else 0.0)

    denom = (n_in or 0) * len(files)
    static_layers = []
    sdir = out_dir / "static"
    for layer in REQUIRED_STATIC_LAYERS:
        if any(sdir.glob(f"*_{layer}.tif")):
            static_layers.append(layer)
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    return FrameSummary(
        frame_id=frame_id,
        direction=direction,
        n_expected=n_expected,
        n_on_disk=len(files),
        n_missing=max(n_expected - len(files), 0),
        n_epochs=len(secs),
        first_date=secs[0].date().isoformat() if secs else None,
        last_date=secs[-1].date().isoformat() if secs else None,
        n_reference_dates=len(refs),
        largest_gap_days=gap_days,
        largest_gap_start=gap_start,
        largest_gap_end=gap_end,
        crs=crs_str,
        shape=shape,
        pixel_size_m=pixel,
        n_pixels_in_bbox=n_in,
        pct_valid=round(100.0 * valid_total / denom, 2) if denom else None,
        pct_valid_epoch_min=round(float(np.min(per_epoch)), 2) if per_epoch else None,
        pct_valid_epoch_median=round(float(np.median(per_epoch)), 2) if per_epoch else None,
        pct_valid_epoch_max=round(float(np.max(per_epoch)), 2) if per_epoch else None,
        valid_rule=VALID_RULE,
        static_layers=static_layers,
        bytes_on_disk=size,
    )


def format_summary(region: Region, frames: Sequence[FrameSummary]) -> str:
    w, s, e, n = region.bbox
    lines = [
        f"Region {region.name}: bbox W {w} S {s} E {e} N {n} (EPSG:4326)",
        "Provenance: counts and dates MEASURED (from product files); percentages DERIVED.",
    ]
    for f in frames:
        lines += [
            "",
            f"Frame F{f.frame_id:05d} ({f.direction})",
            f"  epochs on disk     : {f.n_epochs} (files {f.n_on_disk} of {f.n_expected} listed by CMR, "
            f"{f.n_missing} missing)",
            f"  date range         : {f.first_date} to {f.last_date}",
            f"  reference dates    : {f.n_reference_dates}",
            f"  largest gap        : {f.largest_gap_days} days ({f.largest_gap_start} to {f.largest_gap_end})",
            f"  grid               : {f.shape} pixels, {f.pixel_size_m} m, {f.crs}; "
            f"{f.n_pixels_in_bbox} pixel centres inside bbox",
            f"  % valid pixels     : {f.pct_valid}% overall; per epoch min {f.pct_valid_epoch_min}% / "
            f"median {f.pct_valid_epoch_median}% / max {f.pct_valid_epoch_max}%",
            f"  valid means        : {f.valid_rule}",
            f"  STATIC layers      : {', '.join(f.static_layers) or 'none'}",
            f"  disk               : {f.bytes_on_disk / MIB:.1f} MiB",
        ]
    return "\n".join(lines)


def write_report(path: Path, region: Region, frames: Sequence[FrameSummary], run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# M0 ingest report: {region.name}",
        "",
        f"Generated {run['finished_utc']} by `python -m pipeline.ingest --region {region.name}`.",
        "",
        f"- Region: {region.description}",
        f"- Run: {run['fetched']} subsets downloaded, {run['skipped']} skipped (already on disk), "
        f"{run['failed']} failed; method `{run['method']}`; {run['seconds']:.0f} s.",
    ]
    if run.get("median_mib_per_file") is not None:
        body.append(
            f"- Transfer: median {run['median_mib_per_file']:.1f} MiB read per product (range reads)."
        )
    body += ["", "```", format_summary(region, frames), "```", ""]
    path.write_text("\n".join(body))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_frame_downloads(
    jobs: list[tuple[str, str]],
    bbox: Sequence[float],
    method: str,
    workers: int,
) -> list[FetchResult]:
    results: list[FetchResult] = []
    if not jobs:
        return results
    t0 = time.monotonic()
    if workers <= 1:  # in-process: simpler to debug, used by the tests
        for i, (url, dst) in enumerate(jobs, 1):
            res = fetch_subset(url, dst, tuple(bbox), method)
            results.append(res)
            if res.fatal:
                raise IngestError(f"{res.filename}: {res.error}")
            if not res.ok:
                logger.error("FAILED %s: %s", res.filename, res.error)
            if i % 10 == 0 or i == len(jobs):
                logger.info("  %d/%d done", i, len(jobs))
        return results
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(fetch_subset, url, dst, tuple(bbox), method): dst for url, dst in jobs}
        try:
            for i, fut in enumerate(cf.as_completed(futs), 1):
                res = fut.result()
                results.append(res)
                if res.fatal:
                    raise IngestError(f"{res.filename}: {res.error}")
                if not res.ok:
                    logger.error("FAILED %s: %s", res.filename, res.error)
                if i % 10 == 0 or i == len(jobs):
                    rate = i / (time.monotonic() - t0)
                    eta = (len(jobs) - i) / rate if rate else float("nan")
                    logger.info("  %d/%d done (%.2f files/s, ~%.0f min left)", i, len(jobs), rate, eta / 60)
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return results


def run_ingest(
    region: Region,
    data_dir: Path = DEFAULT_DATA_DIR,
    frames: Sequence[int] | None = None,
    method: str = "range",
    workers: int = 4,
    limit: int | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
    session: requests.Session | None = None,
) -> tuple[list[FrameSummary], dict]:
    session = session or make_session()
    selected = [f for f in region.frames if frames is None or f.frame_id in frames]
    if not selected:
        raise IngestError(f"No frames selected (config has {[f.frame_id for f in region.frames]})")
    t0 = time.monotonic()
    run: dict[str, Any] = {"fetched": 0, "skipped": 0, "failed": 0, "method": method}
    transfer_mib: list[float] = []
    failures: list[FetchResult] = []
    expected: dict[int, int] = {}

    for spec in selected:
        out = frame_dir(data_dir, region.name, spec.frame_id)
        out.mkdir(parents=True, exist_ok=True)
        removed = clean_partials(out) + clean_partials(out / "static")
        if removed:
            logger.info("F%05d: removed %d partial files from an interrupted run", spec.frame_id, removed)

        granules = search_disp(spec.frame_id, region.product_version, session)
        expected[spec.frame_id] = len(granules)
        to_fetch, existing = plan_downloads(granules, out)
        if limit is not None:
            to_fetch = to_fetch[:limit]
        logger.info(
            "F%05d (%s): %d granules in CMR, %d on disk, %d to fetch",
            spec.frame_id, spec.direction, len(granules), len(existing), len(to_fetch),
        )
        run["skipped"] += len(existing)

        sg = search_static(spec.frame_id, session)
        if dry_run:
            for g in to_fetch[:5]:
                logger.info("  would fetch %s", g.filename)
            if len(to_fetch) > 5:
                logger.info("  ... and %d more", len(to_fetch) - 5)
            logger.info("  STATIC %s: %s", sg.granule_ur, sorted(sg.layer_urls))
            continue

        layers = fetch_static(sg, static_dir(data_dir, region.name, spec.frame_id), region.bbox)
        logger.info("F%05d: STATIC layers on disk: %s", spec.frame_id, layers)

        jobs = [(g.url, str(out / g.filename)) for g in to_fetch]
        results = run_frame_downloads(jobs, region.bbox, method, workers)
        for r in results:
            if r.ok:
                run["fetched"] += 1
                if r.bytes_transferred:
                    transfer_mib.append(r.bytes_transferred / MIB)
            else:
                run["failed"] += 1
                failures.append(r)

    run["seconds"] = time.monotonic() - t0
    run["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run["median_mib_per_file"] = float(np.median(transfer_mib)) if transfer_mib else None
    if dry_run:
        return [], run

    summaries = [
        summarize_frame(
            frame_dir(data_dir, region.name, s.frame_id), region.bbox, s.frame_id, s.direction,
            expected[s.frame_id],
        )
        for s in selected
    ]
    manifest = IngestManifest(
        region=region.name,
        bbox=list(region.bbox),
        product_version=region.product_version,
        created_utc=run["finished_utc"],
        frames=summaries,
    )
    mpath = manifest_path(data_dir, region.name)
    mpath.write_text(json.dumps(manifest.to_dict(), indent=2))
    if report_path is not None:
        write_report(report_path, region, summaries, run)
    if failures:
        names = ", ".join(f.filename for f in failures[:5])
        raise IngestError(f"{len(failures)} products failed after retries (e.g. {names}); rerun to retry")
    return summaries, run


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--frames", type=lambda s: [int(v) for v in s.split(",")], default=None,
                    help="comma-separated frame IDs (default: all frames in the config)")
    ap.add_argument("--method", choices=["range", "full"], default="range",
                    help="range: read only the bbox chunks over HTTPS; full: download then crop")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N new products per frame")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--find-frames", action="store_true", help="list frames covering the bbox and exit")
    ap.add_argument("--no-report", action="store_true", help="do not write reports/m0_ingest_<region>.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "fsspec", "aiohttp", "asyncio", "h5py", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        region = load_region(args.region, args.config)
        if args.find_frames:
            rows = find_frames(region.bbox, make_session())
            print(f"{'frame':>6} {'direction':<11}{'path':>5} {'granules':>9} {'first':<11}{'last':<11}{'bbox cover':>10}")
            for f in rows:
                print(f"{f['frame_id']:>6} {f['direction']:<11}{f['path'] or '':>5} {f['n_granules']:>9} "
                      f"{f['first']:<11}{f['last']:<11}{100 * f['coverage']:>9.1f}%")
            return 0
        report = None if args.no_report else DEFAULT_REPORTS_DIR / f"m0_ingest_{region.name}.md"
        summaries, _ = run_ingest(
            region,
            data_dir=args.data_dir,
            frames=args.frames,
            method=args.method,
            workers=args.workers,
            limit=args.limit,
            dry_run=args.dry_run,
            report_path=report,
        )
        if summaries:
            print(format_summary(region, summaries))
        return 0
    except IngestError as e:
        logger.error("STOPPED: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
