"""M0 ingest: find OPERA DISP-S1 products for a region and save bounding-box subsets.

    python -m pipeline.ingest --region houston                # M0 acceptance command
    python -m pipeline.ingest --region houston --find-frames  # which frames cover the bbox
    python -m pipeline.ingest --region houston --dry-run      # what would be downloaded

What it does, per frame listed in ``config/regions.yaml``:
1. Lists every DISP-S1 granule for the frame from NASA CMR (no login needed). Stops if CMR
   returns fewer granules than it reports as hits.
2. For each granule not already on disk, resolves the Earthdata-protected URL to a short-lived
   signed URL, reads only the HDF5 chunks covering the bbox (HTTP range requests), and writes a
   subset NetCDF under the original OPERA file name. Files are written to ``*.partial``, fsynced
   and renamed when complete. On a rerun, an existing file only counts as done if it opens, has
   every required layer and was cut for the current bbox; unreadable files are re-fetched, and
   files cut for a different bbox stop the run (they are never silently mixed).
3. Downloads the frame's DISP-S1-STATIC layers (LOS unit vectors, DEM, layover/shadow mask)
   and keeps a bbox crop of each, with the same rerun checks.
4. Prints, per configured frame: epoch count, date range, % valid pixels, and writes
   ``data/raw/<region>/manifest.json`` plus ``reports/m0_ingest_<region>.md``.

The on-disk layout is the ingest -> stack contract in :mod:`pipeline.schemas`.

Why not ``opera-utils disp-s1-download``: its bbox path needs the OPERA frame database from
GitHub releases (not reachable from every environment), maps the bbox with two corners only
(a UTM-projected bbox is slightly rotated, so corners can be clipped), and over HTTPS downloads
each full product before cropping (~280 GB for the two Houston frames). See the Decision Log
in BUILD_SPINE.md.

Credentials come only from env vars or a netrc file (see :class:`EarthdataAuth`).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import logging
import multiprocessing
import os
import re
import shlex
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
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

BBOX_ATTR = "vigil_subset_bbox_lonlat"  # stored in every subset .nc (root attrs)
STATIC_BBOX_TAG = "VIGIL_SUBSET_BBOX_LONLAT"  # stored in every cropped STATIC GeoTIFF
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
RETRIES = 3
_sleep = time.sleep  # patched in tests


class IngestError(RuntimeError):
    """Anything that must stop the run and be reported (never papered over)."""


class FatalError(IngestError):
    """Retrying cannot help (bad credentials, missing layer, bad arguments): stop the run."""


class EarthdataAuthError(FatalError):
    """The data host wants an Earthdata login we cannot provide."""


class MissingLayerError(FatalError):
    """A product lacks a layer the pipeline needs."""


def _redact(text: str) -> str:
    """Drop query strings from URLs in messages (signed URLs carry credentials there)."""
    return re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\")]*", r"\1?<redacted>", str(text))


def _bbox_matches(stored: Any, bbox: Sequence[float]) -> bool:
    try:
        arr = np.asarray(stored, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    return arr.size == 4 and bool(np.allclose(arr, np.asarray(bbox, dtype=float), atol=1e-9))


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
# HTTP session (never reads ~/.netrc on its own) and retries
# --------------------------------------------------------------------------------------


class _NoNetrcAuth(requests.auth.AuthBase):
    """A session-level auth that does nothing, so requests never looks up ~/.netrc itself."""

    def __call__(self, r):
        return r


class _Session(requests.Session):
    """requests session where EarthdataAuth alone decides which host gets which credential.

    Plain requests would read ~/.netrc on every request and every redirect (including a
    ``default`` entry, which would send the Earthdata password to the signed S3 URL).
    Proxies and CA bundles from the environment are still honoured.
    """

    def __init__(self) -> None:
        super().__init__()
        self.auth = _NoNetrcAuth()
        self.headers["User-Agent"] = "vigil-ingest/0.0.1 (+python-requests)"

    def rebuild_auth(self, prepared_request, response):  # strip on host change, never add
        headers = prepared_request.headers
        if "Authorization" in headers and self.should_strip_auth(
            response.request.url, prepared_request.url
        ):
            del headers["Authorization"]


def make_session() -> requests.Session:
    return _Session()


def get_with_retries(
    session: requests.Session, url: str, what: str, retries: int = RETRIES, **kwargs
) -> requests.Response:
    """GET with retries on network errors, 429 and 5xx. Other 4xx stop immediately."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, **kwargs)
            code = getattr(r, "status_code", 200)
            if code == 429 or code >= 500:
                last = IngestError(f"HTTP {code}")
            elif code >= 400:
                raise IngestError(f"{what}: HTTP {code}")
            else:
                return r
        except requests.RequestException as e:
            last = e
        if attempt + 1 < retries:
            _sleep(2 ** (attempt + 1))
    raise IngestError(f"{what} failed after {retries} attempts: {_redact(last)}")


def with_retries(fn: Callable[[], Any], what: str, retries: int = RETRIES) -> Any:
    """Call fn, retrying network-type errors. FatalError (auth, missing layer) stops at once."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except FatalError:
            raise
        except Exception as e:  # network hiccups, truncated transfers, transient 5xx
            last = e
        if attempt + 1 < retries:
            _sleep(2 ** (attempt + 1))
    raise IngestError(f"{what} failed after {retries} attempts: {type(last).__name__}: {_redact(last)}")


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
        r = get_with_retries(
            session,
            CMR_GRANULES_URL,
            "CMR search",
            params={**params, "page_size": page_size},
            headers=headers,
            timeout=120,
        )
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


def _read_netrc_login(host: str) -> tuple[str, str] | None:
    """(login, password) for `host` from $NETRC, ~/.netrc or ~/_netrc (the files requests reads).

    A file that exists but cannot be parsed stops the run with the parser's message, rather
    than being silently ignored.
    """
    import netrc as netrc_mod

    candidates = [os.environ["NETRC"]] if os.environ.get("NETRC") else ["~/.netrc", "~/_netrc"]
    for cand in candidates:
        path = Path(os.path.expanduser(cand))
        if not path.is_file():
            continue
        try:
            entry = netrc_mod.netrc(str(path)).authenticators(host)
        except (netrc_mod.NetrcParseError, OSError) as e:
            raise EarthdataAuthError(f"Could not read {path}: {e}") from e
        if os.name == "posix" and path.stat().st_mode & 0o077:
            logger.warning("%s is readable by other users; run: chmod 600 %s", path, path)
        if entry and entry[0] and entry[2]:
            return entry[0], entry[2]
        return None
    return None


@dataclass(frozen=True)
class EarthdataAuth:
    """Earthdata credentials from env vars or a netrc file only. Never from the repo.

    Order: ``EARTHDATA_TOKEN`` (user token, sent as a bearer token to the ASF data hosts),
    then ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD``, then a netrc entry for
    ``urs.earthdata.nasa.gov`` (username/password are sent to Earthdata Login only). With
    none of these, requests go out without credentials, which works where a proxy injects them
    (Claude Code cloud sessions).
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
            found = _read_netrc_login(URS_HOST)
            user, pw = found if found else (None, None)
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
        if resp.status_code == 429 or resp.status_code >= 500:
            raise IngestError(f"HTTP {resp.status_code} from {host} (retryable)")
        raise IngestError(f"Unexpected HTTP {resp.status_code} from {host}")
    raise IngestError(f"More than {max_hops} redirects resolving {_redact(url)}")


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


def _as_text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "SU" and v.size == 1:
        return _as_text(v.item())
    return str(v)


def _update_georef(ds, rows: slice, cols: slice) -> None:
    """Make the subset describe its own grid: GDAL GeoTransform and reference_point indices.

    DISP-S1 stores a GDAL ``GeoTransform`` on ``spatial_ref``. GDAL prefers it over the x/y
    coordinates, so a stale full-frame value would place the subset at the frame origin.
    The ``reference_point`` rows/cols attrs index the full frame; they are shifted to the
    subset and the originals kept as ``frame_rows``/``frame_cols``.
    """
    if "spatial_ref" in ds.variables and "GeoTransform" in ds["spatial_ref"].attrs:
        raw = ds["spatial_ref"].attrs["GeoTransform"]
        numeric = isinstance(raw, np.ndarray) and raw.dtype.kind in "fi"
        vals = [float(v) for v in (np.ravel(raw) if numeric else _as_text(raw).split())]
        if len(vals) != 6:
            raise IngestError(f"Unexpected GeoTransform {raw!r}")
        x0, dx, rx, y0, ry, dy = vals
        new = [x0 + cols.start * dx + rows.start * rx, dx, rx, y0 + cols.start * ry + rows.start * dy, ry, dy]
        if "x" in ds.coords and "y" in ds.coords and ds.sizes["x"] and ds.sizes["y"]:
            cx = new[0] + 0.5 * dx + 0.5 * rx
            cy = new[3] + 0.5 * ry + 0.5 * dy
            if abs(cx - float(ds["x"].values[0])) > abs(dx) / 2 or abs(cy - float(ds["y"].values[0])) > abs(dy) / 2:
                raise IngestError(
                    "GeoTransform and x/y coordinates disagree by more than half a pixel in the "
                    "source product; refusing to write an inconsistent subset"
                )
        ds["spatial_ref"].attrs["GeoTransform"] = (
            np.asarray(new, dtype=raw.dtype) if numeric else " ".join(repr(v) for v in new)
        )
    if "reference_point" in ds.variables:
        attrs = ds["reference_point"].attrs
        if "rows" in attrs and "cols" in attrs:
            r = np.asarray(attrs["rows"]).astype(np.int64)
            c = np.asarray(attrs["cols"]).astype(np.int64)
            inside = bool(
                np.all((r >= rows.start) & (r < rows.stop) & (c >= cols.start) & (c < cols.stop))
            )
            attrs["frame_rows"], attrs["frame_cols"] = r, c
            attrs["rows"], attrs["cols"] = r - rows.start, c - cols.start
            attrs["vigil_reference_point_in_subset"] = np.int8(inside)


def _subset_group(ds, envelope, fallback: tuple[slice, slice] | None, full_shape):
    if not ("y" in ds.dims and "x" in ds.dims):
        return ds
    if "x" in ds.coords and "y" in ds.coords:
        rows, cols = index_window(ds["x"].values, ds["y"].values, envelope)
    elif fallback is not None and (ds.sizes["y"], ds.sizes["x"]) == full_shape:
        rows, cols = fallback
    else:
        raise IngestError("Group has y/x dimensions but no coordinates to subset by")
    sub = ds.isel(y=rows, x=cols)
    _update_georef(sub, rows, cols)
    return sub


def _encoding(ds) -> dict[str, dict]:
    enc: dict[str, dict] = {}
    for name, var in ds.data_vars.items():
        if var.ndim >= 2 and var.dtype.kind in "fiub":
            chunks = tuple([1] * (var.ndim - 2) + [min(CHUNK, s) for s in var.shape[-2:]])
            enc[name] = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": chunks}
    return enc


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # directories cannot be opened on Windows; nothing to do there
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _commit(tmp: Path, dst: Path) -> None:
    """Durably move a finished temp file into place (so a crash cannot leave a short file)."""
    _fsync_path(tmp)
    os.replace(tmp, dst)
    _fsync_path(dst.parent)


def write_subset(source: Any, dst: Path, bbox: Sequence[float]) -> dict[str, Any]:
    """Crop one DISP-S1 product to the bbox and write it to `dst` atomically.

    `source` is a local path or an open binary file object (e.g. fsspec over HTTPS).
    Data are copied raw (no CF decoding), so values, fill values and attributes are unchanged,
    except the georeferencing attrs that must describe the subset grid (see _update_georef).
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
        _update_georef(sub, rows, cols)
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
            out.attrs[BBOX_ATTR] = np.asarray(bbox, dtype=float)
            out.attrs["vigil_subset_rows"] = np.asarray([rows.start, rows.stop])
            out.attrs["vigil_subset_cols"] = np.asarray([cols.start, cols.stop])
        _commit(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"shape": shape, "rows": (rows.start, rows.stop), "cols": (cols.start, cols.stop)}


def _proxy_for(url: str) -> str | None:
    host = urlparse(url).hostname or ""
    if urllib.request.proxy_bypass(host):
        return None
    return urllib.request.getproxies().get(urlparse(url).scheme)


def open_remote(resolved: ResolvedUrl, block_size: int = 4 * MIB):
    """Binary file object over HTTPS that fetches only the byte ranges h5py asks for.

    aiohttp's trust_env is off so it never reads ~/.netrc; the proxy is passed explicitly.
    """
    import fsspec

    fs = fsspec.filesystem(
        "https",
        client_kwargs={"headers": resolved.headers, "trust_env": False},
        proxy=_proxy_for(resolved.url),
        skip_instance_cache=True,
    )
    return fs.open(resolved.url, mode="rb", block_size=block_size, cache_type="blockcache")


def _download(resolved: ResolvedUrl, dst: Path, session: requests.Session) -> int:
    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    n = 0
    try:
        with session.get(resolved.url, headers=resolved.headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            expected = r.headers.get("Content-Length")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * MIB):
                    f.write(chunk)
                    n += len(chunk)
            if expected is not None and "Content-Encoding" not in r.headers and int(expected) != n:
                raise IngestError(f"Truncated download: got {n} of {expected} bytes")
        _commit(tmp, dst)
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
    url: str, dst: str, bbox: Sequence[float], method: str = "range", retries: int = RETRIES
) -> FetchResult:
    """Download one product's bbox subset. Runs in a worker process."""
    dst_p = Path(dst)
    t0 = time.monotonic()
    nbytes: list[int | None] = [None]

    def attempt():
        if not url.startswith(("https://", "http://")):  # local file (reprocessing, tests)
            write_subset(Path(url), dst_p, bbox)
            nbytes[0] = 0
            return
        session = make_session()
        resolved = resolve_url(url, session, EarthdataAuth.from_env())
        if method == "range":
            with open_remote(resolved) as fobj:
                write_subset(fobj, dst_p, bbox)
                nbytes[0] = getattr(getattr(fobj, "cache", None), "total_requested_bytes", None)
        elif method == "full":
            full = dst_p.with_name(dst_p.name + ".download")
            try:
                nbytes[0] = _download(resolved, full, session)
                write_subset(full, dst_p, bbox)
            finally:
                full.unlink(missing_ok=True)
        else:
            raise FatalError(f"Unknown method {method!r}")

    try:
        with_retries(attempt, dst_p.name, retries)
        return FetchResult(dst_p.name, True, time.monotonic() - t0, nbytes[0])
    except FatalError as e:
        return FetchResult(dst_p.name, False, time.monotonic() - t0, None, _redact(e), fatal=True)
    except IngestError as e:
        return FetchResult(dst_p.name, False, time.monotonic() - t0, None, _redact(e))


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
                out.update_tags(**{STATIC_BBOX_TAG: json.dumps([float(v) for v in bbox])})
                for b in range(1, ds.count + 1):
                    out.update_tags(b, **ds.tags(b))
                    if ds.descriptions[b - 1]:
                        out.set_band_description(b, ds.descriptions[b - 1])
        _commit(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return int(window.height), int(window.width)


def check_static(path: Path, bbox: Sequence[float]) -> str | None:
    """None if `path` is a readable STATIC crop made for `bbox`, else why not."""
    import rasterio

    try:
        with rasterio.open(path) as ds:
            tag = ds.tags().get(STATIC_BBOX_TAG)
            ds.read(1, window=((0, 1), (0, 1)))
    except Exception as e:  # rasterio raises several error types for broken files
        return f"unreadable ({type(e).__name__})"
    if tag is None:
        return "no bbox record"
    if not _bbox_matches(json.loads(tag), bbox):
        return f"stale: cut for bbox {json.loads(tag)}"
    return None


def fetch_static(sg: StaticGranule, out_dir: Path, bbox: Sequence[float]) -> list[str]:
    """Crop every STATIC layer of the granule into out_dir. Skips layers already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    session = None
    for layer, url in sorted(sg.layer_urls.items()):
        name = Path(urlparse(url).path).name if "://" in url else Path(url).name
        dst = out_dir / name
        if dst.exists():
            why = check_static(dst, bbox)
            if why is None:
                done.append(layer)
                continue
            if why.startswith("stale"):
                raise IngestError(
                    f"{dst} was {why}, not the configured bbox {list(bbox)}. Move or delete "
                    f"{out_dir.parent} and rerun."
                )
            logger.warning("Re-fetching %s: %s", dst.name, why)
            dst.unlink()
        if not url.startswith(("https://", "http://")):
            crop_static(Path(url), dst, bbox)
        else:
            session = session or make_session()

            def attempt(url=url, dst=dst, name=name):
                full = out_dir / (name + ".download")
                try:
                    _download(resolve_url(url, session, EarthdataAuth.from_env()), full, session)
                    crop_static(full, dst, bbox)
                finally:
                    full.unlink(missing_ok=True)

            with_retries(attempt, f"STATIC {name}")
        done.append(layer)
    return done


# --------------------------------------------------------------------------------------
# Planning, summary, report
# --------------------------------------------------------------------------------------


def clean_partials(directory: Path) -> int:
    """Remove leftovers of interrupted writes. Returns how many were removed."""
    n = 0
    if directory.exists():
        for pattern in (f"*{PARTIAL_SUFFIX}", "*.download"):
            for p in directory.glob(pattern):
                p.unlink()
                n += 1
    return n


def check_subset(path: Path, bbox: Sequence[float]) -> str | None:
    """None if `path` is a complete subset made for `bbox`, else why not."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            missing = [v for v in REQUIRED_LAYERS if v not in f]
            corr = f.get("corrections")
            missing += [f"corrections/{v}" for v in REQUIRED_CORRECTIONS if corr is None or v not in corr]
            stored = f.attrs.get(BBOX_ATTR)
    except Exception as e:  # h5py raises OSError/KeyError and others for broken files
        return f"unreadable ({type(e).__name__})"
    if missing:
        return f"missing {missing}"
    if stored is None:
        return "no bbox record"
    if not _bbox_matches(stored, bbox):
        return f"stale: cut for bbox {[float(v) for v in np.ravel(stored)]}"
    return None


def plan_downloads(
    granules: Sequence[DispGranule],
    out_dir: Path,
    bbox: Sequence[float],
    repair: bool = True,
) -> tuple[list[DispGranule], list[DispGranule]]:
    """Split granules into (to_fetch, already_on_disk).

    A file on disk only counts if it is complete and was cut for `bbox`. Broken files are
    deleted (when `repair`) and re-fetched. Files cut for another bbox stop the run, so a
    bbox edit can never mix old and new crops.
    """
    to_fetch, existing, stale, broken = [], [], [], []
    for g in granules:
        path = out_dir / g.filename
        if not path.is_file():
            to_fetch.append(g)
            continue
        why = check_subset(path, bbox)
        if why is None:
            existing.append(g)
        elif why.startswith("stale"):
            stale.append((path.name, why))
        else:
            broken.append((path, why))
            to_fetch.append(g)
    if stale:
        raise IngestError(
            f"{len(stale)} subsets in {out_dir} were cut for a different bbox (e.g. {stale[0][0]}: "
            f"{stale[0][1]}); configured bbox is {list(bbox)}. Move or delete {out_dir} and rerun."
        )
    for path, why in broken:
        logger.warning("Re-fetching %s: %s", path.name, why)
        if repair:
            path.unlink()
    to_fetch.sort(key=lambda g: g.secondary_time)
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
    from shapely.geometry import box as shp_box

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
    n_in = coverage = None
    grid_key = inside = None
    per_epoch: list[float] = []
    valid_total = denom = 0
    for p in files:
        try:
            with xr.open_dataset(p, engine="h5netcdf") as ds:
                x, y = ds["x"].values, ds["y"].values
                key = (float(x[0]), float(x[-1]), x.size, float(y[0]), float(y[-1]), y.size)
                if grid_key is None:
                    grid_key = key
                    crs = dataset_crs(ds)
                    poly = bbox_polygon_projected(bbox, crs)
                    xx, yy = np.meshgrid(x, y)
                    inside = shapely.contains_xy(poly, xx, yy)
                    n_in = int(inside.sum())
                    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
                    grid = shp_box(
                        float(x.min()) - dx / 2, float(y.min()) - dy / 2,
                        float(x.max()) + dx / 2, float(y.max()) + dy / 2,
                    )
                    coverage = round(100.0 * grid.intersection(poly).area / poly.area, 2)
                    epsg = crs.to_epsg()
                    crs_str = f"EPSG:{epsg}" if epsg else crs.to_string()
                    shape = [int(y.size), int(x.size)]
                    pixel = [dx, dy]
                elif key != grid_key:
                    raise IngestError(
                        f"{p.name} is on a different grid from the other subsets in {out_dir}; "
                        "move or delete the folder and rerun"
                    )
                disp = ds["displacement"].values
                mask = ds["recommended_mask"].values
        except IngestError:
            raise
        except Exception as e:
            raise IngestError(f"Cannot read {p} ({type(e).__name__}: {e}); delete it and rerun") from e
        valid = np.isfinite(disp) & (mask == 1) & inside
        valid_total += int(valid.sum())
        denom += n_in
        per_epoch.append(100.0 * valid.sum() / n_in if n_in else 0.0)

    sdir = out_dir / "static"
    static_layers = [
        layer
        for layer in REQUIRED_STATIC_LAYERS
        if any(check_static(t, bbox) is None for t in sdir.glob(f"*_{layer}.tif"))
    ]
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
        bbox_coverage_pct=coverage,
        pct_valid=round(100.0 * valid_total / denom, 2) if denom else None,
        pct_valid_epoch_min=round(float(np.min(per_epoch)), 2) if per_epoch else None,
        pct_valid_epoch_median=round(float(np.median(per_epoch)), 2) if per_epoch else None,
        pct_valid_epoch_max=round(float(np.max(per_epoch)), 2) if per_epoch else None,
        valid_rule=VALID_RULE,
        static_layers=static_layers,
        bytes_on_disk=size,
    )


def _na(v: Any) -> str:
    return "n/a" if v is None else str(v)


def format_summary(region: Region, frames: Sequence[FrameSummary]) -> str:
    """The M0 printout. Every number carries its provenance tag."""
    w, s, e, n = region.bbox
    lines = [
        f"Region {region.name}: bbox W {w} S {s} E {e} N {n} (EPSG:4326, from config)",
        "Tags: MEASURED = read from the satellite products; DERIVED = computed from them.",
    ]
    for f in frames:
        lines += [
            "",
            f"Frame F{f.frame_id:05d} ({f.direction})",
            f"  epochs on disk        [MEASURED]: {f.n_epochs} (files {f.n_on_disk} of {f.n_expected} "
            f"listed by CMR, {f.n_missing} missing)",
            f"  date range            [MEASURED]: {_na(f.first_date)} to {_na(f.last_date)}",
            f"  reference dates       [MEASURED]: {f.n_reference_dates}",
            f"  largest gap           [DERIVED] : {_na(f.largest_gap_days)} days "
            f"({_na(f.largest_gap_start)} to {_na(f.largest_gap_end)})",
            f"  grid                  [MEASURED]: {_na(f.shape)} pixels of {_na(f.pixel_size_m)} m, {_na(f.crs)}",
            f"  pixels inside bbox    [DERIVED] : {_na(f.n_pixels_in_bbox)}; frame covers "
            f"{_na(f.bbox_coverage_pct)}% of the bbox",
            f"  % valid pixels        [DERIVED] : {_na(f.pct_valid)}% overall; per epoch min "
            f"{_na(f.pct_valid_epoch_min)}% / median {_na(f.pct_valid_epoch_median)}% / max "
            f"{_na(f.pct_valid_epoch_max)}%",
            f"  valid means                     : {f.valid_rule}",
            f"  STATIC layers                   : {', '.join(f.static_layers) or 'none'}",
        ]
    return "\n".join(lines)


def write_report(path: Path, region: Region, frames: Sequence[FrameSummary], run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = all(f.n_missing == 0 and f.n_on_disk > 0 for f in frames)
    body = [
        f"# M0 ingest report: {region.name}",
        "",
        f"Generated {run['finished_utc']} by `{run.get('command', 'python -m pipeline.ingest')}`.",
        "",
        f"- Region: {region.description}",
        f"- Status: {'all configured frames complete' if complete else 'INCOMPLETE: see missing counts below'}.",
        f"- This run: {run['fetched']} subsets downloaded, {run['skipped']} already on disk, "
        f"{run['failed']} failed; method `{run['method']}`; {run['seconds']:.0f} s; "
        f"frames processed: {', '.join(str(f) for f in run.get('frames_processed', [])) or 'none'}.",
    ]
    if run.get("median_mib_per_file") is not None:
        body.append(
            f"- Transfer [MEASURED]: median {run['median_mib_per_file']:.1f} MiB read per product "
            f"(range reads), {run['median_seconds_per_file']:.1f} s per product."
        )
    body += ["", "```", format_summary(region, frames), "```", ""]
    path.write_text("\n".join(body))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _worker_init() -> None:
    """Workers ignore Ctrl-C; the parent decides when to stop and terminates them."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_pool(pool: cf.ProcessPoolExecutor) -> None:
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=10)


def _handle_result(res: FetchResult, results: list[FetchResult]) -> None:
    results.append(res)
    if res.fatal:
        raise IngestError(f"{res.filename}: {res.error}")
    if not res.ok:
        logger.error("FAILED %s: %s", res.filename, res.error)


def run_frame_downloads(
    jobs: list[tuple[str, str]],
    bbox: Sequence[float],
    method: str,
    workers: int,
) -> list[FetchResult]:
    """Fetch subsets. At most `workers` jobs are ever in flight, so stopping is immediate."""
    results: list[FetchResult] = []
    if not jobs:
        return results
    t0 = time.monotonic()

    def progress(i: int) -> None:
        if i % 10 == 0 or i == len(jobs):
            rate = i / max(time.monotonic() - t0, 1e-9)
            logger.info("  %d/%d done (%.2f files/s, ~%.0f min left)", i, len(jobs), rate,
                        (len(jobs) - i) / rate / 60 if rate else float("nan"))

    if workers <= 1:  # in-process: simpler to debug, used by most tests
        for i, (url, dst) in enumerate(jobs, 1):
            _handle_result(fetch_subset(url, dst, tuple(bbox), method), results)
            progress(i)
        return results

    pool = cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    )
    queue = iter(jobs)
    inflight: set[cf.Future] = set()

    def submit_next() -> None:
        job = next(queue, None)
        if job is not None:
            inflight.add(pool.submit(fetch_subset, job[0], job[1], tuple(bbox), method))

    try:
        for _ in range(workers):
            submit_next()
        while inflight:
            done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                inflight.discard(fut)
                _handle_result(fut.result(), results)
                progress(len(results))
                submit_next()
    except BaseException:
        _terminate_pool(pool)
        raise
    pool.shutdown(wait=True)
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
    command: str | None = None,
) -> tuple[list[FrameSummary], dict]:
    """Fetch the selected frames, then summarize every configured frame (manifest + report)."""
    session = session or make_session()
    selected = [f for f in region.frames if frames is None or f.frame_id in frames]
    if not selected:
        raise IngestError(f"No frames selected (config has {[f.frame_id for f in region.frames]})")
    t0 = time.monotonic()
    run: dict[str, Any] = {
        "fetched": 0, "skipped": 0, "failed": 0, "method": method,
        "frames_processed": [f.frame_id for f in selected],
        "command": command or f"python -m pipeline.ingest --region {region.name}",
    }
    transfer_mib: list[float] = []
    seconds: list[float] = []
    failures: list[FetchResult] = []
    expected: dict[int, int] = {}

    for spec in region.frames:
        granules = search_disp(spec.frame_id, region.product_version, session)
        expected[spec.frame_id] = len(granules)
        if spec not in selected:
            continue
        out = frame_dir(data_dir, region.name, spec.frame_id)
        if dry_run:
            to_fetch, existing = plan_downloads(granules, out, region.bbox, repair=False)
        else:
            out.mkdir(parents=True, exist_ok=True)
            removed = clean_partials(out) + clean_partials(out / "static")
            if removed:
                logger.info("F%05d: removed %d partial files from an interrupted run", spec.frame_id, removed)
            to_fetch, existing = plan_downloads(granules, out, region.bbox)
        if limit is not None:
            to_fetch = to_fetch[:limit]
        logger.info(
            "F%05d (%s): %d granules in CMR, %d complete on disk, %d to fetch",
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
        for r in run_frame_downloads(jobs, region.bbox, method, workers):
            if r.ok:
                run["fetched"] += 1
                seconds.append(r.seconds)
                if r.bytes_transferred:
                    transfer_mib.append(r.bytes_transferred / MIB)
            else:
                run["failed"] += 1
                failures.append(r)

    run["seconds"] = time.monotonic() - t0
    run["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run["median_mib_per_file"] = float(np.median(transfer_mib)) if transfer_mib else None
    run["median_seconds_per_file"] = float(np.median(seconds)) if seconds else None
    if dry_run:
        return [], run

    # Summarize every configured frame, so a --frames run never drops the others from the
    # manifest (the ingest -> stack contract) or the report.
    summaries = [
        summarize_frame(
            frame_dir(data_dir, region.name, s.frame_id), region.bbox, s.frame_id, s.direction,
            expected[s.frame_id],
        )
        for s in region.frames
    ]
    manifest = IngestManifest(
        region=region.name,
        bbox=list(region.bbox),
        product_version=region.product_version,
        created_utc=run["finished_utc"],
        frames=summaries,
    )
    mpath = manifest_path(data_dir, region.name)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest.to_dict(), indent=2))
    total_mib = sum(f.bytes_on_disk for f in summaries) / MIB
    logger.info("Disk used under %s: %.1f MiB", mpath.parent, total_mib)
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
    ap.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--frames", type=lambda s: [int(v) for v in s.split(",")], default=None,
                    help="comma-separated frame IDs to fetch (default: all frames in the config)")
    ap.add_argument("--method", choices=["range", "full"], default="range",
                    help="range: read only the bbox chunks over HTTPS; full: download then crop")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N new products per frame")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--find-frames", action="store_true", help="list frames covering the bbox and exit")
    ap.add_argument("--no-report", action="store_true", help="do not write <reports-dir>/m0_ingest_<region>.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
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
            print("Tags: frame, direction, path, granules, dates MEASURED (CMR metadata); bbox cover DERIVED.")
            print(f"{'frame':>6} {'direction':<11}{'path':>5} {'granules':>9} {'first':<11}{'last':<11}{'bbox cover':>10}")
            for f in rows:
                print(f"{f['frame_id']:>6} {f['direction']:<11}{f['path'] or '':>5} {f['n_granules']:>9} "
                      f"{f['first']:<11}{f['last']:<11}{100 * f['coverage']:>9.1f}%")
            return 0
        report = None if args.no_report else args.reports_dir / f"m0_ingest_{region.name}.md"
        summaries, _ = run_ingest(
            region,
            data_dir=args.data_dir,
            frames=args.frames,
            method=args.method,
            workers=args.workers,
            limit=args.limit,
            dry_run=args.dry_run,
            report_path=report,
            command="python -m pipeline.ingest " + shlex.join(argv),
        )
        if summaries:
            print(format_summary(region, summaries))
        return 0
    except KeyboardInterrupt:
        logger.error("STOPPED: interrupted. Finished files are kept; rerun to resume.")
        return 130
    except IngestError as e:
        logger.error("STOPPED: %s", _redact(e))
        return 2
    except (requests.RequestException, OSError) as e:
        logger.error("STOPPED: %s: %s", type(e).__name__, _redact(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
