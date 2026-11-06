"""Tests for pipeline.ingest on synthetic data. No network (a local 127.0.0.1 server at most)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import requests
import xarray as xr
from pyproj import CRS

from pipeline import ingest
from pipeline.ingest import (
    CMR_GRANULES_URL,
    DispGranule,
    EarthdataAuth,
    EarthdataAuthError,
    FrameSpec,
    IngestError,
    MissingLayerError,
    Region,
    StaticGranule,
)
from pipeline.schemas import PARTIAL_SUFFIX, frame_dir, manifest_path, static_dir

from .conftest import EPSG, HOUSTON_BBOX, disp_name, write_disp_product, write_static_tif

UTC = timezone.utc
SECS = ["20161016T122243Z", "20161028T122243Z", "20161109T122243Z"]
STATIC_LAYERS = (("line_of_sight_enu", 3), ("dem", 1), ("layover_shadow_mask", 1))


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(ingest, "_sleep", lambda s: None)


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """No real ~/.netrc or Earthdata env vars leak into tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for v in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "NETRC"):
        monkeypatch.delenv(v, raising=False)
    return home


# ---------------------------------------------------------------- helpers ----------


def _granules_for(paths, base_url=None):
    out = []
    for p in paths:
        info = ingest.parse_disp_name(p.name)
        url = f"{base_url}/{p.name}" if base_url else str(p)
        out.append(DispGranule(granule_ur=p.stem, url=url, **info))
    return out


def _make_sources(src: Path, grid, frame_id=38238, secs=SECS, static=True, base_url=None):
    """Synthetic products (+ STATIC layers) in `src`. Returns (granules, static_urls)."""
    x, y = grid
    paths = []
    for i, sec in enumerate(secs):
        p = src / disp_name(frame_id, "20160712T122240Z", sec)
        write_disp_product(p, x, y, seed=i)
        paths.append(p)
    static_urls = {}
    if static:
        for layer, bands in STATIC_LAYERS:
            p = src / f"OPERA_L3_DISP-S1-STATIC_F{frame_id:05d}_20140403_S1A_v1.0_{layer}.tif"
            write_static_tif(p, x, y, bands=bands)
            static_urls[layer] = f"{base_url}/{p.name}" if base_url else str(p)
    return _granules_for(paths, base_url), static_urls


def _patch_search(monkeypatch, by_frame: dict):
    """by_frame: frame_id -> (granules, static_urls)."""
    monkeypatch.setattr(ingest, "search_disp", lambda fid, ver, session: list(by_frame[fid][0]))
    monkeypatch.setattr(
        ingest, "search_static",
        lambda fid, session: StaticGranule("S", fid, "20140403", "1.0", by_frame[fid][1]),
    )


def _region(bbox=HOUSTON_BBOX, frames=(38238,)):
    return Region("testcity", tuple(bbox), tuple(FrameSpec(f, "descending") for f in frames))


# ---------------------------------------------------------------- config ----------


def test_houston_region_config_is_sane():
    r = ingest.load_region("houston")
    w, s, e, n = r.bbox
    assert w < e and s < n
    width_km = (e - w) * 111.32 * np.cos(np.radians((s + n) / 2))
    height_km = (n - s) * 110.57
    assert 25 < width_km < 35 and 25 < height_km < 35  # ~30 x 30 km (spine section 9)
    assert {f.direction for f in r.frames} == {"ascending", "descending"}
    assert len({f.frame_id for f in r.frames}) == len(r.frames)


def test_load_region_rejects_bad_bbox(tmp_path):
    cfg = tmp_path / "regions.yaml"
    cfg.write_text("x:\n  bbox: [-95.5, 29.7, -95.8, 30.0]\n  frames: []\n")
    with pytest.raises(IngestError, match="invalid bbox"):
        ingest.load_region("x", cfg)
    with pytest.raises(IngestError, match="not in"):
        ingest.load_region("nope", cfg)


# ---------------------------------------------------------------- names and CMR ----


def test_parse_disp_name():
    info = ingest.parse_disp_name(
        "OPERA_L3_DISP-S1_IW_F38238_VV_20160712T122240Z_20161016T122243Z_v1.0_20251023T225135Z.nc"
    )
    assert info["frame_id"] == 38238
    assert info["reference_time"] == datetime(2016, 7, 12, 12, 22, 40, tzinfo=UTC)
    assert info["secondary_time"] == datetime(2016, 10, 16, 12, 22, 43, tzinfo=UTC)
    assert info["version"] == "1.0"
    with pytest.raises(IngestError):
        ingest.parse_disp_name("something_else.nc")


def _umm(name: str, frame: int = 38238) -> dict:
    base = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L3_DISP-S1_V1"
    return {
        "GranuleUR": name,
        "AdditionalAttributes": [
            {"Name": "FRAME_NUMBER", "Values": [str(frame)]},
            {"Name": "ASCENDING_DESCENDING", "Values": ["DESCENDING"]},
        ],
        "RelatedUrls": [
            {"URL": f"{base}/{name}/{name}.nc", "Type": "GET DATA"},
            {"URL": f"s3://bucket/{name}/{name}.nc", "Type": "GET DATA VIA DIRECT ACCESS"},
            {"URL": f"{base}/{name}/{name}.zarr.json.gz", "Type": "GET DATA"},
            {"URL": f"{base}/{name}/{name}.png", "Type": "GET RELATED VISUALIZATION"},
        ],
    }


def test_disp_granule_from_umm_picks_https_nc():
    name = disp_name(38238, "20160712T122240Z", "20161016T122243Z")[:-3]
    g = ingest.disp_granule_from_umm(_umm(name))
    assert g.url.startswith("https://") and g.url.endswith(".nc")
    assert g.filename == name + ".nc"
    assert g.frame_id == 38238


def test_dedupe_keeps_latest_production():
    a = ingest.disp_granule_from_umm(_umm(disp_name(1, "20160101T000000Z", "20160201T000000Z", "20250101T000000Z")[:-3]))
    b = ingest.disp_granule_from_umm(_umm(disp_name(1, "20160101T000000Z", "20160201T000000Z", "20251101T000000Z")[:-3]))
    c = ingest.disp_granule_from_umm(_umm(disp_name(1, "20160101T000000Z", "20160115T000000Z")[:-3]))
    out = ingest.dedupe_latest([a, b, c])
    assert [g.secondary_time.day for g in out] == [15, 1]  # sorted by secondary time
    assert out[1].production_time == datetime(2025, 11, 1, tzinfo=UTC)


class _Resp:
    def __init__(self, status=200, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        return self._payload

    def close(self):
        pass


class _CmrSession:
    def __init__(self, pages, hits, fail_first=0):
        self.pages = pages
        self.hits = hits
        self.fail_first = fail_first
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert url == CMR_GRANULES_URL
        if self.fail_first:
            self.fail_first -= 1
            raise requests.ConnectionError("connection reset")
        self.calls.append(dict(headers or {}))
        i = len(self.calls) - 1
        h = {"CMR-Hits": str(self.hits)}
        if i + 1 < len(self.pages):
            h["CMR-Search-After"] = f"token{i}"
        return _Resp(200, h, {"items": self.pages[i]})


def test_cmr_search_follows_search_after():
    s = _CmrSession([[{"a": 1}, {"a": 2}], [{"a": 3}]], hits=3)
    items = ingest.cmr_search({"short_name": "X"}, s, page_size=2)
    assert len(items) == 3
    assert s.calls[1] == {"CMR-Search-After": "token0"}


def test_cmr_search_refuses_truncated_listing():
    s = _CmrSession([[{"a": 1}]], hits=5)
    with pytest.raises(IngestError, match="reported 5 hits"):
        ingest.cmr_search({"short_name": "X"}, s, page_size=2)


def test_cmr_search_retries_network_errors_then_stops_cleanly():
    s = _CmrSession([[{"a": 1}]], hits=1, fail_first=2)
    assert len(ingest.cmr_search({"short_name": "X"}, s)) == 1
    s = _CmrSession([[{"a": 1}]], hits=1, fail_first=5)
    with pytest.raises(IngestError, match="failed after 3 attempts"):
        ingest.cmr_search({"short_name": "X"}, s)


def test_static_granule_from_umm_and_required_layers():
    ur = "OPERA_L3_DISP-S1-STATIC_F08882_20140403_S1A_v1.0"
    base = f"https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L3_DISP-S1-STATIC_V1/{ur}"
    umm = {
        "GranuleUR": ur,
        "RelatedUrls": [
            {"URL": f"{base}/{ur}_{layer}.tif", "Type": "GET DATA"}
            for layer in ("dem", "layover_shadow_mask", "line_of_sight_enu")
        ] + [{"URL": f"s3://b/{ur}_dem.tif", "Type": "GET DATA VIA DIRECT ACCESS"}],
    }
    sg = ingest.static_granule_from_umm(umm)
    assert sg.frame_id == 8882
    assert set(sg.layer_urls) == {"dem", "layover_shadow_mask", "line_of_sight_enu"}
    assert all(u.startswith("https://") for u in sg.layer_urls.values())


# ---------------------------------------------------------------- auth -------------


DATA_URL = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L3_DISP-S1_V1/x/x.nc"
URS_URL = "https://urs.earthdata.nasa.gov/oauth/authorize?client_id=abc&state=s"
SIGNED_URL = "https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/x.nc?X-Amz-Signature=sig"


def test_auth_headers_are_scoped_to_earthdata_hosts():
    tok = EarthdataAuth(token="T")
    assert tok.headers_for(DATA_URL) == {"Authorization": "Bearer T"}
    assert tok.headers_for(SIGNED_URL) == {}
    assert tok.headers_for(URS_URL) == {}
    basic = EarthdataAuth(username="u", password="p")
    assert basic.headers_for(URS_URL)["Authorization"].startswith("Basic ")
    assert basic.headers_for(DATA_URL) == {}
    assert basic.headers_for(SIGNED_URL) == {}
    assert "p" not in repr(basic)


def test_auth_from_env_and_netrc(monkeypatch, _isolated_home):
    assert EarthdataAuth.from_env() == EarthdataAuth()
    nrc = _isolated_home / ".netrc"
    nrc.write_text("machine urs.earthdata.nasa.gov login alice password secret\n")
    nrc.chmod(0o600)
    a = EarthdataAuth.from_env()
    assert (a.username, a.password, a.token) == ("alice", "secret", None)
    monkeypatch.setenv("EARTHDATA_USERNAME", "bob")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")
    assert EarthdataAuth.from_env().username == "bob"  # env vars win over netrc
    monkeypatch.setenv("EARTHDATA_TOKEN", "TOK")
    assert EarthdataAuth.from_env().token == "TOK"


def test_netrc_with_loose_permissions_is_used_and_warned(_isolated_home, caplog):
    nrc = _isolated_home / ".netrc"
    nrc.write_text("machine urs.earthdata.nasa.gov login alice password secret\n")
    nrc.chmod(0o644)
    with caplog.at_level(logging.WARNING, logger="vigil.ingest"):
        a = EarthdataAuth.from_env()
    assert a.username == "alice"
    assert "chmod 600" in caplog.text


def test_netrc_env_var_and_parse_errors(monkeypatch, tmp_path):
    custom = tmp_path / "my_netrc"
    custom.write_text("machine urs.earthdata.nasa.gov login carol password pw\n")
    custom.chmod(0o600)
    monkeypatch.setenv("NETRC", str(custom))
    assert EarthdataAuth.from_env().username == "carol"
    custom.write_text("machine urs.earthdata.nasa.gov login carol password pw\nbogus\n")  # malformed
    with pytest.raises(EarthdataAuthError, match="Could not read"):
        EarthdataAuth.from_env()


def test_session_never_reads_netrc_itself(_isolated_home):
    """A netrc 'default' entry must never reach the signed S3 URL (or any non-Earthdata host)."""
    nrc = _isolated_home / ".netrc"
    nrc.write_text("default login alice password secret\n")
    nrc.chmod(0o600)
    s = ingest.make_session()
    prepared = s.prepare_request(requests.Request("GET", SIGNED_URL))
    assert "Authorization" not in prepared.headers
    plain = requests.Session().prepare_request(requests.Request("GET", SIGNED_URL))
    assert "Authorization" in plain.headers  # proves the fixture would leak with plain requests


class _RedirectSession:
    def __init__(self, routes):
        self.routes = routes  # list of (url_prefix, status, location)
        self.calls = []

    def get(self, url, headers=None, allow_redirects=True, stream=False, timeout=None):
        self.calls.append((url, dict(headers or {})))
        for prefix, status, loc in self.routes:
            if url.startswith(prefix):
                return _Resp(status, {"Location": loc} if loc else {})
        raise AssertionError(f"unexpected url {url}")


def test_resolve_url_token_goes_straight_to_signed_url():
    s = _RedirectSession([(DATA_URL, 307, SIGNED_URL)])
    r = ingest.resolve_url(DATA_URL, s, EarthdataAuth(token="T"))
    assert r.url == SIGNED_URL and r.headers == {}
    assert s.calls[0][1]["Authorization"] == "Bearer T"


def test_resolve_url_login_flow_sends_basic_only_to_urs():
    login_back = "https://cumulus.asf.earthdatacloud.nasa.gov/login?code=c&state=s"
    s = _RedirectSession(
        [
            (DATA_URL, 302, URS_URL),
            ("https://urs.earthdata.nasa.gov/", 302, login_back),
            ("https://cumulus.asf.earthdatacloud.nasa.gov/login", 307, SIGNED_URL),
        ]
    )
    r = ingest.resolve_url(DATA_URL, s, EarthdataAuth(username="u", password="p"))
    assert r.url == SIGNED_URL and r.headers == {}
    sent = {url.split("?")[0]: h.get("Authorization", "") for url, h in s.calls}
    assert sent["https://urs.earthdata.nasa.gov/oauth/authorize"].startswith("Basic ")
    assert sent[DATA_URL] == "" and sent["https://cumulus.asf.earthdatacloud.nasa.gov/login"] == ""


def test_resolve_url_without_credentials_stops_with_help():
    s = _RedirectSession([(DATA_URL, 302, URS_URL)])
    with pytest.raises(EarthdataAuthError, match="authorized"):
        ingest.resolve_url(DATA_URL, s, EarthdataAuth())


def test_resolve_url_urs_page_means_app_not_authorized():
    s = _RedirectSession([(DATA_URL, 302, URS_URL), ("https://urs.earthdata.nasa.gov/", 200, None)])
    with pytest.raises(EarthdataAuthError, match="authorize the application"):
        ingest.resolve_url(DATA_URL, s, EarthdataAuth(username="u", password="p"))


def test_resolve_url_401_is_auth_error():
    s = _RedirectSession([(DATA_URL, 401, None)])
    with pytest.raises(EarthdataAuthError):
        ingest.resolve_url(DATA_URL, s, EarthdataAuth(token="bad"))


def test_redact_strips_signed_query_strings():
    msg = ingest._redact(f"404 Client Error for url: {SIGNED_URL}&X-Amz-Security-Token=abc")
    assert "sig" not in msg and "abc" not in msg and "amazonaws.com/x.nc?<redacted>" in msg


# ---------------------------------------------------------------- geometry ---------


def test_index_window_covers_bbox_on_both_axis_orders(grid):
    x, y = grid
    env = ingest.bbox_envelope(HOUSTON_BBOX, CRS.from_epsg(EPSG))
    for yy in (y, y[::-1]):  # north-up and south-up grids
        rows, cols = ingest.index_window(x, yy, env, pad=0)
        xs, ys = x[cols], yy[rows]
        half = (x[1] - x[0]) / 2
        assert xs.min() - half <= env[0] and xs.max() + half >= env[2]
        assert ys.min() - half <= env[1] and ys.max() + half >= env[3]
        # minimal: dropping an edge pixel would uncover the envelope
        assert xs.min() + half > env[0] and xs.max() - half < env[2]


def test_index_window_pad_and_clip(grid):
    x, y = grid
    env = (x[0] - 10_000, y[-1] - 10_000, x[2], y[-3])  # hangs off two edges
    rows, cols = ingest.index_window(x, y, env, pad=1)
    assert cols.start == 0 and cols.stop == 4
    assert rows.stop == y.size


def test_index_window_no_overlap_raises(grid):
    x, y = grid
    with pytest.raises(IngestError, match="does not overlap"):
        ingest.index_window(x, y, (0, 0, 10, 10))


def test_envelope_contains_projected_bbox_corners():
    from pyproj import Transformer

    env = ingest.bbox_envelope(HOUSTON_BBOX, CRS.from_epsg(EPSG))
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    w, s, e, n = HOUSTON_BBOX
    for lon, lat in [(w, s), (w, n), (e, s), (e, n), ((w + e) / 2, n)]:
        px, py = tr.transform(lon, lat)
        assert env[0] <= px <= env[2] and env[1] <= py <= env[3]


# ---------------------------------------------------------------- subsetting -------


def test_write_subset_roundtrip(tmp_path, grid):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    arrays = write_disp_product(src, x, y, seed=1)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()
    info = ingest.write_subset(src, dst, HOUSTON_BBOX)
    assert dst.exists() and not dst.with_name(dst.name + PARTIAL_SUFFIX).exists()
    (r0, r1), (c0, c1) = info["rows"], info["cols"]
    assert 0 < r1 - r0 < y.size and 0 < c1 - c0 < x.size  # actually cropped

    with xr.open_dataset(dst, engine="h5netcdf", decode_cf=False) as ds:
        np.testing.assert_array_equal(ds["x"].values, x[c0:c1])
        np.testing.assert_array_equal(ds["y"].values, y[r0:r1])
        for name in ingest.REQUIRED_LAYERS:
            np.testing.assert_array_equal(ds[name].values, arrays[name][r0:r1, c0:c1])
            assert ds[name].dtype == arrays[name].dtype
        assert ds["displacement"].attrs["grid_mapping"] == "spatial_ref"
        assert "crs_wkt" in ds["spatial_ref"].attrs
        assert ds.attrs["product_version"] == "1.0"
    with xr.open_dataset(dst, engine="h5netcdf", group="corrections") as corr:
        assert corr["ionospheric_delay"].shape == (r1 - r0, c1 - c0)
        assert float(corr["perpendicular_baseline"]) == 42.0
    with h5py.File(dst) as f:
        assert f["metadata/platform_id"][()] == b"S1A"
        assert int(f["identification/frame_id"][()]) == 38238
        assert list(f.attrs["vigil_subset_rows"]) == [r0, r1]
        assert list(f.attrs[ingest.BBOX_ATTR]) == list(HOUSTON_BBOX)


def test_write_subset_rewrites_georeferencing_for_gdal(tmp_path, grid):
    """GDAL reads spatial_ref.GeoTransform; a stale full-frame value would misplace the subset."""
    import rasterio

    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y, reference_rowcol=(10, 12))
    dst = tmp_path / ("sub_" + src.name)
    info = ingest.write_subset(src, dst, HOUSTON_BBOX)
    (r0, _), (c0, _) = info["rows"], info["cols"]
    with xr.open_dataset(dst, engine="h5netcdf") as ds:
        sx, sy = ds["x"].values, ds["y"].values
    for var in ("displacement", "/corrections/ionospheric_delay"):
        with rasterio.open(f'netcdf:"{dst}":{var}') as r:
            assert r.transform.c == pytest.approx(sx[0] - 300.0)  # left edge = first centre - dx/2
            assert r.transform.f == pytest.approx(sy[0] + 300.0)  # top edge (north-up)
    with xr.open_dataset(dst, engine="h5netcdf", group="corrections") as corr:
        a = corr["reference_point"].attrs
        one = lambda v: np.atleast_1d(v).tolist()  # 1-element attrs read back as scalars
        assert one(a["rows"]) == [10 - r0] and one(a["cols"]) == [12 - c0]
        assert one(a["frame_rows"]) == [10] and one(a["frame_cols"]) == [12]
        assert int(a["vigil_reference_point_in_subset"]) == int(r0 <= 10 and c0 <= 12)


def test_write_subset_from_file_object(tmp_path, grid):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y)
    dst = tmp_path / ("copy_" + src.name)
    with open(src, "rb") as fobj:  # same code path as the HTTPS range-read file object
        ingest.write_subset(fobj, dst, HOUSTON_BBOX)
    with xr.open_dataset(dst, engine="h5netcdf") as ds:
        assert ds["displacement"].ndim == 2


@pytest.mark.parametrize(
    "drop,drop_corr",
    [(("recommended_mask",), ()), (("temporal_coherence",), ()), ((), ("solid_earth_tide",))],
)
def test_write_subset_missing_layer_writes_nothing(tmp_path, grid, drop, drop_corr):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y, drop=drop, drop_corrections=drop_corr)
    dst = tmp_path / "out.nc"
    with pytest.raises(MissingLayerError):
        ingest.write_subset(src, dst, HOUSTON_BBOX)
    assert not dst.exists() and not dst.with_name(dst.name + PARTIAL_SUFFIX).exists()


def test_write_subset_is_atomic_when_the_final_rename_fails(tmp_path, grid, monkeypatch):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()

    def boom(a, b):
        raise OSError("disk full")

    monkeypatch.setattr(ingest.os, "replace", boom)
    with pytest.raises(OSError):
        ingest.write_subset(src, dst, HOUSTON_BBOX)
    assert list(dst.parent.iterdir()) == []  # neither a final nor a partial file


def test_fetch_subset_reports_missing_layer_as_fatal(tmp_path, grid):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y, drop=("displacement",))
    res = ingest.fetch_subset(str(src), str(tmp_path / "o.nc"), HOUSTON_BBOX)
    assert not res.ok and res.fatal and "displacement" in res.error


def test_crop_static_keeps_grid_values_band_names_and_bbox_tag(tmp_path, grid):
    import rasterio

    x, y = grid
    src = tmp_path / "OPERA_L3_DISP-S1-STATIC_F38238_20140403_S1A_v1.0_line_of_sight_enu.tif"
    data = write_static_tif(src, x, y)
    dst = tmp_path / "crop.tif"
    h, w = ingest.crop_static(src, dst, HOUSTON_BBOX)
    env = ingest.bbox_envelope(HOUSTON_BBOX, CRS.from_epsg(EPSG))
    with rasterio.open(dst) as ds:
        assert (ds.height, ds.width) == (h, w) and ds.count == 3
        b = ds.bounds
        assert b.left <= env[0] and b.bottom <= env[1] and b.right >= env[2] and b.top >= env[3]
        assert ds.descriptions == ("east", "north", "up")
        assert json.loads(ds.tags()[ingest.STATIC_BBOX_TAG]) == list(HOUSTON_BBOX)
        col0 = int(round((b.left - (x[0] - (x[1] - x[0]) / 2)) / (x[1] - x[0])))
        row0 = int(round(((y[0] + (x[1] - x[0]) / 2) - b.top) / (x[1] - x[0])))
        np.testing.assert_array_equal(ds.read(), data[:, row0 : row0 + h, col0 : col0 + w])
    assert ingest.check_static(dst, HOUSTON_BBOX) is None
    assert ingest.check_static(dst, (-95.9, 29.7, -95.5, 30.0)).startswith("stale")


# ---------------------------------------------------------------- plan, summary ----


def test_plan_downloads_classifies_files_on_disk(tmp_path, grid):
    x, y = grid
    src = tmp_path / "src"
    src.mkdir()
    gs, _ = _make_sources(src, grid, static=False)
    out = tmp_path / "F38238"
    out.mkdir()
    ingest.write_subset(Path(gs[0].url), out / gs[0].filename, HOUSTON_BBOX)  # complete
    (out / gs[1].filename).write_bytes(b"truncated")  # broken final file
    (out / (gs[2].filename + PARTIAL_SUFFIX)).write_bytes(b"half")  # interrupted write

    to_fetch, existing = ingest.plan_downloads(gs, out, HOUSTON_BBOX, repair=False)
    assert [g.filename for g in existing] == [gs[0].filename]
    assert [g.filename for g in to_fetch] == [gs[1].filename, gs[2].filename]
    assert (out / gs[1].filename).exists()  # repair=False (dry run) deletes nothing

    ingest.plan_downloads(gs, out, HOUSTON_BBOX, repair=True)
    assert not (out / gs[1].filename).exists()
    assert ingest.clean_partials(out) == 1

    with pytest.raises(IngestError, match="different bbox"):
        ingest.plan_downloads(gs, out, (-95.9, 29.7, -95.5, 30.0))


def test_summarize_frame_counts_epochs_dates_and_valid_pixels(tmp_path, grid):
    x, y = grid
    out = tmp_path / "F38238"
    out.mkdir()
    pairs = [
        ("20160712T122240Z", "20161016T122243Z"),
        ("20160712T122240Z", "20161028T122243Z"),
        ("20161028T122243Z", "20170315T122243Z"),  # new reference date (ministack)
    ]
    for i, (ref, sec) in enumerate(pairs):
        src = tmp_path / disp_name(38238, ref, sec)
        write_disp_product(src, x, y, seed=i)
        ingest.write_subset(src, out / src.name, HOUSTON_BBOX)
    s = ingest.summarize_frame(out, HOUSTON_BBOX, 38238, "descending", n_expected=4)
    assert (s.frame_id, s.direction) == (38238, "descending")
    assert (s.n_expected, s.n_epochs, s.n_on_disk, s.n_missing) == (4, 3, 3, 1)
    assert s.n_reference_dates == 2
    assert (s.first_date, s.last_date) == ("2016-10-16", "2017-03-15")
    assert (s.largest_gap_days, s.largest_gap_start, s.largest_gap_end) == (138.0, "2016-10-28", "2017-03-15")
    assert s.crs == f"EPSG:{EPSG}" and s.pixel_size_m == [600.0, 600.0]
    with xr.open_dataset(next(out.glob("*.nc")), engine="h5netcdf") as ds:
        assert s.shape == [ds.sizes["y"], ds.sizes["x"]]
    assert s.bbox_coverage_pct == 100.0
    assert s.static_layers == [] and s.valid_rule == ingest.VALID_RULE and s.bytes_on_disk > 0

    # Recompute % valid independently, inside the bbox polygon only
    import shapely

    poly = ingest.bbox_polygon_projected(HOUSTON_BBOX, CRS.from_epsg(EPSG))
    total = valid = 0
    per_epoch = []
    for p in sorted(out.glob("*.nc")):
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            xx, yy = np.meshgrid(ds.x.values, ds.y.values)
            inside = shapely.contains_xy(poly, xx, yy)
            ok = np.isfinite(ds.displacement.values) & (ds.recommended_mask.values == 1) & inside
            total += inside.sum()
            valid += ok.sum()
            per_epoch.append(100 * ok.sum() / inside.sum())
    assert s.pct_valid == round(100 * valid / total, 2)
    assert 60 < s.pct_valid < 85  # ~80% mask-good x ~95% finite
    assert s.n_pixels_in_bbox == total // 3
    assert s.pct_valid_epoch_min == round(min(per_epoch), 2)
    assert s.pct_valid_epoch_median == round(float(np.median(per_epoch)), 2)
    assert s.pct_valid_epoch_max == round(max(per_epoch), 2)


def test_summarize_frame_names_an_unreadable_file(tmp_path, grid):
    out = tmp_path / "F38238"
    out.mkdir()
    bad = out / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    bad.write_bytes(b"not hdf5")
    with pytest.raises(IngestError, match="Cannot read .*delete it and rerun"):
        ingest.summarize_frame(out, HOUSTON_BBOX, 38238, "descending", 1)


# ---------------------------------------------------------------- end to end -------


def test_run_ingest_end_to_end_then_rerun_skips(tmp_path, monkeypatch, grid):
    """M0 acceptance logic offline: subsets + STATIC + summary, and a rerun fetches nothing."""
    src = tmp_path / "remote"
    src.mkdir()
    gs, static_urls = _make_sources(src, grid)
    _patch_search(monkeypatch, {38238: (gs, static_urls)})
    region = _region()
    data = tmp_path / "data"
    report = tmp_path / "reports" / "m0.md"

    summaries, run = ingest.run_ingest(region, data_dir=data, workers=1, report_path=report, session=object())
    assert run["fetched"] == 3 and run["skipped"] == 0 and run["failed"] == 0
    (s,) = summaries
    assert s.n_epochs == 3 and s.n_missing == 0
    assert s.static_layers == ["line_of_sight_enu", "dem", "layover_shadow_mask"]
    assert s.pct_valid is not None
    fdir = frame_dir(data, "testcity", 38238)
    assert sorted(p.name for p in fdir.glob("*.nc")) == sorted(g.filename for g in gs)
    manifest = json.loads(manifest_path(data, "testcity").read_text())
    assert manifest["frames"][0]["n_epochs"] == 3
    text = report.read_text()
    assert "all configured frames complete" in text and "[MEASURED]" in text and "[DERIVED]" in text

    mtimes = {p.name: p.stat().st_mtime_ns for p in fdir.rglob("*") if p.suffix in (".nc", ".tif")}
    _, run2 = ingest.run_ingest(region, data_dir=data, workers=1, report_path=None, session=object())
    assert run2["fetched"] == 0 and run2["skipped"] == 3
    assert mtimes == {p.name: p.stat().st_mtime_ns for p in fdir.rglob("*") if p.suffix in (".nc", ".tif")}


def test_run_ingest_cleans_leftovers_and_refetches_broken_files(tmp_path, monkeypatch, grid):
    src = tmp_path / "remote"
    src.mkdir()
    gs, static_urls = _make_sources(src, grid)
    _patch_search(monkeypatch, {38238: (gs, static_urls)})
    region, data = _region(), tmp_path / "data"
    ingest.run_ingest(region, data_dir=data, workers=1, session=object())

    fdir, sdir = frame_dir(data, "testcity", 38238), static_dir(data, "testcity", 38238)
    (fdir / gs[0].filename).write_bytes(b"\x89HDF truncated")  # e.g. crash before writeback
    leftovers = [fdir / "a.nc.partial", fdir / "b.nc.download", sdir / "c.tif.partial", sdir / "d.tif.download"]
    for p in leftovers:
        p.write_bytes(b"x")
    dem = next(sdir.glob("*_dem.tif"))
    dem.write_bytes(b"broken tif")

    summaries, run = ingest.run_ingest(region, data_dir=data, workers=1, session=object())
    assert run["fetched"] == 1 and run["skipped"] == 2
    assert not any(p.exists() for p in leftovers)
    assert ingest.check_subset(fdir / gs[0].filename, HOUSTON_BBOX) is None
    assert ingest.check_static(dem, HOUSTON_BBOX) is None
    assert summaries[0].static_layers == ["line_of_sight_enu", "dem", "layover_shadow_mask"]


def test_run_ingest_stops_when_bbox_changed(tmp_path, monkeypatch, grid):
    src = tmp_path / "remote"
    src.mkdir()
    gs, static_urls = _make_sources(src, grid)
    _patch_search(monkeypatch, {38238: (gs, static_urls)})
    data = tmp_path / "data"
    small = (-95.82, 29.76, -95.67, 29.90)
    ingest.run_ingest(_region(bbox=small), data_dir=data, workers=1, session=object())
    with pytest.raises(IngestError, match="different bbox"):
        ingest.run_ingest(_region(), data_dir=data, workers=1, session=object())
    # STATIC is checked too (with the subsets moved away)
    for p in frame_dir(data, "testcity", 38238).glob("*.nc"):
        p.unlink()
    with pytest.raises(IngestError, match="stale"):
        ingest.run_ingest(_region(), data_dir=data, workers=1, session=object())


def test_run_ingest_stops_on_missing_layer(tmp_path, monkeypatch, grid):
    x, y = grid
    p = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(p, x, y, drop=("recommended_mask",))
    monkeypatch.setattr(ingest, "search_disp", lambda fid, ver, session: _granules_for([p]))
    monkeypatch.setattr(ingest, "search_static", lambda fid, session: StaticGranule("S", fid, "2014", "1.0", {}))
    with pytest.raises(IngestError, match="recommended_mask"):
        ingest.run_ingest(_region(), data_dir=tmp_path / "d", workers=1, session=object())


def test_frames_option_keeps_every_configured_frame_in_manifest(tmp_path, monkeypatch, grid):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _patch_search(monkeypatch, {38238: _make_sources(a, grid), 8882: _make_sources(b, grid, frame_id=8882)})
    region, data = _region(frames=(38238, 8882)), tmp_path / "data"
    report = tmp_path / "r.md"
    ingest.run_ingest(region, data_dir=data, workers=1, session=object())
    summaries, run = ingest.run_ingest(region, data_dir=data, frames=[8882], workers=1, report_path=report, session=object())
    assert run["frames_processed"] == [8882]
    assert [s.frame_id for s in summaries] == [38238, 8882]
    assert all(s.n_epochs == 3 for s in summaries)
    assert [f["frame_id"] for f in json.loads(manifest_path(data, "testcity").read_text())["frames"]] == [38238, 8882]
    assert "F38238" in report.read_text() and "F08882" in report.read_text()


def test_cli_prints_acceptance_summary_and_rerun_skips(tmp_path, monkeypatch, grid, capsys):
    src = tmp_path / "remote"
    src.mkdir()
    gs, static_urls = _make_sources(src, grid)
    _patch_search(monkeypatch, {38238: (gs, static_urls)})
    cfg = tmp_path / "regions.yaml"
    cfg.write_text(
        "testcity:\n  description: test\n  bbox: [-95.82, 29.76, -95.51, 30.03]\n"
        "  frames:\n    - {frame_id: 38238, direction: descending}\n"
    )
    args = ["--region", "testcity", "--config", str(cfg), "--data-dir", str(tmp_path / "data"),
            "--reports-dir", str(tmp_path / "reports"), "--workers", "1"]
    assert ingest.main(args) == 0
    out = capsys.readouterr().out
    manifest = json.loads(manifest_path(tmp_path / "data", "testcity").read_text())["frames"][0]
    assert "epochs on disk        [MEASURED]: 3 (files 3 of 3 listed by CMR, 0 missing)" in out
    assert "date range            [MEASURED]: 2016-10-16 to 2016-11-09" in out
    assert f"% valid pixels        [DERIVED] : {manifest['pct_valid']}% overall" in out
    report = (tmp_path / "reports" / "m0_ingest_testcity.md").read_text()
    assert "--region testcity" in report and "epochs on disk" in report

    assert ingest.main(args) == 0  # rerun
    assert "epochs on disk        [MEASURED]: 3" in capsys.readouterr().out


def test_cli_reports_failures_with_exit_code(tmp_path, monkeypatch, grid, capsys, caplog):
    x, y = grid
    p = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(p, x, y, drop=("displacement",))
    monkeypatch.setattr(ingest, "search_disp", lambda fid, ver, session: _granules_for([p]))
    monkeypatch.setattr(ingest, "search_static", lambda fid, session: StaticGranule("S", fid, "2014", "1.0", {}))
    cfg = tmp_path / "regions.yaml"
    cfg.write_text("t:\n  bbox: [-95.82, 29.76, -95.51, 30.03]\n  frames:\n    - {frame_id: 38238, direction: descending}\n")
    with caplog.at_level(logging.ERROR, logger="vigil.ingest"):
        code = ingest.main(["--region", "t", "--config", str(cfg), "--data-dir", str(tmp_path / "d"), "--no-report", "--workers", "1"])
    assert code == 2 and "STOPPED" in caplog.text and "displacement" in caplog.text


# ---------------------------------------------------------------- over HTTP --------


@pytest.mark.parametrize("method", ["range", "full"])
def test_http_fetch_matches_local_subset_and_sends_no_credentials(tmp_path, monkeypatch, grid, range_server, method, _isolated_home):
    base, srv = range_server
    nrc = _isolated_home / ".netrc"
    nrc.write_text("default login alice password secret\n")  # must never be sent anywhere
    nrc.chmod(0o600)
    gs, static_urls = _make_sources(srv.root, grid, base_url=base)
    _patch_search(monkeypatch, {38238: (gs, static_urls)})
    data = tmp_path / "data"
    summaries, run = ingest.run_ingest(_region(), data_dir=data, method=method, workers=1, session=object())
    assert run["fetched"] == 3 and summaries[0].n_epochs == 3
    assert summaries[0].static_layers == ["line_of_sight_enu", "dem", "layover_shadow_mask"]
    assert all(r["auth"] is None for r in srv.requests)
    if method == "range":
        assert any(r["range"] and r["range"] != "bytes=0-0" for r in srv.requests)

    local = tmp_path / "local.nc"
    ingest.write_subset(srv.root / gs[0].filename, local, HOUSTON_BBOX)
    with xr.open_dataset(local, engine="h5netcdf") as a, xr.open_dataset(
        frame_dir(data, "testcity", 38238) / gs[0].filename, engine="h5netcdf"
    ) as b:
        xr.testing.assert_identical(a, b)

    srv.requests.clear()
    _, run2 = ingest.run_ingest(_region(), data_dir=data, method=method, workers=1, session=object())
    assert run2["fetched"] == 0 and srv.requests == []  # rerun downloads nothing


def test_http_multiprocess_pool_and_failure_exit(tmp_path, monkeypatch, grid, range_server):
    """The CLI default path: spawn worker pool, range reads, and 'stop and report' on failures."""
    base, srv = range_server
    gs, static_urls = _make_sources(srv.root, grid, base_url=base)
    missing = DispGranule(**{**gs[0].__dict__, "url": f"{base}/" + disp_name(38238, "20160712T122240Z", "20161121T122243Z"),
                             "secondary_time": datetime(2016, 11, 21, tzinfo=UTC)})
    _patch_search(monkeypatch, {38238: (gs + [missing], static_urls)})
    data = tmp_path / "data"
    with pytest.raises(IngestError, match="1 products failed after retries"):
        ingest.run_ingest(_region(), data_dir=data, workers=2, session=object())
    fdir = frame_dir(data, "testcity", 38238)
    assert sorted(p.name for p in fdir.glob("*.nc")) == sorted(g.filename for g in gs)
    assert not list(fdir.glob("*.partial"))
    manifest = json.loads(manifest_path(data, "testcity").read_text())  # still written, with the gap
    assert manifest["frames"][0]["n_missing"] == 1


def test_http_pool_stops_on_fatal_error(tmp_path, monkeypatch, grid, range_server):
    base, srv = range_server
    x, y = grid
    bad = srv.root / disp_name(38238, "20160712T122240Z", "20161004T122243Z")
    write_disp_product(bad, x, y, drop=("recommended_mask",))
    gs, static_urls = _make_sources(srv.root, grid, base_url=base)
    _patch_search(monkeypatch, {38238: (_granules_for([bad], base) + gs, static_urls)})
    with pytest.raises(IngestError, match="recommended_mask"):
        ingest.run_ingest(_region(), data_dir=tmp_path / "data", workers=2, session=object())
    assert not list(frame_dir(tmp_path / "data", "testcity", 38238).glob("*.partial"))
