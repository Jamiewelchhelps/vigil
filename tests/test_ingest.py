"""Tests for pipeline.ingest on synthetic data. No network."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from pipeline import ingest
from pipeline.ingest import (
    CMR_GRANULES_URL,
    DispGranule,
    EarthdataAuth,
    EarthdataAuthError,
    IngestError,
    MissingLayerError,
    Region,
    FrameSpec,
    StaticGranule,
)
from pipeline.schemas import PARTIAL_SUFFIX, frame_dir, manifest_path

from .conftest import EPSG, HOUSTON_BBOX, disp_name, frame_grid, write_disp_product, write_static_tif

UTC = timezone.utc


# ---------------------------------------------------------------- config ----------


def test_houston_region_config_is_sane():
    r = ingest.load_region("houston")
    w, s, e, n = r.bbox
    assert w < e and s < n
    # ~30 x 30 km box (spine section 9)
    width_km = (e - w) * 111.32 * np.cos(np.radians((s + n) / 2))
    height_km = (n - s) * 110.57
    assert 25 < width_km < 35 and 25 < height_km < 35
    directions = {f.direction for f in r.frames}
    assert directions == {"ascending", "descending"}
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
    assert out[1].production_time.year == 2025 and out[1].production_time.month == 11


class _Resp:
    def __init__(self, status=200, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


class _CmrSession:
    def __init__(self, pages, hits):
        self.pages = pages
        self.hits = hits
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert url == CMR_GRANULES_URL
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


def test_auth_from_env_and_netrc(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for v in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    assert EarthdataAuth.from_env() == EarthdataAuth()
    nrc = tmp_path / ".netrc"
    nrc.write_text("machine urs.earthdata.nasa.gov login alice password secret\n")
    nrc.chmod(0o600)
    a = EarthdataAuth.from_env()
    assert (a.username, a.password, a.token) == ("alice", "secret", None)
    monkeypatch.setenv("EARTHDATA_TOKEN", "TOK")
    assert EarthdataAuth.from_env().token == "TOK"


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


def test_fetch_subset_reports_missing_layer_as_fatal(tmp_path, grid):
    x, y = grid
    src = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(src, x, y, drop=("displacement",))
    res = ingest.fetch_subset(str(src), str(tmp_path / "o.nc"), HOUSTON_BBOX)
    assert not res.ok and res.fatal and "displacement" in res.error


def test_crop_static_keeps_grid_values_and_band_names(tmp_path, grid):
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
        col0 = int(round((b.left - (x[0] - (x[1] - x[0]) / 2)) / (x[1] - x[0])))
        row0 = int(round(((y[0] + (x[1] - x[0]) / 2) - b.top) / (x[1] - x[0])))
        np.testing.assert_array_equal(ds.read(), data[:, row0 : row0 + h, col0 : col0 + w])


# ---------------------------------------------------------------- plan, summary ----


def _granules_for(paths):
    out = []
    for p in paths:
        info = ingest.parse_disp_name(p.name)
        out.append(DispGranule(granule_ur=p.stem, url=str(p), **info))
    return out


def test_plan_skips_complete_files_but_not_partials(tmp_path):
    names = [disp_name(38238, "20160712T122240Z", f"201610{d:02d}T122243Z") for d in (4, 16, 28)]
    gs = _granules_for([tmp_path / "src" / n for n in names])
    out = tmp_path / "F38238"
    out.mkdir()
    (out / names[0]).write_bytes(b"done")
    (out / (names[1] + PARTIAL_SUFFIX)).write_bytes(b"half")
    to_fetch, existing = ingest.plan_downloads(gs, out)
    assert [g.filename for g in existing] == [names[0]]
    assert [g.filename for g in to_fetch] == names[1:]
    assert ingest.clean_partials(out) == 1
    assert not (out / (names[1] + PARTIAL_SUFFIX)).exists()


def test_summarize_frame_counts_epochs_dates_and_valid_pixels(tmp_path, grid):
    x, y = grid
    out = tmp_path / "F38238"
    out.mkdir()
    secs = ["20161016T122243Z", "20161028T122243Z", "20170315T122243Z"]
    for i, sec in enumerate(secs):
        src = tmp_path / disp_name(38238, "20160712T122240Z", sec)
        write_disp_product(src, x, y, seed=i)
        ingest.write_subset(src, out / src.name, HOUSTON_BBOX)
    s = ingest.summarize_frame(out, HOUSTON_BBOX, 38238, "descending", n_expected=4)
    assert (s.n_epochs, s.n_on_disk, s.n_missing) == (3, 3, 1)
    assert (s.first_date, s.last_date) == ("2016-10-16", "2017-03-15")
    assert s.largest_gap_start == "2016-10-28" and s.largest_gap_days == 138.0
    assert s.crs == f"EPSG:{EPSG}" and s.pixel_size_m == [600.0, 600.0]

    # Recompute % valid independently, inside the bbox polygon only
    import shapely

    poly = ingest.bbox_polygon_projected(HOUSTON_BBOX, CRS.from_epsg(EPSG))
    total = valid = 0
    for p in sorted(out.glob("*.nc")):
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            xx, yy = np.meshgrid(ds.x.values, ds.y.values)
            inside = shapely.contains_xy(poly, xx, yy)
            ok = np.isfinite(ds.displacement.values) & (ds.recommended_mask.values == 1) & inside
            total += inside.sum()
            valid += ok.sum()
    assert s.pct_valid == round(100 * valid / total, 2)
    assert 60 < s.pct_valid < 85  # ~80% mask-good x ~95% finite
    assert s.n_pixels_in_bbox == total // 3


# ---------------------------------------------------------------- end to end -------


def test_run_ingest_end_to_end_then_rerun_skips(tmp_path, monkeypatch, grid):
    """M0 acceptance logic offline: subsets + STATIC + summary, and a rerun fetches nothing."""
    x, y = grid
    src = tmp_path / "remote"
    src.mkdir()
    paths = []
    for i, sec in enumerate(["20161016T122243Z", "20161028T122243Z", "20161109T122243Z"]):
        p = src / disp_name(38238, "20160712T122240Z", sec)
        write_disp_product(p, x, y, seed=i)
        paths.append(p)
    static_urls = {}
    for layer, bands in (("line_of_sight_enu", 3), ("dem", 1), ("layover_shadow_mask", 1)):
        p = src / f"OPERA_L3_DISP-S1-STATIC_F38238_20140403_S1A_v1.0_{layer}.tif"
        write_static_tif(p, x, y, bands=bands)
        static_urls[layer] = str(p)

    monkeypatch.setattr(ingest, "search_disp", lambda fid, ver, session: _granules_for(paths))
    monkeypatch.setattr(
        ingest, "search_static",
        lambda fid, session: StaticGranule("S", fid, "20140403", "1.0", static_urls),
    )
    region = Region("testcity", HOUSTON_BBOX, (FrameSpec(38238, "descending"),))
    data = tmp_path / "data"
    report = tmp_path / "reports" / "m0.md"

    summaries, run = ingest.run_ingest(region, data_dir=data, workers=1, report_path=report, session=object())
    assert run["fetched"] == 3 and run["skipped"] == 0 and run["failed"] == 0
    (s,) = summaries
    assert s.n_epochs == 3 and s.n_missing == 0
    assert s.static_layers == ["line_of_sight_enu", "dem", "layover_shadow_mask"]
    assert s.pct_valid is not None
    fdir = frame_dir(data, "testcity", 38238)
    assert sorted(p.name for p in fdir.glob("*.nc")) == sorted(p.name for p in paths)
    manifest = json.loads(manifest_path(data, "testcity").read_text())
    assert manifest["frames"][0]["n_epochs"] == 3
    assert "epochs on disk" in report.read_text()

    mtimes = {p.name: p.stat().st_mtime_ns for p in fdir.rglob("*") if p.is_file() and p.suffix in (".nc", ".tif")}
    _, run2 = ingest.run_ingest(region, data_dir=data, workers=1, report_path=None, session=object())
    assert run2["fetched"] == 0 and run2["skipped"] == 3
    assert mtimes == {p.name: p.stat().st_mtime_ns for p in fdir.rglob("*") if p.is_file() and p.suffix in (".nc", ".tif")}


def test_run_ingest_stops_on_missing_layer(tmp_path, monkeypatch, grid):
    x, y = grid
    p = tmp_path / disp_name(38238, "20160712T122240Z", "20161016T122243Z")
    write_disp_product(p, x, y, drop=("recommended_mask",))
    monkeypatch.setattr(ingest, "search_disp", lambda fid, ver, session: _granules_for([p]))
    monkeypatch.setattr(ingest, "search_static", lambda fid, session: StaticGranule("S", fid, "2014", "1.0", {}))
    region = Region("t", HOUSTON_BBOX, (FrameSpec(38238, "descending"),))
    with pytest.raises(IngestError, match="recommended_mask"):
        ingest.run_ingest(region, data_dir=tmp_path / "d", workers=1, session=object())
