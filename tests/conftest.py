"""Synthetic OPERA DISP-S1 / DISP-S1-STATIC files for tests. No network anywhere in tests/."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from pipeline.ingest import bbox_envelope

HOUSTON_BBOX = (-95.82, 29.76, -95.51, 30.03)
EPSG = 32615  # UTM 15N, the zone DISP-S1 uses over Houston
DX = 600.0  # coarse pixels keep the synthetic files tiny


def disp_name(frame_id: int, ref: str, sec: str, prod: str = "20251023T225135Z") -> str:
    return f"OPERA_L3_DISP-S1_IW_F{frame_id:05d}_VV_{ref}_{sec}_v1.0_{prod}.nc"


def frame_grid(bbox=HOUSTON_BBOX, margin_m: float = 6000.0, dx: float = DX):
    """North-up UTM pixel-centre coordinates covering the bbox with a margin (like a frame)."""
    xmin, ymin, xmax, ymax = bbox_envelope(bbox, CRS.from_epsg(EPSG))
    x0 = np.floor((xmin - margin_m) / dx) * dx
    y0 = np.ceil((ymax + margin_m) / dx) * dx
    nx = int(np.ceil((xmax + margin_m - x0) / dx))
    ny = int(np.ceil((y0 - (ymin - margin_m)) / dx))
    x = x0 + dx / 2 + dx * np.arange(nx)
    y = y0 - dx / 2 - dx * np.arange(ny)
    return x, y


def write_disp_product(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    drop: tuple[str, ...] = (),
    drop_corrections: tuple[str, ...] = (),
    bad_fraction: float = 0.2,
    nan_fraction: float = 0.05,
) -> dict[str, np.ndarray]:
    """Write a small product with the DISP-S1 v1.0 group/variable layout. Returns the arrays."""
    rng = np.random.default_rng(seed)
    shape = (y.size, x.size)
    disp = rng.normal(0, 0.01, shape).astype("float32")
    disp[rng.random(shape) < nan_fraction] = np.nan
    mask = (rng.random(shape) >= bad_fraction).astype("uint8")
    arrays = {
        "displacement": disp,
        "short_wavelength_displacement": (disp * 0.5).astype("float32"),
        "temporal_coherence": rng.random(shape).astype("float32"),
        "recommended_mask": mask,
        "persistent_scatterer_mask": (rng.random(shape) > 0.9).astype("uint8"),
        "estimated_phase_quality": rng.random(shape).astype("float32"),
        "water_mask": np.ones(shape, dtype="uint8"),
    }
    crs = CRS.from_epsg(EPSG)
    spatial_ref = xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt()})
    root = xr.Dataset(
        {
            name: (("y", "x"), arr, {"grid_mapping": "spatial_ref", "units": "meters"})
            for name, arr in arrays.items()
            if name not in drop
        },
        coords={"x": ("x", x, {"units": "meters"}), "y": ("y", y, {"units": "meters"})},
    )
    root["spatial_ref"] = spatial_ref
    root.attrs["product_version"] = "1.0"
    enc = {n: {"_FillValue": np.nan} for n in root.data_vars if root[n].dtype.kind == "f"}
    enc.update({n: {"_FillValue": 255} for n in root.data_vars if root[n].dtype == np.uint8})
    root.to_netcdf(path, engine="h5netcdf", encoding=enc)

    corr = xr.Dataset(
        {
            name: (("y", "x"), rng.normal(0, 0.001, shape).astype("float32"))
            for name in ("ionospheric_delay", "solid_earth_tide")
            if name not in drop_corrections
        },
        coords={"x": ("x", x), "y": ("y", y)},
    )
    corr["perpendicular_baseline"] = xr.DataArray(np.float32(42.0))
    corr["reference_point"] = xr.DataArray(0, attrs={"rows": 10, "cols": 12})
    corr.to_netcdf(path, mode="a", engine="h5netcdf", group="corrections")

    with h5py.File(path, "a") as f:
        md = f.create_group("metadata")
        md.create_dataset("pge_runconfig", data=np.bytes_("runconfig: synthetic"))
        md.create_dataset("platform_id", data=np.bytes_("S1A"))
        ident = f.create_group("identification")
        ident.create_dataset("frame_id", data=np.int64(38238))
        ident.create_dataset("product_version", data=np.bytes_("1.0"))
    return arrays


def write_static_tif(path: Path, x: np.ndarray, y: np.ndarray, bands: int = 3, seed: int = 0) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_origin

    dx = float(x[1] - x[0])
    transform = from_origin(x[0] - dx / 2, y[0] + dx / 2, dx, dx)
    rng = np.random.default_rng(seed)
    data = rng.random((bands, y.size, x.size)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", width=x.size, height=y.size, count=bands, dtype="float32",
        crs=f"EPSG:{EPSG}", transform=transform,
    ) as ds:
        ds.write(data)
        ds.update_tags(PRODUCT="synthetic")
        for b, name in zip(range(1, bands + 1), ["east", "north", "up"]):
            ds.set_band_description(b, name)
    return data


@pytest.fixture
def grid():
    return frame_grid()
