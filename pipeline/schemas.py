"""Contracts between pipeline stages (BUILD_SPINE.md section 3.5).

Only the contracts needed by the current milestone live here. Later milestones add theirs
when they are built (M1 adds stack -> velocity, and so on).

Ingest -> stack (M0)
--------------------
On disk, under ``<data_dir>/raw/<region>/``::

    F<frame:05d>/<OPERA DISP-S1 granule name>.nc          bbox subset of one product, OPERA name kept
    F<frame:05d>/static/<OPERA STATIC granule>_<layer>.tif bbox crop of each STATIC layer
    manifest.json                                         IngestManifest (below)

The stack stage reads the subsets named in ``manifest.json`` (``frames[].files``), not a glob of
the folder: only those are current CMR products, readable, and cut for the configured bbox.
Every data variable is written with HDF5 fletcher32 checksums, so a corrupted chunk raises on
read instead of returning wrong values.

Each ``.nc`` subset keeps the product's groups: root layers (``displacement``,
``short_wavelength_displacement``, ``temporal_coherence``, ``recommended_mask``, ...),
``/corrections``, ``/metadata`` and ``/identification``, cropped to the pixels that touch the
region bbox (plus a 1-pixel margin) on the product's native UTM grid. Nothing is resampled.
A file only appears under its final name once it is complete (written to ``*.partial`` first).

Georeferencing in each subset describes the subset, not the full frame:
- ``spatial_ref.GeoTransform`` (root and ``/corrections``) is shifted to the subset origin.
- ``/corrections/reference_point`` attrs ``rows``/``cols`` index the subset grid (they can be
  outside it: see ``vigil_reference_point_in_subset``); the frame indices are kept as
  ``frame_rows``/``frame_cols``.
- Root attrs ``vigil_subset_bbox_lonlat`` (the region bbox it was cut for),
  ``vigil_subset_rows`` and ``vigil_subset_cols`` (the window in the full frame, [start, stop)).
Each cropped STATIC GeoTIFF carries the tag ``VIGIL_SUBSET_BBOX_LONLAT``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PARTIAL_SUFFIX = ".partial"


def region_raw_dir(data_dir: Path, region: str) -> Path:
    return Path(data_dir) / "raw" / region


def frame_dir(data_dir: Path, region: str, frame_id: int) -> Path:
    return region_raw_dir(data_dir, region) / f"F{frame_id:05d}"


def static_dir(data_dir: Path, region: str, frame_id: int) -> Path:
    return frame_dir(data_dir, region, frame_id) / "static"


def manifest_path(data_dir: Path, region: str) -> Path:
    return region_raw_dir(data_dir, region) / "manifest.json"


@dataclass
class FrameSummary:
    """What is on disk for one frame after ingest. Feeds the M0 summary and report.

    Provenance: counts and dates are MEASURED (read from the products); percentages and gap
    lengths are DERIVED from them.
    """

    frame_id: int
    direction: str
    n_expected: int  # granules CMR lists for this frame
    n_on_disk: int  # complete subsets on disk
    n_missing: int  # listed by CMR but not usable on disk (DERIVED)
    n_superseded: int  # older productions on disk that CMR no longer lists; ignored
    n_removed_broken: int  # unreadable files removed by this run; re-fetched next run
    n_epochs: int  # distinct secondary (acquisition) dates on disk
    first_date: str | None  # ISO date of the first secondary acquisition
    last_date: str | None
    n_reference_dates: int  # distinct reference dates (DISP-S1 re-references in ministacks)
    largest_gap_days: float | None
    largest_gap_start: str | None
    largest_gap_end: str | None
    crs: str | None  # e.g. "EPSG:32615"
    shape: list[int] | None  # [rows, cols] of the subset grid
    pixel_size_m: list[float] | None  # [dx, dy] in metres
    n_pixels_in_bbox: int | None  # pixel centres inside the bbox polygon
    bbox_coverage_pct: float | None  # share of the bbox polygon inside the subset grid
    pct_valid: float | None  # valid pixel-epochs / (pixels in bbox x epochs), percent
    pct_valid_epoch_min: float | None
    pct_valid_epoch_median: float | None
    pct_valid_epoch_max: float | None
    valid_rule: str = ""  # how "valid" was decided, in words
    static_layers: list[str] = field(default_factory=list)
    bytes_on_disk: int = 0
    files: list[str] = field(default_factory=list)  # the usable subsets, by OPERA name


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestManifest:
    region: str
    bbox: list[float]  # [west, south, east, north], EPSG:4326
    product_version: str
    created_utc: str
    frames: list[FrameSummary]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frames"] = [f.to_dict() for f in self.frames]
        return d
