# VIGIL (working name) : BUILD SPINE
Every building, bridge and dam in America, watched from space.
Source of truth for this build. Claude Code reads this before doing anything and rewrites the Status, Milestone, Decision Log and Session Log sections at the end of every working session.
Last updated: 2026-11-06 | Session #: 1 (M0 code written and tested offline; real download blocked on ASF authorization)

Tags used in this doc: [V] = verified against a published source when this spec was written. [E] = estimate, to be replaced with a measured number during the build.

---

## 0. STATUS (read this first)
- Current milestone: M0, Environment and data access (in progress: code done, 30 offline tests pass, frame search and dry run work against live CMR)
- Blocked by: ASF download authorization. An Earthdata token is saved in the `vigil` cloud environment, but every product request still redirects to Earthdata Login. Most likely the account has not yet authorized ASF's download application.
- Next 3 actions (in order):
  1. D: log in at urs.earthdata.nasa.gov, open one product URL in the same browser, approve the ASF authorization/EULA prompt; confirm the token came from "Generate Token" and is not expired (5 min)
  2. Claude Code: fetch one real product, record bytes read, seconds and layer attributes (`recommended_mask` meaning), then run `python -m pipeline.ingest --region houston` (both frames) and rerun to prove skipping
  3. Claude Code: record the M0 pass/fail, commit `reports/m0_ingest_houston.md`, update this block

---

## RULES FOR CLAUDE CODE
- Read this whole doc before writing code. Open every session with the Status block in 2 lines.
- Build ONE milestone at a time. Stop when its acceptance test passes or fails, report the result, and wait for D before starting the next milestone. Never build ahead.
- The plan changes only through a Decision Log entry with a reason. Do not re-plan from scratch.
- A list of subsystems is not a plan. Every task ties to the current milestone. Anything else goes in the Parking Lot.
- Locked decisions are not reopened without new evidence. Name the evidence when proposing a change.
- Never invent data. If a download fails or a layer is missing, stop and report. No placeholder numbers in outputs, charts or UI.
- Every number shown to a user carries a provenance tag: MEASURED (satellite or GPS), DERIVED (computed from measured, for example a fitted velocity), or ESTIMATED (model output). This also feeds the engineering report D writes for each project.
- Every value shown for an asset carries a confidence level and an error bar. Never show a bare number.
- Wording rule for structures: never say a structure is "unsafe", "failing" or "dangerous". The product says "moving outside its normal pattern" and shows the evidence.
- No secrets in the repo. Credentials come from `.netrc` or env vars only.
- Tests: every pipeline module ships with pytest tests that run on small synthetic data stacks (no downloads needed in CI).
- End of session, or when D says "save": update Status, Milestone table, Decision Log and Session Log in this file, then commit.

---

## 1. MISSION
- **What it is (one sentence):** a 3D map of the United States where every building, bridge and dam is colored by how much it has moved since 2016, measured by radar satellites, with a time slider that replays the motion, a per-address movement report, and alerts when a structure starts moving outside its normal pattern.
- **Who it is for / who pays:**
  - Public, free: the 3D map (awe, reach, press)
  - Paid, consumer: per-address ground movement report for home buyers, inspectors, agents. UK precedent: Terrafirma's homebuyer Ground Report uses satellite ground-movement data at GBP 50 + VAT [V]
  - Paid, B2B: watchlists and alerts for insurers, HOAs (homeowners associations), cities
  - Paid, B2G (business to government): bridge watch for state DOTs (departments of transportation) and county bridge owners
- **Definition of done (v1):**
  - One metro (Houston) fully live: every building footprint, NBI (National Bridge Inventory) bridge and NID (National Inventory of Dams) dam in the metro has either a movement time series with confidence, or an explicit "no reliable data" reason
  - Velocities validated against independent GPS stations with a published error figure
  - Bridge anomaly model back-tested on a real pre-collapse case
  - Public 3D web map with time slider, asset click-through and an address report
  - Pipeline updates automatically as new satellite products land
- **Explicitly out of scope for v1:**
  - Our own InSAR (interferometric synthetic aperture radar) processing from raw radar images (v2, see Parking Lot)
  - Nationwide coverage (v1 is one metro, v1.5 is several)
  - Mobile app (the web map must work on a phone browser, but no native app)
  - User accounts beyond email alerts

## 2. WHY WE BUILD
- **Money:** per-address reports (UK version at GBP 50 + VAT [V], US price to test at $49 to $99 [E]), insurer and HOA subscriptions, DOT bridge-watch contracts. First dollar: a paid address report or a paid pilot after M6.
- **Influence and power:** the first public tool showing which structures in a city are moving abnormally. Gives credibility with DOTs, insurers, journalists, and anyone working on infrastructure.
- **Portfolio:** a city replaying ten years of ground motion in 3D is a demo people remember.
- **Makes D better:** satellite radar (SAR, synthetic aperture radar) and InSAR time-series analysis, a core ISR (intelligence, surveillance, reconnaissance) skill, on top of D's terrain and 3D background.
- **Helps his life:** none directly (D can check his own future rental properties before buying).

---

## 3. ARCHITECTURE

### 3.1 Data sources (all free)
| Source | What it gives | Access | Notes |
|---|---|---|---|
| NASA OPERA DISP-S1 [V] | Surface displacement time series, North America, ~30 m pixels, every 6 to 12 days, data from 2016 [V] | ASF DAAC (Alaska Satellite Facility data archive), Earthdata login, `opera-utils` Python package [V] | Displacement is in radar line of sight (LOS), relative to a reference date and a reference pixel [V]. Frames are ~200 x 250 km, built from ~27 radar bursts [V] |
| DISP-S1 layers used [V] | `displacement`, `short_wavelength_displacement`, `temporal_coherence`, `recommended_mask`, `persistent_scatterer_mask`, `estimated_phase_quality`, plus correction layers (`ionospheric_delay`, `solid_earth_tide`) | inside each product file | `short_wavelength_displacement` removes signals larger than 30 km [V]. Good for local motion, wrong for comparing with GPS over long distances |
| OPERA DISP-S1-STATIC [V] | LOS unit vectors, DEM (digital elevation model) used, layover/shadow mask, per frame | ASF DAAC, released July 15, 2026 [V] | Needed to convert LOS motion into vertical and east motion |
| Nevada Geodetic Laboratory GPS (NGL) [V] | Daily GPS position time series (tenv3 format) for thousands of stations, free [V] | geodesy.unr.edu station pages | Ground truth for validation |
| Harris-Galveston Subsidence District GPS map [V] | Local GPS network in Houston | hgsubsidence.org | Extra ground truth for the launch city |
| Overture Maps buildings [V] | Building footprints, some with heights from lidar [V], GeoParquet on S3, queryable with DuckDB by bounding box [V] | `s3://overturemaps-us-west-2/release/<latest>/theme=buildings/type=building/` [V] | Licensed ODbL [V]: keep attribution, and confirm what share-alike means for selling derived data before M8 |
| FHWA National Bridge Inventory [V] | 600,000+ bridges with coordinates, condition ratings 0 to 9, scour-critical flag, year built [V] | fhwa.dot.gov/bridge/nbi/ascii.cfm, annual files through 2025 data [V] | Some coordinates are wrong [V]: snap to road geometry before trusting |
| USACE National Inventory of Dams [V] | 91,000+ dams with coordinates and hazard class [V] | nid.sec.usace.army.mil, CSV/GeoPackage download and query API [V] | |
| Open-Meteo archive (ERA5) [V] | Hourly air temperature back to 1940, keyless [V] | archive-api.open-meteo.com | Free tier is non-commercial only [V]. Before revenue: buy a plan or pull ERA5 directly from Copernicus |

### 3.2 Satellite timeline facts the model must handle [V]
- Sentinel-1B stopped delivering data on Dec 23, 2021. Sentinel-1C launched Dec 5, 2024 and has acquired data regularly since Mar 26, 2025. Some areas had little or no coverage between Dec 2021 and Apr 2025.
- Sentinel-1D launched Nov 4, 2025, with calibrated data from Apr 17, 2026. ASF documentation lists Sentinel-1A as stopping collection on Jun 29, 2026.
- Consequence: sampling is irregular. Every fit must handle uneven time steps and gaps, and the UI must show gaps instead of interpolating across them silently.

### 3.3 Subsystems
1. **Ingest** (`pipeline/ingest.py`): find DISP-S1 frames covering a bounding box, download bounding-box subsets only (`opera-utils disp-s1-download --bbox ... --frame-id ...` [V]), plus the STATIC layers for the same frames.
2. **Stack and quality** (`pipeline/stack.py`): build an xarray (labeled N-dimensional array) time series per frame, apply `recommended_mask`, record per-pixel quality (temporal coherence, persistent-scatterer flag, number of valid epochs).
3. **Velocity** (`pipeline/velocity.py`): per pixel, fit displacement(t) = offset + rate x t + annual sine/cosine + optional step terms. Output rate, seasonal amplitude, uncertainties, residual RMS (root mean square).
4. **Decomposition** (`pipeline/decompose.py`): where ascending and descending frames overlap, solve for vertical and east motion using the STATIC LOS unit vectors (north motion is assumed negligible, the standard choice for polar-orbit radar; document it). Where only one geometry exists, keep LOS only and flag the pixel `geometry=single`.
5. **Validation** (`pipeline/validate.py`): load GPS stations in the bounding box, project GPS east/north/up velocity into the radar LOS, reference both datasets to the same stable point, compare. Output a validation report (table + scatter plot) with the error figure.
6. **Assets** (`pipeline/assets.py`): pull buildings (Overture), bridges (NBI), dams (NID) for the bounding box. Sample pixels under or around each asset (footprint polygon for buildings, buffered point or deck line for bridges), aggregate with quality weights, output a per-asset time series, velocity, acceleration and confidence, or a "no reliable data" reason.
7. **Anomaly** (`pipeline/anomaly.py`): per bridge or dam, learn its normal pattern (trend + seasonal + temperature response using ERA5 temperature), then score new observations by deviation from that pattern and by trend acceleration. Requires at least 12 months of history [V: a published study found 12 months needed to capture the seasonal cycle].
8. **Tiles and export** (`pipeline/export.py`): building footprints with properties into PMTiles (single-file vector tiles) via tippecanoe [V]. Per-asset time series as compact static JSON or binary chunks keyed by tile. No database in v1.
9. **Web** (`web/`): Next.js + MapLibre GL + deck.gl. 3D extruded buildings colored by vertical velocity, time slider replaying cumulative displacement, bridges and dams as symbols, click any asset for a chart with error bands, address search, report page.
10. **Scheduler** (`pipeline/update.py`): incremental update when new DISP-S1 products appear. Append epochs, refit, re-score anomalies, rebuild only changed tiles.
11. **Alerts** (v1 late): email subscription per address or asset; send when an asset's anomaly score crosses a threshold.

### 3.4 Repo layout
```
vigil/
  CLAUDE.md                 # copy of RULES FOR CLAUDE CODE + pointer to this spine
  BUILD_SPINE.md            # this file
  config/
    regions.yaml            # bbox, frame IDs, GPS station list per region
  pipeline/
    ingest.py stack.py velocity.py decompose.py validate.py
    assets.py anomaly.py export.py update.py
    schemas.py              # pydantic or dataclass definitions of every contract below
  tests/                    # pytest, synthetic stacks, no network
  data/                     # gitignored: raw/, interim/, products/
  reports/                  # validation reports, back-test reports (committed)
  web/                      # Next.js app
```

### 3.5 Interfaces (contracts that must hold)
- **Ingest to stack:** NetCDF/HDF5 DISP-S1 subsets on disk, one folder per frame, plus that frame's STATIC file. File names keep the OPERA naming convention.
- **Stack to velocity:** `xarray.Dataset` dims (time, y, x), variables `disp_mm`, `quality`, `mask`, CRS (coordinate reference system) attached, time as UTC datetimes.
- **Velocity output** `products/velocity_<region>.parquet`, one row per pixel:
  `x, y, lat, lon, frame_id, geometry(asc|desc|both|single), v_los_mm_yr, v_los_sigma, v_up_mm_yr?, v_up_sigma?, v_east_mm_yr?, seasonal_amp_mm, residual_rms_mm, temporal_coherence, n_epochs, first_date, last_date`
- **Asset output** `products/assets_<region>.parquet`, one row per asset:
  `asset_id, asset_type(building|bridge|dam), source(overture|nbi|nid), source_id, name?, lat, lon, height_m?, n_pixels, v_up_mm_yr, v_sigma, accel_mm_yr2, confidence(high|medium|low|none), no_data_reason?, series_ref`
- **Asset series** `products/series/<tile>.json`: `{asset_id: {dates:[...], disp_mm:[...], sigma_mm:[...], gaps:[[start,end],...]}}`
- **Anomaly output** `products/anomalies_<region>.parquet`:
  `asset_id, detected_on, score, metric(acceleration|seasonal_break|step|residual), window_start, window_end, model_version, evidence_ref`
- **Web tiles** `buildings.pmtiles`, feature properties: `id, h, v_up, conf, prov(MEASURED|DERIVED|ESTIMATED)`

### 3.6 Key constraints
- Resolution: DISP-S1 is ~30 m [V]. Many houses are smaller than one pixel. Per-asset values are neighborhood-scale for small buildings and must say so (confidence and a "measured at ~30 m resolution" note).
- Radar sees in line of sight, not straight down. Vertical needs two viewing geometries.
- Coherence drops over vegetation, water and fresh construction. Expect holes.
- Disk: bounding-box subsets keep M0 to M6 small. Full frames are large; the frame raster in one published example is 7,733 x 9,464 pixels [V], so full-frame stacks over ~10 years run to tens of GB per frame [E].
- Compute: everything up to M6 runs on a laptop (D's Dell Precision 5770). If Claude Code runs in a cloud sandbox with small disk, run the pipeline on the laptop or a cloud VM (virtual machine) and do the web work anywhere.

---

## 4. MILESTONE LADDER
Each milestone is a working end-to-end slice with a pass/fail test, ordered so the biggest risks die first. Time estimates are for D working with Claude Code, all [E].

| # | Milestone | Pass/fail acceptance test | Depends on | Risk it retires | Est. time | Status |
|---|---|---|---|---|---|---|
| M0 | Environment and data access | One command (`python -m pipeline.ingest --region houston`) downloads a Houston bbox subset for every available epoch from at least one frame plus its STATIC file, and prints: epoch count, date range, % valid pixels. Rerunning skips existing files | none | Can we get the data at all | 2 to 4 h | in progress: code + offline tests done; real download blocked on ASF authorization |
| M1 | One city, one truth | LOS velocity map for the Houston bbox, compared with at least 10 GPS stations projected into LOS. Validation report committed with scatter plot. PASS if median absolute velocity difference is at most 3 mm/yr [E: threshold to revisit after first run, record the measured figure either way] | M0 | Is the data accurate enough to publish | 1 to 2 days | |
| M2 | Vertical motion | Where ascending and descending overlap, vertical velocity computed and validated against GPS vertical with the same method as M1. Single-geometry pixels flagged. PASS if vertical median absolute difference is at most 3 mm/yr [E] and coverage map shows which areas are `both` vs `single` | M1 | Can we honestly say "sinking" | 0.5 to 1 day | |
| M3 | Every asset gets an answer | All Overture buildings, NBI bridges and NID dams in the bbox processed. PASS if 100% of assets have either a series with confidence or a `no_data_reason`, and a manual spot check of 20 random assets matches the raw pixels | M2 | Can we go from pixels to "your house" | 1 day | |
| M4 | The awe demo | Web map of Houston: 3D buildings colored by vertical velocity, time slider replays 2016 to now, click any asset for a chart with error bands and gap shading, provenance and confidence shown. PASS if it loads in under 3 s on broadband [E target], stays above 30 fps on D's laptop, and 3 people who have never seen it can say what it shows within 10 seconds | M3 | Does it actually produce awe | 2 to 3 days | |
| M5 | Bridge watch back-test | Anomaly model run on Dresden's Carola Bridge (collapsed Sept 2024; a published study found InSAR anomalies in the months before [V]) using EGMS (European Ground Motion Service) data up to the day before collapse, plus at least 20 other Dresden bridges as controls. PASS if Carola is flagged before collapse and the false-positive rate on controls is reported. Then run on Houston bridges and publish nothing until D reviews | M3 | Can we see failure coming without crying wolf | 2 to 3 days | |
| M6 | Address report | Type any Houston address and get a report page (exportable to PDF): movement history, velocity with error bar, confidence, data gaps, nearby bridges and dams, method note. PASS if 10 test addresses each produce a report in under 30 s with zero untagged numbers | M4 | Is there a product someone pays for | 1 to 2 days | |
| M7 | Living system | Incremental update: new DISP-S1 epochs picked up automatically, refit, re-score, rebuild changed tiles only. Add 2 more metros (pick from cities with documented subsidence, for example from the Nature Cities study of 28 sinking US cities [V]). PASS if an update run touches only new epochs and finishes within a set time budget [E: set after M6] | M6 | Can it run without D babysitting it | 2 to 3 days | |
| M8 | First dollar | Launch page, payment for address reports (Stripe), email alerts live. Unconventional move: before launch, send the Texas DOT and Harris County a free private bridge-watch summary for their bridges. PASS if one paid report is sold or one pilot meeting is booked | M7 | Will anyone pay | ongoing | |

---

## 5. RISK REGISTER
Ranked by how likely each one is to sink the project.

| Risk | Severity | Retired by | Fix / mitigation | Status |
|---|---|---|---|---|
| Accuracy challenged publicly. Science reported in June 2026 that satellite maps of sinking coastlines are under scrutiny [V] | major | M1, M2 | Validate against GPS, publish the error figure, show error bars everywhere, link the method | open |
| ~30 m pixels are coarser than many houses | major | M3 (honest labels), v2 (fix) | Confidence levels plus a resolution note now; v2 processes the 5 m x 10 m CSLC (coregistered single-look complex) inputs [V] with the open-source dolphin library [V] for building-level points | open |
| False alarms on bridges from normal thermal and seasonal motion | major | M5 | Learn each structure's normal pattern with temperature as an input, require 12+ months of history, back-test on a real collapse plus controls | open |
| Publicly naming a bridge as abnormal triggers backlash or legal threats | major | M5, M8 | Wording rule (Rules section), private disclosure to the owner first, show evidence instead of verdicts | open |
| Coverage gap Dec 2021 to Apr 2025 [V] weakens fits in some areas | minor | M1 | Irregular-sampling fits, gap shading in charts, per-asset `n_epochs` | open |
| Only one viewing geometry in some areas | minor | M2 | LOS-only fallback with flag; NISAR (NASA-ISRO SAR satellite) DISP product later adds regular ascending and descending coverage over the US [V] | open |
| Commercial-use limits: Open-Meteo free tier [V], Overture ODbL share-alike [V license] | minor | M8 | Paid Open-Meteo plan or ERA5 from Copernicus; confirm the ODbL position before selling derived building data | open |
| Data size and compute grow with each metro | minor | M7 | bbox subsets, incremental updates, Parquet + PMTiles, no database until needed | open |
| UK incumbents (Terrafirma, SatSense [V]) or InSAR firms enter the US consumer market | minor | M8 | Speed to launch, the public map as distribution, US-specific bridge and dam layer | open |
| DISP-S1 arrives in batches months after acquisition. Measured 2026-11-06 for Houston: newest epoch 2026-03-22 (F38238) and 2026-02-19 (F8882), both produced 2026-08-17; production batches 2025-10, 2025-11, 2026-03, 2026-08. Alerts (section 3.3 item 11, M8) would be months late | major for alerts, minor for the map and reports | M7 | Measure lag per batch in M7; word alerts honestly ("data through <date>"); v2 own processing from CSLC, or NISAR, could shorten it | open |
| Sampling halves after 2021: F38238 has ~60 epochs/yr 2017-2021, ~30/yr 2022-2025; no Sentinel-1C densification visible in 2025 (31 epochs) [measured from CMR] | minor | M1 | Irregular-sampling fits already planned; record epochs/yr in reports | open |

## 6. PROCUREMENT / BOM (bill of materials)
| Item | Purpose | Needed by | Lead time | Cost | Source | Status |
|---|---|---|---|---|---|---|
| NASA Earthdata login | DISP-S1 download | M0 | minutes | free | urs.earthdata.nasa.gov | token saved in `vigil` cloud environment; ASF app authorization pending |
| Disk space ~50 to 100 GB | Raw subsets + products for 1 to 3 metros | M0 | none | $0 on laptop [E] | D's Dell Precision 5770 | |
| Static hosting for PMTiles + JSON | Serve the web map | M4 | hours | $0 to $20/mo [E] | Cloudflare R2 or any S3-compatible storage + Vercel | |
| Domain | Public launch | M8 | minutes | ~$15/yr [E] | any registrar | |
| Open-Meteo commercial plan or ERA5 via Copernicus | Temperature input once commercial | M8 | days | [E] check pricing at the time | open-meteo.com | |
| Stripe | Payments | M8 | days | per-transaction fees | stripe.com | |

## 7. DECISION LOG
| Date | Decision | Why | Alternatives rejected | Revisit only if |
|---|---|---|---|---|
| 2026-11-05 | Build on NASA OPERA DISP-S1, not our own InSAR processing, for v1 | Free, validated, continent-wide [V]; gets to the awe demo in days | Processing raw Sentinel-1 ourselves (weeks, v2) | Resolution blocks paid reports in M6 |
| 2026-11-05 | Launch metro: Houston | USGS calls the Houston-Galveston area possibly the most subsidence-affected US metro [V]; local GPS network for validation [V] | D's home area (Triangle, NC: weaker signal [E]), Phoenix, Central Valley | M1 validation fails there for data reasons |
| 2026-11-05 | No database in v1; Parquet + PMTiles + static JSON | Cheapest, fastest, scales to several metros | Supabase/PostGIS from day one | Alerts or accounts need queries static files can't serve |
| 2026-11-05 | MapLibre GL + deck.gl on Next.js | Free, handles 3D extrusion and time animation | Mapbox (paid at scale), Cesium (heavier) | Performance test in M4 fails |
| 2026-11-05 | Bridges are the influence wedge; address reports are the money wedge | Bridge pre-collapse evidence exists [V]; UK proves consumers pay for reports [V] | Insurer-first B2B sale | Pilot conversations say otherwise |
| 2026-11-06 | M0 ingest uses its own CMR search and HTTPS range-read subsetting (`pipeline/ingest.py`) instead of `opera-utils disp-s1-download` | Read in opera-utils 0.25.9 source: its bbox path loads the frame database from GitHub releases (blocked in the cloud sandbox), converts the bbox with two corners only (a UTM-projected bbox is rotated, so its corners can be clipped), and over HTTPS downloads each full product before cropping (718 full products for Houston). DISP-S1 files are cloud-optimized HDF5 (4 MB pages), so reading only the bbox chunks is possible | opera-utils CLI as-is; download-then-crop (kept as `--method full` fallback) | Range reads prove slower or less reliable than full downloads in the first real run |
| 2026-11-06 | Houston M0 bbox `[-95.82, 29.76, -95.51, 30.03]` (NW Harris County: Katy, Cypress, Jersey Village); frames 38238 (descending, path 143) and 8882 (ascending, path 34); frame 38239 excluded | Measured from NASA CMR: both frames cover 100% of the bbox, so both orbit directions exist for M2; 38239 covers 0% and has only 14 granules (2016) | Downtown/east Harris box (historic subsidence, slower now [E]) | M1 finds fewer than 10 GPS stations in the box, or no subsidence signal |
| 2026-11-06 | D asked to see what the app looks like before any real download: a clickable HTML prototype (`design/prototype.html`, published as a private claude.ai artifact) of the M4 map, M6 address report and M5 bridge watch, outside the milestone ladder | Lets D judge the product shape before M1 to M6 are built. It uses a fictional neighbourhood (Larkin Bayou) and simulated numbers, labelled SIMULATED on every screen, so the "no placeholder numbers" rule for real outputs is not broken. It already follows the product rules: provenance tags, error bars, confidence, wording | Waiting for M4 to show anything; mock-up over real Houston addresses (would look like real measurements) | D wants changes to the product shape: record them here before M4 |
| 2026-11-06 | Earthdata credentials: also accept an Earthdata user token in `EARTHDATA_TOKEN` (sent only to the ASF data hosts). `.netrc` and `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` still work. In Claude Code cloud sessions the token is a saved network secret the proxy injects | A token avoids putting a password on disk; still "env vars or .netrc only" | Password in env vars in the cloud session | ASF stops accepting tokens |

## 8. OPEN QUESTIONS
- ~~Which DISP-S1 frame IDs cover the Houston bbox, and do ascending and descending both exist there?~~ Answered 2026-11-06 from NASA CMR: F38238 (descending, path 143, 449 epochs 2016-07-24 to 2026-03-22) and F8882 (ascending, path 34, 269 epochs 2016-10-09 to 2026-02-19), each covering 100% of the bbox. Both geometries exist.
- Do we validate with `displacement` (full) or `short_wavelength_displacement`? Expected answer: full displacement for GPS comparison, short-wavelength for local display. Confirmed in M1.
- What error figure can we honestly publish? Answered by M1 and M2 measurements.
- What price converts for a US address report? Answered in M8 with a price test.
- Can a pre-collapse signal be seen in US OPERA data, not just European EGMS data? Find a US case during M5.

## 9. CURRENT MILESTONE: WORKING CHECKLIST (M0)
- [~] D: Earthdata login created and credentials in place (token saved; ASF app authorization pending)
- [x] Scaffold repo per Section 3.4, add `CLAUDE.md` with the Rules section
- [x] `config/regions.yaml` with Houston bbox (start with a ~30 x 30 km box over an area of known subsidence)
- [x] Find frame IDs covering the bbox; record both orbit directions if they exist (38238 desc, 8882 asc)
- [x] `pipeline/ingest.py` bbox download (own range-read subsetting instead of `opera-utils`, see Decision Log); skip already-downloaded files
- [ ] Download matching DISP-S1-STATIC files (code done and tested offline; real download blocked)
- [~] Print summary: epochs, date range, % valid pixels per frame (code done and tested offline; real numbers pending)
- [x] pytest: ingest path logic on a fake file list (no network)
- [x] Update Status block and Session Log

## 10. SESSION LOG (newest first)
- 2026-11-06 (session 1, later): fixed 13 confirmed findings from an adversarial review of `pipeline/ingest.py` (georeferencing of subsets, bbox-aware rerun skip, corrupt-file repair, netrc scoping, worker shutdown, retries, provenance tags); tests 30 -> 47, all pass | at D's request built a clickable HTML prototype with simulated data (`design/prototype.html`, private artifact) | M0 still blocked on ASF authorization
- 2026-11-06 (session 1): created `vigil` cloud environment (allowlist: cmr.earthdata.nasa.gov, urs.earthdata.nasa.gov, datapool.asf.alaska.edu, cumulus.asf.alaska.edu, cumulus.asf.earthdatacloud.nasa.gov) and saved an Earthdata token as a network secret | scaffolded repo, wrote `pipeline/ingest.py` + `pipeline/schemas.py` + 30 synthetic-data tests (all pass) | found frames 38238/8882 (both 100% bbox cover), dry run plans 718 products + STATIC for both frames | measured DISP-S1 batch lag (newest epoch 2026-03-22, produced 2026-08-17) and post-2021 sampling drop, added to Risk Register | adversarial code review running | blocked: product requests still redirect to Earthdata Login | next: D authorizes ASF app, then real M0 run
- 2026-11-05: spec written in chat | no code yet | next: M0

## 11. PARKING LOT
Good ideas, not for now. Do not chase until the current milestone passes.
- v2: building-level resolution by running dolphin [V] on the 5 m x 10 m CSLC inputs [V] for paid-report areas
- NISAR L-band displacement (sees through vegetation better) once its DISP product ships
- Moonshot: the whole continent live, with a public "structures moving abnormally this month" feed
- Sinkhole early warning layer (Florida)
- Rail lines, pipelines and levees as asset types
- Insurer API (bulk address scoring)
- Engineering report for this project (D's project-reports plan): every figure tagged MEASURED / DERIVED / ESTIMATED, which this pipeline already produces
