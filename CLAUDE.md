# Vigil: instructions for Claude Code

The source of truth for this build is [`BUILD_SPINE.md`](BUILD_SPINE.md). Read the whole file before writing code,
and open every session with its Status block in 2 lines.

## RULES FOR CLAUDE CODE (copied from the spine; the spine wins if they ever differ)
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

## Working commands
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # synthetic data only, no network
python -m pipeline.ingest --region houston --find-frames   # list DISP-S1 frames over the bbox (no login needed)
python -m pipeline.ingest --region houston --dry-run       # show what would be downloaded
python -m pipeline.ingest --region houston                 # M0 acceptance command
```

## Earthdata credentials (never commit them)
`pipeline/ingest.py` looks for credentials in this order:
1. `EARTHDATA_TOKEN` env var (an Earthdata Login user token, sent only to the ASF data hosts)
2. `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` env vars, or a `~/.netrc` entry for `urs.earthdata.nasa.gov`
   (sent only to Earthdata Login)
3. Nothing local: in a Claude Code cloud environment, the network proxy can inject a saved token for
   `cumulus.asf.earthdatacloud.nasa.gov`.

The Earthdata account must have authorized ASF's download application once (open any product URL in a
browser while logged in and approve).
