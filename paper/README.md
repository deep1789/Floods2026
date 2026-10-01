# Nepal flood & weather study: layout

* `corrected/PAPER.md` — **current paper** (corrected discharge). Tables in `corrected/tables*/`, figures in `corrected/figures/`, results in `corrected/results/`.
* `PAPER_PLAN.md` — earlier version on the uncorrected V2 discharge (superseded; kept for the before/after comparison).
* `READINESS.md` — what is and is not fit for publication.
* `floodlab/` — pipeline package; `tests/test_leakage.py` — leakage tests.
* `run_*.py`, `evaluate*.py`, `analyses.py`, `ablation.py` — experiments (configurable via `FLOOD_*` environment variables, see `floodlab/config.py`).
* `fetch_extended.py`, `scan_from_cache.py`, `build_corrected.py` — GloFAS cell scan and corrected panel; `build_extended.py`, `extended/run_all.sh` — record extension (needs weather-API quota; untested).
* `build_paper.py` — assembles `corrected/PAPER.md`; reproduction commands are in Section 10.1b of the paper.

## Resuming the record extension

`build_extended.py` was stopped partway because the Open-Meteo daily quota was exhausted (2 of 10 locations done). The fetched responses are in `extended/api_cache.tar.gz`; the header of `build_extended.py` explains how to restore them and continue. After it writes `extended/panel_2010_2026.csv`, run `bash extended/run_all.sh` (about 1-2 hours) and then adapt `build_paper.py` to the new tables.
