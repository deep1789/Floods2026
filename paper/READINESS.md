# Are the results fit for writing a paper? Assessment

*Based on the code and results in this repository at the commit that contains this file.*

## Verdict

**Fit for a carefully scoped dataset-and-benchmark paper, with its limits stated. Not fit for a paper that claims forecasting skill for Nepal's rivers.**
Two blockers remain and both need network access that this sandbox does not have; they are listed first below.

## What now stands up (can be written as findings)

| Finding | Evidence | Strength |
|---|---|---|
| Data audit: discharge scale, quantised low flows, intermittent zero flow, elevation inconsistency, same-name rivers | Section 3.4, Table 4 | Solid as *observations of the file*; the *cause* of the scale mismatch is unverified |
| Leakage-tested protocol with hard baselines | `tests/test_leakage.py` (3 tests incl. a negative control), Algorithms 1–4 | Solid |
| Persistence is very hard to beat (median NSE 0.953 / 0.867 / 0.752 at 1 / 3 / 7 d) | all splits | Solid |
| After nested tuning, boosting, LSTM, Transformer-style and graph networks are within ~0.02 log-NSE of each other and significant against persistence at only 1–4 of 10 sites | Tables 14–15, Section 8.10 | Solid within this data; tuning effort differs by model |
| Tuning changed untuned conclusions (boosting NSE +0.037 at 3 d, +0.116 at 7 d) | Section 8.10 | Solid; a methodological warning |
| Per-site ridge distributed-lag model beats persistence at 10/10 sites under leave-one-monsoon-out | Table 15 | Solid |
| No detectable soil-moisture effect (event model p = 0.91; ablation; permutation importance) | Sections 8.8, 8.10 | Solid for *this modelled system*; not a statement about real catchments |
| Explicit river connectivity does not help; no upstream-leads-downstream signal beyond common rain | Sections 8.8, 8.10 | Solid but weakly powered (10 sites, 2 physical edges) |
| Sept 2024 storm captured; Thame GLOF, Bhote Koshi flash flood invisible; same conclusion from residuals and an isolation forest | Sections 8.5, 8.9, 8.10 | Solid, but the negative controls are three events |
| Tree models cannot extrapolate to the storm (predicts ~25 % of observed peak) | Section 8.10, Table 36 | Solid |
| Prediction intervals under-cover on high-flow days (54–74 % for nominal 90 %) | Section 8.6 | Solid |

## Blockers (cannot be fixed offline)

1. **The target may not be the river.** Mean modelled discharge at Karnali/Chisapani is 0.8 m³/s, Narayani/Devghat 2.3, Saptakoshi/Chatara 1.5, against published long-term means of order 10^3 m³/s. This is most likely a small tributary cell. Without a check, a reviewer will reject any claim about "the river".
   *Fix:* run `python fetch_extended.py scan` where `flood-api.open-meteo.com` is reachable (it scans a 5×5 grid of GloFAS cells around each coordinate and compares them with the dataset), or obtain DHM gauge series for the ten stations.
2. **The record is ~4.7 monsoons.** Even leave-one-monsoon-out has only four held-out seasons; extreme-value statements are descriptive only.
   *Fix:* `python fetch_extended.py extend 1990 2022` (GloFAS discharge reanalysis starts in 1984; ERA5-based weather is longer). This would give ~30 more monsoons and would make most of the planned statistics meaningful.

**Network blocker:** the environment's network policy denies `flood-api.open-meteo.com` (and, untested, `archive-api.open-meteo.com`). Allow those hosts under the environment's network settings and these two steps can be run and the paper re-run end to end (`PAPER_PLAN.md` build takes the new data once the loaders are pointed at it; the pipeline is otherwise unchanged).
`fetch_extended.py` is **untested** here; verify the Open-Meteo variable names against the current API reference.

## Other limitations to keep in the paper

- Discharge is modelled and partly circular with the weather inputs (Section 3.5).
- The Transformer-style and graph models are simplified variants, not reference implementations; the LSTM autoencoder and TreeSHAP were not run.
- Tuning was bounded and applied to boosting and the LSTM only; ARX has built-in ridge selection; TFT-lite and graph use defaults.
- Leave-one-location-out is a transfer test (the held-out site's own recent flow is an input), and its full-period variant is optimistic.
- References were written from memory and must be verified.
- The manuscript is a Markdown working paper, not in a journal template.

## Suggested route

1. Allow the two hosts; run the cell scan; decide whether the corrected cell series replaces the dataset series.
2. Extend the record; re-run `run_benchmark.py`, `run_lstm.py`, `run_tune.py`, `run_extra.py`, `run_extras2.py`, `evaluate*.py`, `analyses.py`, `ablation.py`, `build.py`.
3. Re-check every number quoted in the text against the regenerated tables (the text quotes values by hand).
4. Verify references; reformat for the target venue (dataset-and-benchmark track, or a hydrology methods journal if the gauge validation succeeds).
