# Are the results fit for writing a paper? Assessment (after the discharge correction)

*Based on `corrected/PAPER.md` and the code and results in `corrected/`.*

## Verdict

**Fit for a carefully scoped dataset-and-benchmark paper (audit + correction + leakage-tested benchmark).**
**Not yet fit for a Q2 hydrology paper that claims forecasting skill for Nepal's rivers.** Two blockers remain.

## What changed since the previous assessment

* The main blocker, the discharge scale, is **diagnosed and largely fixed**. A scan of the 5 × 5 GloFAS cells around every coordinate showed that the dataset's own cell is reproduced exactly at all ten locations, and that at seven a neighbouring cell carries 17-1,565x more flow (Chisapani 0.8 -> 1,325 m3/s; Devghat 2.3 -> 1,609; Chatara 1.5 -> 1,829). This is itself a publishable data-quality finding with a reproducible scan and a corrected panel.
* The full pipeline was re-run on the corrected panel. Some conclusions are robust to the correction (learned models close to each other and to persistence; tuning matters for boosting; no soil-moisture effect; connectivity does not help; Thame GLOF and the flash flood are invisible) and some changed (stronger persistence baseline, upstream-downstream coupling now clearly present, better high-flow interval calibration). Table 38 of the paper lists both.

## What now stands up

| Finding | Strength |
|---|---|
| Grid-cell error in the V2 discharge at 7/10 locations, with scan and corrected panel | Solid (reproducible scan); the *replacement* cells are a heuristic |
| Leakage-tested protocol with hard baselines | Solid |
| Persistence is very hard to beat (median NSE 0.984 / 0.949 / 0.893 at 1 / 3 / 7 d) | Solid |
| Learned models within ~0.02 log-NSE of each other; beat persistence at 7-10 of 10 locations but significantly at 0-5 | Solid within this data |
| Ridge distributed-lag model beats persistence at 9-10/10 locations under leave-one-monsoon-out (significant at 8, 8, 3) | Solid |
| Tuning changes conclusions (boosting NSE at 3 d: 0.904 -> 0.965; LSTM tuning does not help) | Solid; a methodological warning |
| No detectable soil-moisture effect (event model p = 0.81; ablation; permutation importance) | Solid for this modelled system |
| Upstream-downstream residual correlation ~0.52 vs 0.11-0.17 for controls (appears only after correction) | Moderate (lag-0 co-movement; two pairs) |
| Sept 2024 storm stands out in residuals and an isolation forest; Thame GLOF and Bhote Koshi flash flood invisible | Solid, but only three events |

## Remaining blockers

1. **The corrected cells are unvalidated.** No gauge data were available. Khokana's and Kusum's series still look erratic at low flow (so they may still not be on the main channel), two replacement cells sit at the corner of the scanned block, and the weather is taken at the original coordinate (up to ~12 km from the discharge cell).
   *Fix:* validate against DHM gauge discharge (or published long-term means) at least at Chisapani, Devghat, Chatara and Khokana; widen the scan around Khokana, Kusum and Rasuwagadhi.
2. **The record is ~4.7 monsoons.** Extending it needs `archive-api.open-meteo.com` quota (the daily limit was exhausted for this environment's shared IP; the discharge API works).
   *Fix:* run `python build_extended.py` once the quota resets, then `bash extended/run_all.sh`. Both are written and **untested end to end**.

## Other limitations to keep in the paper

* Discharge is modelled and partly circular with the weather inputs.
* The Transformer-style and graph models are simplified variants (5 seeds, default settings); the LSTM autoencoder, TreeSHAP and the Pettitt/CUSUM tests were not run.
* The text quotes numbers by hand; re-check them against the tables after any re-run.
* References were written from memory and must be verified.
* The manuscript is Markdown, not a journal template.
