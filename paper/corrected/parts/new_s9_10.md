## 9. Threats to Validity, Ethics and Limitations

**Construct validity.** The target is a modelled discharge, not a gauge record (Section 3.5), and its identity depends on a processing choice we found to be wrong in the original file (Section 3.4). The corrected series is a better, but still unvalidated, representation of the named rivers (the Khokana and Kusum series in particular still look erratic at low flow, Section 3.4): no upstream area or gauge series was available; two replacement cells (Khokana and Rasuwagadhi) lie at the corner of the scanned block; the "highest mean flow in the block" rule could in principle select a different river near a confluence; and weather and soil moisture are still taken at the original coordinate, up to about 0.11° from the discharge cell. Until the corrected cells are checked against DHM gauge series, we interpret results as properties of the corrected *modelled* data. A future version of the dataset should record the GloFAS cell used and its upstream area.

**Internal validity.** The principal risks are temporal leakage (mitigated by Algorithm 1 and three automated leakage tests, one of which is a negative control), selection of thresholds on test data (fixed from training) and seed- or split-dependent conclusions (mitigated by multiple splits, 5–10 seeds and bootstrap intervals). The ten series are not independent: the effective sample size for monsoon-level conclusions is a handful of seasons. The replacement-cell rule was chosen after looking at the scan, using the data themselves (flow magnitude), and not on any forecast skill.

**External validity.** Four years (about 4.7 monsoon seasons by the end of August 2026, with the 2026 monsoon incomplete) is a short record that includes at least one record-setting storm. Conclusions about the typical or the extreme must be framed accordingly. Results for ten locations cannot be extrapolated to all Nepali rivers, particularly the snow- and glacier-dominated headwaters.

**Statistical conclusion validity.** Metric choice affects rankings (Section 8.4, items 4 and 5); we report both raw and log-space metrics plus event metrics. Multiple comparisons across locations, horizons and models are corrected by false-discovery-rate control within each split scheme. Hyperparameters were tuned only for boosting and the LSTM, with a bounded nested search (Section 8.5); the Transformer-style and graph networks use default settings, so conclusions about the relative merit of model families remain conditional on the search effort spent on each. The tuning of the LSTM did not transfer to the test period, which indicates that the inner validation signal is noisy.

**Data limitations.** Weather values are model/reanalysis-based; precipitation in steep terrain can be strongly biased, and soil moisture is a model variable. The elevation field has a plausibility problem at Chisapani (A7). Zeros in precipitation are genuine and retained.

**Ethical and societal considerations.** Flood information is safety-relevant. The paper states explicitly that the models are research tools and not warning systems; that they cannot detect glacial-lake outburst, landslide-dam or avalanche-triggered floods; and that the dataset is not a substitute for DHM's official monitoring and forecasting. We avoid any presentation that could be mistaken for an operational forecast and avoid publishing a ranked "most dangerous river" list derived from modelled magnitudes. Open-Meteo, GloFAS and DHM are credited according to the licence terms. Death and displacement figures from the 2024 floods are cited only from official reports.

**Negative results are reportable.** That model complexity buys little over persistence, that tuning of the LSTM did not help, that soil moisture shows no effect and that connectivity does not help robustly are findings of this study and are reported as such.

---

## 10. Implementation Status, Reproduction and Remaining Work

### 10.1 Implementation status

**Table 131. What was implemented and run, and what remains.**

| Plan item | Status | Where |
|---|---|---|
| Audit tests A1–A3 as code; A4 as a reproducible scan | done | `floodlab/data.py`, `fetch_extended.py scan`, `scan_from_cache.py` |
| Corrected discharge panel | done (cached scan responses; heuristic cell rule; **not validated against gauges**) | `build_corrected.py` |
| Algorithm 1 (leakage-safe features) + leakage tests | done; 3 tests pass on the corrected panel | `floodlab/features.py`, `tests/test_leakage.py` |
| Algorithm 2 (purged blocked CV) | done (LOMO with 7-day pre- and 30-day post-purge) and forward chaining | `run_benchmark.py` |
| Algorithms 3 and 4 (events, stationary bootstrap) | done | `floodlab/events.py`, `metrics.py` |
| Baselines B0–B3 | done | `floodlab/models.py` |
| Pooled boosting, quantile boosting | done; default and nested-tuned versions | Sections 8.4, 8.5 |
| Joint LSTM, 10 seeds | done; default and nested-tuned versions | `floodlab/lstm.py` |
| Transformer-style network, graph models | done as **simplified variants** (TFT-lite, three graph variants), default settings, 5 seeds; the reference TFT was not used | `floodlab/nets.py` |
| Chronological, LOMO and forward-chaining splits | done | Section 8.4 |
| LOLO | done in the chronological variant and in an optimistic full-period variant | Section 8.6 |
| Metrics, DM tests with BH, block-bootstrap intervals, event and quantile scores | done | `evaluate.py` |
| Lag weights, soil-moisture mixed model, ablation, POT/GPD, joint extremes, upstream–downstream, recession | done | `analyses.py`, `ablation.py` |
| Grouped permutation importance, scenario analysis, isolation forest, smearing check | done | `run_extras2.py` |
| TreeSHAP, LSTM autoencoder, wavelet coherence, profile-likelihood GPD intervals, Pettitt/CUSUM tests | **not run** | — |
| Validation of the corrected cells against DHM gauges; catchment attributes | **not done**: no gauge data available | Section 7.11 |
| Record extension (2010–2022) with corrected discharge | **scripts written, not run**: the weather API's daily quota was exhausted; the discharge API works | `build_extended.py`, `extended/run_all.sh` |

### 10.1b Reproducing the results

```
pip install pandas numpy scipy scikit-learn statsmodels matplotlib torch
cd paper
python fetch_extended.py scan          # needs flood-api.open-meteo.com; writes the cached cell scan (about 250 requests)
python scan_from_cache.py              # summarises the scan from the cache
python build_corrected.py              # builds corrected/panel_corrected_2023_2026.csv from the cache
bash corrected/run_all.sh              # leakage tests, benchmark, tuning, LSTM, extra arms, analyses, evaluation (about 1 hour)
cd corrected && FLOOD_CSV=$PWD/panel_corrected_2023_2026.csv python ../prelim.py && python ../make_tables.py
cd .. && python build_paper.py         # assembles corrected/PAPER.md
```

Run the heavy scripts one at a time; they use all CPU cores and slow each other down badly if run together. The original uncorrected analysis is reproduced by the same scripts without `FLOOD_CSV`.

### 10.2 Remaining work

1. Validate or correct the replacement cells against DHM gauge discharge (or published long-term means), at least at Chisapani, Devghat, Chatara and Khokana.
2. Extend the record when the weather API quota allows (`build_extended.py`, then `extended/run_all.sh`); about 13 more training monsoons would make leave-one-monsoon-out and the extreme-value analyses meaningful.
3. Run the unrun items of Table 131, and re-check every number quoted in the text against the regenerated tables (the text quotes values by hand).
4. Verify the references, and reformat for the target venue (a dataset-and-benchmark track is the natural fit).

### 10.3 Suggested venues

Hydrology and water-resources journals that accept data-driven studies, and data-focused venues if the audit, correction and benchmark are packaged as a dataset-and-benchmark paper. A datasets paper would put more weight on Sections 3 and 8.9; a methods paper on Sections 4, 5 and 7.

---

