## 8. Results (corrected discharge)

All numbers in this section are produced by code in `paper/` on the corrected panel and are reproducible with the commands of Section 10.1b. Sections 8.1–8.3 are descriptive; 8.4–8.6 report the forecasting benchmark, the model arms and the transfer experiment; 8.7 reports the process analyses; 8.8 the documented events; 8.9 what changed relative to the uncorrected series; and 8.10 a discussion.

### 8.1 Annual maxima and the September 2024 storm

**Table 102. Annual maximum modelled discharge (m³/s), corrected series.** 2026 includes data only to 31 August.

{{T:annmax}}

The 2024 annual maximum is the largest in the panel at six of the ten locations. The exceptions are Bahrabise (14 August 2023), Belsot (16 August 2025), Chameliya/Nayalbadi (12 July 2025) and Rasuwagadhi (18 July 2026). The all-time maximum falls on 28 September 2024 at Bhada Bridge, Khokana and Kusum and on 29 September at Chatara, but at Devghat and Chisapani it falls earlier, on 7 and 8 July 2024, so the September storm was not the largest event everywhere. At Chameliya/Nayalbadi only 7.2 mm fell over 26–28 September, because far-western Nepal lay outside the storm's footprint.

**Table 103. The 27–29 September 2024 storm (positive control).** "Pre-event Q" is the median of 15–24 September; the peak is the maximum over 20 September–8 October; the lag is from the largest precipitation day (24–30 September) to the peak discharge day.

{{T:event}}

![Figure 7. The late-September 2024 storm at three locations.](figures/fig5_event2024.png)

*Figure 7. Precipitation (bottom) and discharge normalised by its pre-event level (top) at Khokana (Bagmati), Devghat (Narayani) and Chatara (Saptakoshi).*

The data capture the rainfall-driven event: three-day precipitation of 120–328 mm at nine of ten locations, and discharge peaks of 1.2 to 38 times the pre-event flow. With the corrected series the large rivers are visible as well: the Saptakoshi at Chatara peaks at 8,649 m³/s (1.36 times its 99th percentile) two days after the rain peak, the Narayani at Devghat at 7,888 m³/s (1.23 times; its record in this panel, 9,098 m³/s, is on 7 July 2024), and the Bagmati at Khokana rises 38-fold to 1,632 m³/s, 4.0 times its 99th percentile. The Department of Hydrology and Meteorology reported flows above historical levels on the Bagmati, Narayani and Sunkoshi; the modelled panel is consistent for the Bagmati and, more weakly, for the Bhote Koshi at Bahrabise (1.15 times the 99th percentile), but its four-year record cannot test the statement for the Narayani. At Kusum the 28 September peak of 3,176 m³/s occurs on a day with only 8.4 mm of local rain, following 117.6 mm the day before, which shows that point precipitation at the coordinate is an incomplete description of upstream catchment forcing.

### 8.2 Dependence between locations

The cross-site correlation of $\log(1+Q)$ ranges from 0.60 (Belsot–Khokana) to 0.98 and averages 0.85, dominated by the shared seasonal cycle, so ten locations are far fewer than ten independent samples. The correlation of $\log(1+P)$ is lower and more spatially structured: Bhada Bridge–Kusum 0.88, Bhada Bridge–Chisapani 0.87 and Chatara–Belsot 0.85 are the strongest pairs, consistent with neighbouring western and eastern groups, and Rasuwagadhi is the most isolated (0.46–0.61 with every other location). This structure motivates the leave-one-location-out experiment and block bootstraps that respect dependence.

### 8.3 Rainfall–discharge association

**Table 104. Correlation between $\log(1+P_t)$ and $\log(1+Q_{t+k})$ (raw, not prewhitened), corrected series.**

{{T:lag}}

![Figure 6. Lag correlation heat map.](figures/fig3_lagcorr.png)

*Figure 6. Lagged correlation between log precipitation and log discharge, by location (rows ordered by elevation as in Table 1). Khokana responds within one day and then decays; most others keep increasing to the largest lag examined, which reflects seasonality and not a long response time (Section 8.7).*

Khokana is again the only location where the correlation peaks and then falls (maximum 0.87 at $k=1$). Elsewhere the correlation rises or plateaus out to $k=7$, which the prewhitened analysis of Section 8.7 shows to be a seasonal artefact.

### 8.4 Forecasting benchmark

The implementation (`paper/floodlab/`) follows Algorithms 1–4 and the protocol of Section 6. Before any model was run, `tests/test_leakage.py` passed three tests on the corrected panel: (a) all 28 features at origins up to a cutoff are unchanged when every observation after the cutoff is randomly rescaled; (b) the predictions of a model fitted before the cutoff are unchanged by the same perturbation; and (c) a deliberately leaky feature (a centred rolling mean) *is* detected, so the test can fail. All fitted quantities (soil-moisture and discharge climatology, recession constants, thresholds, scalers) use the training period only.

**What was run.** Baselines B0–B3; a pooled `HistGradientBoostingRegressor` on the increment $\Delta y_{t+h}$ (28 features plus site and elevation; default 400 iterations, learning rate 0.05, 15 leaves, $L_2=1$) and a nested-tuned version (Section 8.5); quantile versions of the boosted model at $h=1,3$; a joint multi-site LSTM (30-day window, 12 dynamic inputs, site embedding, three multi-horizon heads, early stopping on the most recent 15 % of training dates; 10 seeds averaged) and a tuned version; a simplified Transformer-style network ("TFT-lite") and three graph networks (5 seeds averaged). Splits: chronological (train to 31 August 2025, test 1 September 2025–31 August 2026); leave-one-monsoon-out (LOMO) over June–September of 2023–2026 with a 7-day purge before and a 30-day purge after each held-out block; forward chaining for the 2025 and 2026 monsoons (which the sequence models use in place of LOMO); and chronological leave-one-location-out (LOLO). Statistical comparisons use the Diebold–Mariano test with a Newey–West variance on squared log-space errors, Benjamini–Hochberg control within each split scheme, and the paired stationary bootstrap (Algorithm 4).

**Table 105. Median across the ten locations of NSE (raw $Q$) and log-NSE ($\log(1+Q)$) at horizons $h=1,3,7$ days.** B2 is the seasonal climatology, B3 the per-site ridge distributed-lag model, B1 recession-persistence; `_tuned` rows use the nested search of Section 8.5 and exist for the chronological split and the 2026 forward-chaining fold only. The sequence models were not run under LOMO (blank cells).

{{T:main_all}}

**Table 106. Number of locations (of 10) at which a model beats persistence in log-space MSE, and how many of those differences are significant after BH correction.** "Sig." means $q<0.05$ on the Diebold–Mariano test.

{{T:beat_persistence}}

The results, with the corrected discharge, are as follows.

1. **Persistence is a very strong baseline** (chronological split: median NSE 0.984, 0.949 and 0.893 at 1, 3 and 7 days; log-NSE 0.996, 0.981 and 0.944), stronger than in the uncorrected series because the large rivers are smooth.
2. **The learned models are close to one another.** On the chronological split the median log-NSE of boosting (default and tuned), the LSTM, TFT-lite and the three graph networks lies between 0.996 and 0.997 at one day, 0.980 and 0.988 at three, and 0.947 and 0.966 at seven; the corresponding values for persistence are 0.996, 0.981 and 0.944. Median NSE ranges from 0.904 (default boosting) to 0.968 (physical-adjacency graph) at three days and from 0.854 (tuned LSTM) to 0.922 (tuned boosting) at seven, so raw-flow NSE separates them somewhat more than log-NSE does.
3. **Most models beat persistence at most locations but rarely significantly.** On the chronological split every model beats persistence at 7–10 of 10 locations (Table 106). The differences are significant at one location at one day for most learned models (three for the ridge model and the learned-adjacency graph), and at between 0 and 5 locations at seven days (the learned-adjacency graph is highest with 5, the ridge model and default boosting 0). The recession-persistence baseline B1 is significantly better than persistence at 8 locations at one day and 6 at three days, although its median scores equal those of persistence; this arises from small, consistent gains at many locations.
4. **Under leave-one-monsoon-out the ridge model is the most reliable.** It beats persistence at 9, 10 and 10 of 10 locations at 1, 3 and 7 days and significantly at 8, 8 and 3 of them; boosting does so at 9, 9 and 10 (significant at 6, 2 and 3). In raw-flow NSE, however, the LOMO medians show no gain at three days (persistence 0.802, ridge 0.801, boosting 0.802) and a loss for the ridge model at seven (0.553 against 0.585), while log-NSE shows gains (0.917 and 0.926 against 0.897 at three days; 0.742 and 0.789 against 0.672 at seven). The monsoon-only evaluation blocks are dominated by a few large peaks, so NSE and log-NSE answer different questions, and both must be reported.
5. **The result depends on the held-out year and on the fold scheme.** Under LOMO the largest gains over persistence at seven days occur for the 2024 monsoon (boosting log-NSE 0.812 against 0.606) and 2025 (0.745 against 0.591); in 2026 the gain is small (0.833 against 0.802). Under forward chaining (Table 105, "Forward chaining" rows) the default boosting and LSTM models are below persistence in raw NSE at seven days (0.517 and 0.512 against 0.621), while TFT-lite (0.724) and the learned-adjacency graph (0.708) are above it.
6. **Seed variability is large relative to model differences.** Across ten seeds the LSTM's chronological median log-NSE at seven days ranges from 0.934 to 0.965 (mean 0.952, standard deviation 0.011), so single seeds fall both below and above persistence (0.944); the ten-seed ensemble (0.964) is better than the typical seed. At three days the range is 0.971–0.987 (persistence 0.981).

**Table 107. Seed variability of the LSTM (median log-NSE across locations, chronological split).**

{{T:lstm_seeds}}

**Table 108. Median log-NSE across locations for each held-out monsoon (LOMO for the baselines, ridge and boosting; forward chaining for the sequence models).** Blank cells are folds for which the model was not run.

{{T:by_fold}}

![Figure 8. MSE skill relative to persistence at $h=3$, by location, chronological split (left) and leave-one-monsoon-out (right).](figures/fig12_skill_sites.png)

*Figure 8. Skill relative to persistence in log-space MSE at $h=3$. Positive values mean the model beats persistence. The horizontal scales differ between panels.*

**Per-location detail.** Khokana remains the location where learned models help most (chronological split, NSE at three days: persistence −0.071; ridge 0.373; boosting 0.352; LSTM 0.361) because its discharge is flashy even after correction. Where persistence is already near 0.98 (Chameliya/Nayalbadi, Rasuwagadhi, Chisapani) nothing improves on it meaningfully. Bhada Bridge and Kusum, the two locations with the highest coefficients of variation after Khokana, show the largest disagreement between models: at seven days the ridge model's NSE is only 0.028 at Bhada Bridge and 0.251 at Kusum, against 0.688 and 0.637 for persistence, while its log-NSE is higher than persistence's at both. A model can therefore lose at the peaks while winning on the baseflow, and reporting a single number would hide this.

**Table 109. Per-location NSE and log-NSE at $h=3$, chronological split.**

{{T:site_chrono_h3}}

**Table 110. Per-location NSE and log-NSE at $h=7$, chronological split.**

{{T:site_chrono_h7}}

**Event-based skill.** Using the training-period 95th percentile of each location as the threshold, the test window contains 27 declustered observed events across the ten locations (Algorithm 3, $r=3$ days). Persistence "detects" every event at one day (event POD 1.00) because a high flow persists into the next day, with a day-level false-alarm ratio of 0.20 and CSI of 0.66. The learned models have lower event POD at one day (0.70–0.89) and comparable false-alarm ratios (0.12–0.21), and the best day-level CSI at one day belongs to the graph network with physical adjacency (0.694) and the LSTM (0.692) against 0.661 for persistence. At three days the tuned boosted model has the best CSI (0.510 against 0.473 for persistence) and a false-alarm ratio of 0.295 against 0.364. Timing errors are 0.9–1.3 days at one day for the ridge, LSTM, TFT-lite and graph models but 2.6–2.7 days for boosting, whose forecast peaks therefore lag. With 27 events these differences are not statistically resolvable.

**Table 111. Event-based scores, chronological split (threshold = training-period 95th percentile).**

{{T:events}}

**Probabilistic skill.** The quantile boosted model gives 90 % intervals with a coverage of 0.865 at $h=1$ and 0.848 at $h=3$, slightly below nominal, and 0.847 and 0.815 on days above the training 95th percentile, so the intervals are close to equally reliable at high flows, unlike in the uncorrected series. The approximate CRPS is 0.036 (one day) and 0.076 (three days) in log units. Coverage by location ranges from 0.83 to 0.92 at one day and from 0.78 to 0.90 at three.

**Table 112. Quantile forecast evaluation (chronological split).**

{{T:prob}}

![Figure 9. Quantile forecasts at Khokana and Devghat.](figures/fig13_intervals.png)

*Figure 9. Five-to-ninety-five percent prediction intervals and median forecast of the quantile boosted model for $h=1$ over the test window, with the corrected modelled discharge in black.*

### 8.5 Tuning, the Transformer-style network and the graph networks

**Nested hyperparameter search.** We ran a bounded search in which the final test period was never touched: the inner folds are the 2023, 2024 and 2025 monsoons *inside the training period* (before 1 September 2025), each held out in turn with the same purges, and the criterion is the median across locations of log-space NSE. For boosting, 30 random configurations plus the default were evaluated at each horizon; for the LSTM, the default plus 11 random configurations of hidden size, dropout, learning rate and weight decay, three seeds each, on a forward-chained inner fold (train before June 2025, validate on the 2025 monsoon). For boosting, the selected configurations are shallow and regularised (learning rate 0.02, 7 leaves, minimum leaf size 50, 200 iterations at one and three days; learning rate 0.1, 7 leaves, minimum leaf size 10, $L_2=10$ at seven). The inner score gain over the default is 0.005 at one day (0.982 against 0.977), 0.017 at three (0.909 against 0.891) and 0.003 at seven (0.755 against 0.752), and the 31 candidates span 0.974–0.982, 0.880–0.909 and 0.720–0.755. Tuned results exist only for the chronological split and the 2026 forward-chaining fold, because tuning on seasons that later serve as LOMO folds would leak.

**Tuning helped boosting and did not help the LSTM.** On the chronological test window the tuned boosted model has a median NSE of 0.989, 0.965 and 0.922 at 1, 3 and 7 days against 0.990, 0.904 and 0.878 for the default configuration, and a median log-NSE of 0.997, 0.984 and 0.951 against 0.997, 0.980 and 0.947; it improves on the default at 9 of 10 locations at three days and 8 of 10 at seven (median log-NSE gain +0.002 and +0.005). Without tuning, boosting looked worse than persistence at three days in raw NSE (0.904 against 0.949); with tuning it is better (0.965). The LSTM configuration chosen on the inner fold (hidden size 64, dropout 0.4, learning rate 0.003, weight decay 0.01) did worse on the test window than the default (median NSE 0.943 and 0.854 against 0.963 and 0.905 at three and seven days; log-NSE 0.984 and 0.958 against 0.987 and 0.964). The inner scores of the LSTM candidates range from 0.846 to 0.896 on a single validation monsoon, which reflects configuration-and-seed noise and did not transfer. *Tuning can therefore change the ranking of model families, and the effect differs by family.*

**Transformer-style and graph arms.** The reference Temporal Fusion Transformer was not used. We implemented a simplified variant ("TFT-lite": gated variable selection, an LSTM encoder, one self-attention layer over the 30-day window, a gated output head and a site embedding, without quantile outputs or known-future inputs) and a multi-site graph network (shared LSTM encoder per site, one graph-convolution layer; variants with no edges, a fixed physical adjacency with two edges, Rasuwagadhi → Devghat and Bahrabise → Chatara, and a learned adjacency), each with 5 seeds averaged and default settings. On the chronological split TFT-lite is equivalent to the LSTM (median log-NSE difference 0.000, −0.001 and −0.002 at 1, 3 and 7 days, better at 5, 4 and 3 of 10 locations). Under forward chaining its median scores are higher than the LSTM's (log-NSE 0.961 and 0.870 against 0.942 and 0.793 at three and seven days), but at the level of individual locations it is better at only 5 of 10 at those horizons, so we do not read this as a robust advantage. **Explicit river connectivity does not help robustly.** Relative to the same network with no edges, the physical adjacency changes the median log-NSE by 0.000, +0.001 and 0.000 at 1, 3 and 7 days on the chronological split (better at 5, 6 and 5 of 10 locations) and by −0.001, −0.005 and +0.006 in forward chaining; the learned adjacency is +0.018 at seven days in forward chaining (9 of 10 locations) and about zero on the chronological split. With only four of ten locations connected by the physical graph, we do not expect a large effect, and the data do not show one.

**Table 113. Paired comparisons of model arms: median difference in site-level log-NSE and the number of locations where the first model is better.**

{{T:arm_comparisons}}

The event-based scores (Table 111) and Table 106 show the same plateau: all learned models are within a narrow band, each is better than persistence at most locations, and few of the differences are statistically distinguishable from persistence.

### 8.6 Transfer to an unseen location (leave-one-location-out)

The LOLO experiment trains on nine locations (chronological training period) and predicts the tenth, with no site identifier and no elevation as inputs. *It is not a prediction for an ungauged river:* the model still receives the held-out location's own recent discharge as an input. It tests whether the learned rainfall-to-increment mapping transfers across locations, which is a weaker claim than the ungauged-basin test of Kratzert et al. (2019).

**Table 114. LOLO results at $h=1$ (log-NSE), with the within-site pooled boosted model as the reference and the regime distance of Table 116.**

{{T:lolo_h1}}

**Table 115. LOLO results at $h=3$ (log-NSE).**

{{T:lolo_h3}}

Transfer costs little at the median: log-NSE falls from 0.997 (within-site boosting) to 0.995 at $h=1$ and from 0.980 to 0.969 at $h=3$, and the LOLO LSTM is at 0.993 and 0.964, against persistence at 0.996 and 0.981. The losses are concentrated: at $h=1$ Khokana loses 0.022 and Bhada Bridge 0.006; at $h=3$ Belsot loses 0.026 for boosting and the LOLO LSTM collapses to 0.831 there (persistence 0.968), and Chameliya/Nayalbadi loses 0.022. A regime distance computed from seven descriptors on the training period (four flow-duration-curve quantiles of $\log_{10}(Q/\tilde Q)$, the monsoon share of precipitation, the lag-1 autocorrelation of $\log(1+Q)$ and the standard deviation of daily log-increments; Table 116) singles out Khokana (6.19), then Rasuwagadhi (4.02) and Bhada Bridge (3.51); the others lie between 2.3 and 3.2. The rank correlation between regime distance and the LOLO loss is negative in three of four cases and not significant with ten locations (Spearman $\rho=-0.60$, 95 % bootstrap interval $[-0.99, 0.08]$ for boosting at $h=1$; $-0.27$ for the LSTM; $+0.02$ and $-0.33$ at $h=3$), so the data cannot establish that more distinctive locations transfer worse.

**Table 116. Regime descriptors used for the distance (training period).** fdc$p$: $\log_{10}$ of the flow-duration-curve quantile at $p$ divided by the median.

{{T:regime}}

**Table 117. Rank correlation between regime distance and the change in log-NSE under LOLO.**

{{T:lolo_corr}}

![Figure 10. LOLO change in log-NSE against regime distance.](figures/fig11_lolo.png)

*Figure 10. Change in log-NSE when a location is left out of training (LOLO minus within-site boosting, for both models), against its regime distance to the other nine locations. The correlation is not significant.*

**Full-period variant.** Training on nine locations over all dates and testing on the tenth over all dates, without site identity or elevation, gives a median log-NSE of 0.995, 0.985 and 0.968 at 1, 3 and 7 days against 0.995, 0.980 and 0.944 for persistence, ahead of persistence at 8, 8 and 10 of 10 locations (median NSE 0.975, 0.946 and 0.924 against 0.978, 0.930 and 0.877). This variant is optimistic: the training locations include the same dates as the held-out location, so concurrent storms are in the training data, and it should be read as spatial transfer under shared weather rather than out-of-sample prediction.

**Table 118. LOLO over the full period (median across held-out locations).**

{{T:lolo_full}}

