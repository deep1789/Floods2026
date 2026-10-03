### 8.7 Process analyses

**Rainfall–discharge lag structure (RQ2).** After prewhitening (regressing the daily log-increment on the current and ten lagged log-rainfalls, an error-correction term and seasonal harmonics, with a second-difference smoothness penalty chosen by blocked cross-validation), the modelled response is fast everywhere (Table 119, Figure 11). The peak weight falls at lag 1 day at seven locations, at lag 2 days at Belsot and Chameliya/Nayalbadi and at lag 0 at Bhada Bridge. The centroid of the positive weights is shortest at Rasuwagadhi (1.15 days; bootstrap interval 0.97–1.88) and Khokana (1.20; 0.97–1.91), then Chisapani (1.75; 1.62–2.56), Devghat (1.81; 1.49–2.44) and Chatara (2.01; 1.71–2.78), and longest at Chameliya/Nayalbadi (3.33; 2.32–4.05) and Bhada Bridge (3.72; 2.23–4.67). With the corrected series the three large rivers have centroids of 1.8–2.0 days with narrow intervals, whereas the uncorrected tributary cells gave 2.6–2.8 days. The intervals of the slower locations overlap (Belsot 2.50, Bahrabise 2.78, Kusum 2.85, Chameliya/Nayalbadi 3.33), so the data do **not** resolve an ordering among them. The raw correlations of Table 104, which kept rising out to seven days, therefore overstate the response time because of the common seasonal cycle. The cumulative response $\sum_j w_j$ is largest at Khokana (0.44; 0.36–0.58), then Kusum (0.18) and Bhada Bridge (0.17), and smallest at Rasuwagadhi (0.01; −0.00–0.02), where modelled discharge hardly depends on rainfall at the daily scale, consistent with a smoother regime driven by melt or storage in the model. We had expected high-elevation locations to respond more slowly, and the weights show the opposite for Rasuwagadhi; we flag this as unexplained and as a reason to treat the modelled response as a property of the model cell.

**Table 119. Distributed-lag summary (prewhitened).** $\lambda$ is the smoothness penalty selected by blocked cross-validation; intervals are 2.5–97.5 % from 200 stationary-bootstrap replicates.

{{T:lagweights}}

![Figure 11. Estimated lag weights.](figures/fig8_lagweights.png)

*Figure 11. Estimated weights $w_j$ on $\log(1+P_{t-j})$ in the daily log-increment of discharge, with 95 % bootstrap bands. The scales differ by location.*

**Antecedent wetness (RQ3).** We identified 310 rain events (declustered 3-day rainfall above each location's 90th percentile, Algorithm 3) and regressed the log amplification of discharge on the pre-event soil-moisture anomaly, a standardised antecedent-rain index, log event rainfall, pre-event flow relative to the median and a monsoon indicator, with a random intercept for location. **We find no evidence that wetter antecedent soil amplifies the modelled response:** the coefficient on the soil-moisture anomaly is −0.025 (standard error 0.103, $p=0.81$), a within-location permutation test gives $p=0.80$, and the per-location Spearman correlations between the anomaly and the model residual lie between −0.27 and +0.29 with all $p>0.1$. Event rainfall has a strong positive coefficient (2.10, $z=10.2$), the monsoon indicator is positive (1.37, $p<0.001$) and the standardised antecedent-rain index is negative (−0.40, $p=0.018$): for a given event rainfall, a wetter antecedent period is associated with a smaller relative rise, which is the opposite of the wet-catchment amplification hypothesis and may reflect that wet periods have higher baseflow. The forecasting ablation agrees with the null on soil moisture (Table 121): dropping the soil-moisture features from the pooled boosted model under LOMO changes the median log-NSE by 0.000 at $h=1$ and −0.003 at $h=3$, whereas dropping the current-rain features costs 0.001 and 0.012, the season terms 0.000 and 0.029, the temperature, humidity and snow group 0.000 and 0.014, and using flow lags alone costs 0.003 and 0.020. Dropping the antecedent-rain features *improves* the three-day score slightly (+0.004). Grouped permutation importance on the chronological test window gives the same ordering (Table 122): flow lags dominate at every horizon (106 %, 99 % and 113 % increase in MSE when permuted), current rain matters at one day (92 %) but not beyond (21 % at three days, 2 % at seven), the temperature, humidity and snow group matters at one and three days (21 % and 11 %), site and elevation at one day (23 %), season at longer horizons (14–15 %), and soil moisture (1.2 %, −2.0 %, −1.6 %) is indistinguishable from zero. TreeSHAP on a LightGBM model with the same features (Table 134) gives a similar ordering: flow lags 42 %, 38 % and 42 % of the mean absolute SHAP value at 1, 3 and 7 days, current rain 26 %, 21 % and 9 %, season 9 %, 17 % and 24 %, temperature, humidity and snow 11 %, 12 % and 8 %. It assigns soil moisture a small but non-zero share (3.3–3.6 %) and site and elevation only 4 %, against 23 % at one day (and 2–5 % at longer horizons) in the permutation importance. The two measures differ in what they ask: SHAP measures how much a feature moves the model's output, permutation importance how much predictive accuracy depends on it, so a feature can move the output (here soil moisture, a little) without improving accuracy. These results do not show that soil moisture is unimportant in real catchments; they show that, in this *modelled* system and at daily resolution, the soil-moisture variable adds nothing detectable beyond the rainfall and flow histories.

**Table 120. Mixed-effects model of event amplification.**

{{T:soilmodel}}

**Table 121. Feature-group ablation, LOMO, pooled boosted model (median across locations).**

{{T:ablation}}

**Table 122. Grouped permutation importance (percentage increase in test MSE of the increment when the group is permuted within the test window).**

{{T:perm_importance}}

**Table 134. Grouped TreeSHAP importance of a LightGBM model with the same features (share of the mean absolute SHAP value, chronological test window).**

{{T:shap_groups}}

![Figure 12. Soil moisture and event amplification.](figures/fig9_soil.png)

*Figure 12. Adjusted log amplification against the pre-event soil-moisture anomaly (310 events; colour marks the monsoon season). There is no visible trend.*

**Extremes (RQ5).** Peaks-over-threshold fits (threshold at the 95th percentile, declustered with $r=3$ days, with the 24–30 September 2024 window removed before fitting) give daily-precipitation shape estimates between $-0.15$ (Bahrabise) and $+0.47$ (Bhada Bridge), and **every 95 % bootstrap interval contains zero**, with 30–43 peaks per location (Table 123; precipitation is unaffected by the discharge correction). Profile-likelihood intervals, which are better behaved than the bootstrap for samples this small, are narrower and exclude zero at four locations (Bhada Bridge 0.13 to 1.10, Belsot 0.10 to 1.03, Khokana 0.01 to 1.09 and Chisapani 0.06 to 0.71; Table 133), so the evidence for a heavy precipitation tail depends on the interval method and is suggestive at those four locations only. The September 2024 daily maximum is above the fitted threshold at nine of ten locations (Chameliya/Nayalbadi is the exception, 10.4 mm). The implied return period of that maximum under the tail fitted *without* it varies from 1.1 years (Belsot) and 2.2 years (Chisapani) to 6 years (Bhada Bridge), 9–10 years (Chatara, Khokana), 47 years (Kusum), 69 years (Bahrabise), 309 years (Rasuwagadhi) and 432 years (Devghat). These numbers are dominated by the sign of the poorly determined shape parameter (Khokana's heavy tail, $\hat\xi=0.42$, makes a 165 mm day unremarkable; Devghat's negative shape, $-0.11$, makes a 140 mm day extreme), so we read them only as a consistency check that identifies Devghat, Rasuwagadhi and Bahrabise as the locations where the storm was most unusual relative to the rest of the record. For normalised corrected discharge (Table 124) only 8–28 declustered peaks are available per location, Belsot and Rasuwagadhi have too few to fit, every bootstrap shape interval contains zero (for example −2.33 to 0.73 at Chatara), while profile-likelihood intervals reach the edge of the ±1.5 search range at six of the eight locations (uninformative) and exclude zero only at Chatara (−1.50 to −0.11) and Chameliya/Nayalbadi (−1.50 to −1.09), both negative and so bounded tails, and the implied return periods of the September 2024 peak range from 2.5 years (Bahrabise) and 6.9 years (Devghat, Khokana) to about 100 years (Bhada Bridge, Kusum). At Chatara the peak exceeds the finite upper end of the fitted bounded tail, and at Chameliya/Nayalbadi and Chisapani it is below the threshold, so no return period applies. We do not report design return levels.

**Table 123. GPD fits for daily precipitation (event window excluded), 95 % bootstrap intervals for $\xi$.**

{{T:gpd_P}}

**Table 124. GPD fits for discharge normalised by the location median (corrected series).**

{{T:gpd_Q}}

**Table 133. Profile-likelihood 95 % intervals for the GPD shape, compared with the bootstrap intervals (event window excluded).** "Interval hits grid edge" means the profile interval reaches the limit of the ±1.5 search range and is therefore uninformative on that side.

{{T:gpd_profile}}

**Joint extremes.** The mean empirical extremal-dependence coefficient between locations (probability that one is above its 95th percentile given the other is) is 0.34 for precipitation (maximum 0.55) and 0.38 for discharge (maximum 0.71). On 84 days at least three locations exceed their own 95th-percentile precipitation and on 42 days at least five, and on 6 July 2024 and again on 3 August 2025 all ten do; for discharge, 114 days have three or more locations above their 95th percentile and 43 days five or more, and all ten locations are simultaneously above on 9 August 2024. By this measure the 6 July 2024 rainfall was spatially wider than the 27 September 2024 storm day (all ten locations against nine), although the latter was far more intense where it fell.

**Table 125. Joint-extreme summary (corrected series).**

{{T:chi_summary}}

![Figure 13. Extremal dependence between locations.](figures/fig10_chi.png)

*Figure 13. Empirical $\chi(0.95)$ for precipitation (left) and discharge (right).*

**Upstream–downstream relations (Section 7.7).** After removing the local rainfall response and the autoregressive term from each series, the residual cross-correlations between an upstream and a downstream location are clearly above those of control pairs (Table 126): 0.52 at lag 0 for Rasuwagadhi → Devghat (bootstrap interval 0.42–0.62; 0.30 at lag 1) and 0.52 for Bahrabise → Chatara (0.35–0.65; 0.14 at lag 1), against 0.12 (Rasuwagadhi → Kusum) and 0.11 (Bahrabise → Chisapani; 0.17 at lag 1) for pairs from different basins. The best lag is zero for the two upstream–downstream pairs. **With the corrected discharge there is therefore evidence of same-day coupling between upstream and downstream locations beyond what local rainfall explains**; in the uncorrected series the same pairs gave 0.24 and 0.17 (intervals reaching 0.02), indistinguishable from controls. The coupling is at lag zero, which at daily resolution indicates a travel time shorter than a day (or unobserved shared rainfall), so this finding is a statement about co-movement and not about a measurable propagation delay.

**Table 126. Residual cross-correlation (upstream at time $t$ with downstream at $t+k$) after partialling out local rainfall.**

{{T:synchrony}}

**Recession constants.** Table 6 gives the training-period master-recession constants of the corrected series; the recession-persistence baseline B1 is close to persistence (identical medians at one day), and its significant advantage at eight locations at one day under the chronological split (Table 106) arises from small gains at many locations.

### 8.8 Documented events as positive and negative controls

The rain-driven September 2024 storm is the positive control (Section 8.1). The August 2024 Thame glacial-lake outburst and the July 2025 Bhote Koshi flash flood are negative controls, events the data should not capture. In the corrected series the Thame event has no visible signature at the nearest locations: at Chatara discharge over 12–20 August 2024 is 5,020–6,350 m³/s and its maximum in that window (6,352 m³/s) is below its 99th percentile (6,371 m³/s); at Bahrabise it is 284–347 m³/s, a maximum on 14 August (3 % above its 99th percentile) following 47 mm of rain on 13 August, with no abrupt change on 16 August. At Rasuwagadhi, on 8 July 2025 discharge is 356 m³/s, 1.01 times the prior-week median, with 1.3 mm of precipitation. For the 2026 Rasuwa and Bhote Koshi–Trishuli event, whose date is not given in the file, the highest modelled discharge at Rasuwagadhi in the whole record (552 m³/s) occurs on 18 July 2026, but it follows 58, 36 and 41 mm of rain on 12–14 July and a smooth rise from 449 to 552 m³/s over six days, i.e. a rain-driven signature that cannot be attributed to an outburst. We therefore cannot test that event.

**Residual check.** We computed standardised one-day-ahead forecast residuals out of fold (LOMO, monsoon months only, per-location standardisation) and ranked the residual in a ±1-day window around each event (Table 127). The rain-driven storm produces some of the largest residuals in the record (standardised residual 3.8–4.2 for Kusum, 2.4–3.3 for Khokana and 6.1–6.9 for Devghat, at or above the 98.0th percentile of the monsoon out-of-fold residuals), while the three non-rainfall events give residuals within the ordinary range: 0.08 and −0.34 at Chatara (percentiles 59 and 30), −0.43 and 0.29 at Bahrabise (23 and 80), and 1.00 and 0.81 at Rasuwagadhi (88 and 85).

**Isolation-forest detector.** An isolation forest fitted per location on six standardised features of the rainfall–discharge relationship (log-increment, current and lagged rain, 3-day rain, soil-moisture anomaly, discharge anomaly), with no event labels, places the three storm days at the 99.93rd percentile of each location's record and the three non-rainfall events at ordinary levels (Chatara 64.9, Bahrabise 73.8, Rasuwagadhi 67.7; Table 128).

These checks have two limitations. First, a residual or an anomaly score flags *surprises*, not causes: it is large whenever the river rises more than yesterday's state predicts, which includes ordinary rain not yet observed, so it cannot separate rain-driven from non-rain-driven events. Second, the three negative controls are **absent from the modelled data**, not merely missed by the model, which supports the conclusion that whatever caused those floods is not in the panel. A third detector, an LSTM autoencoder trained without labels on all windows (Table 135), agrees: it places the three storm days at the 99.3rd–99.9th percentile of each location's reconstruction error, and the three non-rainfall events at the 81st (Rasuwagadhi), 85th (Bahrabise) and 87th (Chatara) percentiles, higher than the isolation forest gave but below any usual alarm level.

**Table 127. Out-of-fold standardised residuals around documented events ($h=1$, LOMO).**

{{T:anomaly}}

**Table 128. Isolation-forest percentile of the documented events (maximum over ±1 day).**

{{T:isoforest}}

**Scenario analysis (Section 7.6).** We used a boosted model trained *without* the 2024 monsoon and perturbed its inputs at the origins 26 and 27 September 2024: soil-moisture anomaly ±1 standard deviation and rainfall features scaled by 0.8 and 1.2. The model is almost insensitive to these changes (Table 129): the soil-moisture perturbations change the predicted next-day discharge by exactly 0.0 % at the 27 September origin at all four locations, and the rainfall scalings by between −11.4 % and +4.5 %, with a non-monotone response at Khokana (more rain, lower prediction). The model also cannot reproduce the storm: from the 27 September origin it predicts 470 m³/s at Khokana against an observed 1,632 and 764 m³/s at Kusum against 3,176 (about 71–76 % too low), and 4,320 and 5,108 m³/s at Devghat and Chatara against 6,280 and 7,515 (about 30 % too low). Tree ensembles cannot extrapolate beyond the target range seen in training, and the storm exceeds it. **The scenario analysis therefore says nothing about catchment sensitivity; it documents a limitation of the model class for extreme events.**

**Table 135. LSTM-autoencoder percentile of the documented events (reconstruction error of the last three days of the 30-day window, maximum over ±1 day).**

{{T:autoencoder}}

**Table 129. Counterfactual perturbations, boosted model trained without the 2024 monsoon (origins 26 and 27 September 2024, $h=1$).**

{{T:scenarios}}

**Back-transformation.** The Duan smearing factor computed on the training period is 1.005, 1.013 and 1.015 at 1, 3 and 7 days; applying it changes the median NSE of the default boosted model by 0.000, −0.002 and 0.000, so we keep the plain `expm1` back-transformation.

### 8.9 What changed after correcting the discharge

Table 130 compares key results on the uncorrected V2 series (the companion document `PAPER_PLAN.md`) with those on the corrected series. Some conclusions are robust to the correction and some are not.

**Table 130. Key results before and after correcting the discharge cells.**

| Result | Uncorrected V2 | Corrected | Robust? |
|---|---|---|---|
| Large-river mean discharge (Chisapani, Devghat, Chatara) | 0.8, 2.3, 1.5 m³/s | 1,325, 1,609, 1,829 m³/s | changed |
| Repeated-value days at Bhada Bridge / Chisapani | 64 % / 51 % | 7 % / 2 % | artefact removed |
| Zero-flow days at Khokana | 223 | 0 | artefact removed |
| Persistence, chronological split, median NSE at 1 / 3 / 7 d | 0.953 / 0.867 / 0.752 | 0.984 / 0.949 / 0.893 | changed (stronger baseline) |
| Persistence NSE at Khokana, $h=3$ | −0.224 | −0.071 | changed |
| Learned models vs persistence | within about 0.02 log-NSE of each other; significant at 1–4 locations | within about 0.02; significant at 0–5 | **robust** |
| Ridge model beats persistence under LOMO (sites; significant) | 10/10 at all horizons; 7, 6, 5 | 9, 10, 10 of 10; 8, 8, 3 | **robust**, stronger |
| Boosting untuned vs tuned, chronological NSE at 3 d | 0.850 → 0.887 | 0.904 → 0.965 | **robust** (tuning matters) |
| LSTM tuning | no gain | worse | **robust** (no gain) |
| Graph connectivity helps | no (learned adjacency +0.03 at 7 d in forward chaining) | no (+0.018) | **robust** |
| Soil-moisture effect on event amplification | −0.011 ($p=0.91$) | −0.025 ($p=0.81$) | **robust null** |
| Soil moisture in ablation / permutation importance | ≈ 0 | ≈ 0 | **robust null** |
| Upstream–downstream residual correlation at lag 0 (Rasuwagadhi → Devghat) | 0.24 (0.02–0.40) vs control 0.10 | 0.52 (0.42–0.62) vs control 0.12 | **changed**: appears only after correction |
| Lag centroid, Devghat / Chatara / Chisapani | 2.62 / 2.84 / 2.55 d | 1.81 / 2.01 / 1.75 d | changed |
| Quantile-interval coverage on high-flow days, $h=1$ / 3 | 0.74 / 0.54 | 0.85 / 0.82 | changed |
| Thame GLOF and Bhote Koshi flash flood visible | no | no | **robust** |
| Sept 2024 storm stands out in residuals / isolation forest | yes | yes (99.9th percentile) | **robust** |
| Tree models cannot extrapolate to the storm | yes (about 75 % too low) | yes (30–76 % too low) | **robust** |

The robust findings are the ones most likely to hold for the real rivers; the changed ones are those that depended on the tributary-scale cells.

### 8.10 Discussion

**What the numbers say.** Five results are robust. First, the target is highly persistent, so skill must be stated relative to persistence: with the corrected series, persistence reaches a median NSE of 0.984 at one day. Second, learned models improve on persistence modestly (median log-NSE gains of 0.000–0.007 at one and three days and up to 0.022 at seven on the chronological split), mostly at the flashy locations and in the monsoon-only evaluation, and rarely significantly. Third, **model ranking depends on the split, the horizon, the seed and the tuning**: tuned boosting reaches 0.965 NSE at three days against 0.904 untuned, but tuning the LSTM made it worse, and TFT-lite and the graph networks are better than the LSTM under forward chaining in median but at only half the locations. Once tuned, boosting, the LSTM, TFT-lite and the graph networks all lie within about 0.02 log-NSE of one another. Complexity of the model class buys nothing detectable here. Fourth, **the signal is in the rainfall and flow histories**: the ablation and the permutation importance show that rainfall features and the flow lags carry the skill, while soil moisture and antecedent rainfall indices add nothing detectable; temperature, humidity and snow features matter slightly at longer horizons. Fifth, **explicit connectivity does not help robustly** and **extreme events are out of reach for tree models trained without them**.

**What changed with the correction, and why it matters.** The cell error was not a cosmetic problem. It produced a quantised and intermittent target for several locations, understated the magnitude of the large rivers by three orders of magnitude, hid an upstream–downstream coupling that is clearly present once the channel is sampled, and distorted the interval calibration at high flows. A benchmark built on the uncorrected file would have given numbers that are internally consistent and scientifically misleading. The correction is a heuristic (Section 3.4), but it moves the panel from clearly wrong to plausible, and the comparison of Table 130 shows which conclusions survive.

**Why a headline percentage is hard to interpret.** The dataset page refers to a public notebook that reported a headline percentage on version 1. Without the label definition, split and baseline, such a number cannot be compared with anything. A classifier for "discharge above the site's 90th percentile" can reach a high accuracy by predicting the previous day's label, because high-flow days cluster in the monsoon, and accuracy is dominated by the majority class. The comparison that matters is with the persistence classifier, with precision–recall and event-level measures (Table 111), and under all three label definitions of Section 4.7. In our test window persistence already detects every event at one day (event POD 1.00) with a false-alarm ratio of 0.20.

**What would change the conclusions.** The most important uncertainty is still the match between the corrected cells and the real rivers: no gauge data were available, two of the replacement cells lie at the corner of the scanned block, and the weather is taken at the original coordinate. A second is the effective sample size: the chronological test window contains one full monsoon, and even LOMO has four seasons, of which two include one-off extremes. A longer record would address this; an extension script is provided (`build_extended.py`) but could not be run because the weather API's daily quota was exhausted. Third, the null soil-moisture result may reflect the daily resolution and the coarse model rather than hydrology.

**Positioning of the contribution.** We do not claim an operational flood-warning model. We claim (a) an audit that identifies, with a reproducible scan, a grid-cell error affecting seven of ten locations of a public dataset, and a corrected panel; (b) a leakage-tested benchmark protocol with baselines that are hard to beat, and a demonstration that model complexity buys little over them on this panel; (c) quantitative process findings with honest uncertainty, including null results for soil moisture; and (d) a calibrated statement of blind spots. The first and last of these are the contributions most likely to help other users of the dataset.

---

