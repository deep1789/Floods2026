## 3. Data Description, Audit and Correction

### 3.1 Provenance and construction

The panel is constructed by joining two API products by date and location, then attaching station metadata.

- **Weather and soil moisture:** Open-Meteo Historical Weather API, "Best Match" model selection (a seamless blend of available model datasets, not a single fixed reanalysis). Variables are daily aggregates of hourly model output.
- **River discharge:** Open-Meteo Flood API, which serves GloFAS data. Discharge is a modelled quantity.
- **Metadata:** river, basin, elevation, latitude/longitude and DHM station identifiers from information published by Nepal's DHM.

Attribution follows the CC BY 4.0 requirements of the Open-Meteo API. The V2 release corrects the 2023 weather-variable mapping and station metadata found after validation. **This paper starts from V2 and then corrects its discharge series** (Section 3.4, test A4). Analyses on the uncorrected V2 discharge are kept in a companion document (`PAPER_PLAN.md`), and Section 8.9 compares the two sets of results.

### 3.2 Locations

Table 1 lists the ten locations ordered by the elevation field, with discharge summaries for the *corrected* series. Two rivers carry the name *Bhote Koshi*: the Rasuwagadhi site is on the Bhote Koshi that becomes the Trishuli and then the Narayani, whereas Bahrabise is on the Bhote Koshi that joins the Sun Koshi in the Koshi basin. Analysts who group by `river` instead of `basin` will merge two distinct drainage systems, so we group by `location` and by `basin` and never by `river` alone.

**Table 1. Monitoring locations and corrected discharge summary (m³/s).**

{{T:sites}}

![Figure 1. Locations coloured by the elevation field.](figures/fig1_locations.png)

*Figure 1. The ten locations. The panel spans roughly 80.6–87.2 °E and 26.9–29.7 °N, covering the Mahakali/Karnali system in the west through the Koshi system in the east.*

### 3.3 Variable dictionary

**Table 2. Variables, with their roles in this study.**

| Variable | Unit | Type | Role in study |
|---|---|---|---|
| `date` | day | index | time index; daily, no gaps |
| `location`, `river`, `basin` | — | categorical | panel identifier; basin used for grouping |
| `dhm_station` | id | numeric id | metadata only (not a numeric predictor) |
| `latitude`, `longitude`, `elevation_m` | °, °, m | static | static covariates (with caveat, test A7) |
| `precipitation_mm` | mm/day | dynamic | main forcing; heavy-tailed, zero-inflated |
| `rain_mm` | mm/day | dynamic | liquid part of precipitation |
| `precipitation_hours` | h/day | dynamic | duration; intensity = P / hours |
| `soil_moisture_0_100cm_m3m3` | m³/m³ | dynamic | antecedent wetness |
| `temperature_mean_c` | °C | dynamic | melt/phase proxy |
| `dew_point_mean_c`, `relative_humidity_mean_pct` | °C, % | dynamic | moisture state |
| `wind_speed_max_kmh`, `wind_gusts_max_kmh`, `wind_direction_dominant_deg` | km/h, km/h, ° | dynamic | storm-regime descriptors; direction is circular |
| `river_discharge_m3s` | m³/s | target | modelled discharge (corrected cell, Section 3.4) |

The difference `precipitation_mm − rain_mm` is a proxy for solid precipitation (snow). It is non-zero on 223 days at Rasuwagadhi and on only 4 days at Chameliya/Nayalbadi, and exactly zero everywhere else, which is consistent with Rasuwagadhi's high elevation and cool climate (mean temperature 17.5 °C).

### 3.4 Audit findings

The audit consists of eight tests. They are cheap to run and form a reproducible checklist (A1–A3 are implemented as code in `floodlab/data.py`; the others are analyses with the scripts named in Section 10).

**A1 — Completeness and key integrity.** 10 locations × 1,339 days with no duplicate (location, date) pairs and no missing values. *Pass.*

**A2 — Zero-inflation of precipitation.** Between 30 % and 59 % of days have exactly zero precipitation, rain and precipitation hours, and the three zero patterns are internally consistent (no row has precipitation zero and positive precipitation hours, or the reverse). We follow the curator's advice and **do not impute or remove zeros**; we use the hurdle structure of Section 4.4 where needed. *Pass.*

**A3 — Rain versus precipitation.** `rain_mm` never exceeds `precipitation_mm`, and equals it except where snow is plausible (Section 3.3). *Pass.*

**A4 — Discharge magnitude and the grid cell it came from (critical; failed, then corrected).** In the V2 file, mean discharge at the three large-river locations Chisapani (Karnali), Devghat (Narayani) and Chatara (Saptakoshi) is 0.8, 2.3 and 1.5 m³/s, with maxima of 17, 21 and 23 m³/s. We recall published long-term mean flows for these rivers at or near the named stations of order 10³ m³/s; we have not checked this against DHM gauge series, so the statement is an expectation and not a validated fact. We tested the obvious explanation, that the requested coordinate falls in a GloFAS grid cell that is not on the main channel. The Flood API reports the model cell it used (for example, a request at 28.65 °N is answered for 28.675 °N), and the cell size is 0.05° (about 5 km). For each location we requested the discharge of the 5 × 5 block of cells centred on the dataset coordinate (offsets of ±0.05° and ±0.10° in latitude and longitude), 250 requests over 1 January 2023–31 August 2026, and compared each with the dataset series (Table 101).

**Table 101. GloFAS cell scan. The dataset's own cell reproduces the file exactly at all ten locations (maximum absolute difference below 0.02 m³/s); at seven locations a neighbouring cell carries 17–1,565 times more flow.** "Offset" is (latitude, longitude) in steps of 0.05° from the dataset coordinate; "cells >10×" counts the 24 neighbouring cells with a mean above ten times the dataset cell's.

{{T:cellscan}}

The outcome is unambiguous for the large rivers: the dataset sampled a cell of tiny upstream area beside the channel, and one or two cells away the same request returns means of 1,325 m³/s (Chisapani), 1,609 m³/s (Devghat) and 1,829 m³/s (Chatara), the order of magnitude we expect for those rivers. Bhada Bridge (65 m³/s, from 0.7), Bahrabise (84, from 5.1), Khokana (49, from 1.7) and Rasuwagadhi (133, from 1.4) are affected in the same way. Belsot, Chameliya/Nayalbadi and Kusum are not: their best neighbour is within 27 % of the dataset cell.

*Correction.* We built a corrected series (`paper/corrected/panel_corrected_2023_2026.csv`) that, at each location where a neighbouring cell has more than ten times the mean flow of the dataset cell, uses the neighbouring cell with the highest mean flow (seven locations), and otherwise keeps the dataset cell (three locations). Weather and soil moisture are unchanged. **This is a heuristic, and it has limits we cannot remove without gauge data:** (i) for Khokana and Rasuwagadhi the best cell lies at the corner of the scanned block, so a larger cell may exist beyond it; (ii) "highest mean in the block" could select a different river near a confluence; (iii) the weather is still taken at the original coordinate, up to about 0.11° (12 km) from the discharge cell; and (iv) no upstream area or gauge series was available to confirm any replacement. A visual check supports caution at two locations (Figure 2): the corrected Khokana series is still erratic, with dry-season flows of 0.05–1 m³/s and spikes of 10²–10³ m³/s, and the unchanged Kusum series drops to about 0.1 m³/s at monsoon onset, so neither cell is guaranteed to be on the main channel; the three large-river series (Chatara, Devghat, Chisapani) and Bahrabise, Rasuwagadhi and Bhada Bridge look like smooth monsoon hydrographs. Where we quote results for the corrected panel they describe the *corrected modelled series*, not the gauged river.

**A5 — Quantisation and repeated values (consequence of A4).** In the V2 series, 64 % of Bhada Bridge days and 51 % of Chisapani days had discharge identical to the previous day, because flows were so small that they were rounded at the second decimal. In the corrected series the repeated-value fraction is 2–13 % at every location (Table 4).

**A6 — Intermittent zero flow (consequence of A4).** The V2 series had 223 days of exactly zero discharge at Khokana (16.7 % of the record). The corrected series has none (its 5th percentile is 0.1 m³/s), so the apparent intermittency was an artefact of the small cell and no zero-flow model is needed.

**A7 — Elevation field plausibility.** The `elevation_m` field gives Chisapani as 2,215 m, but its mean temperature is 23.1 °C, similar to the 150–230 m sites (23.9–24.6 °C) and far above the 17.5 °C of the 1,749 m Rasuwagadhi site. A 2,215 m Nepal site with a 23 °C annual mean is not plausible. We therefore **do not treat `elevation_m` as a physical covariate** and run models with and without it. *Fail, unresolved.*

**A8 — Dates beyond the present.** The panel ends on 31 August 2026 and contains the 2026 monsoon through that date. The documentation mentions a 2026 flood in the Rasuwa and Bhote Koshi–Trishuli corridor without giving its date, so it cannot be tested directly (Section 8.8).

**Table 3. Seasonality and tail diagnostics per location (June–September = monsoon), corrected discharge.**

{{T:sitestats}}

**Table 4. Data-quality diagnostics per location, corrected discharge.** "Zero P days" are genuine dry days, retained by design. "Repeated Q" is the fraction of days where discharge is identical to the previous day.

{{T:audit}}

### 3.5 Circularity and the status of the target

Let $F$ denote Open-Meteo/GloFAS's hydrological model, forced with meteorological inputs $X^{\mathrm{GloFAS}}_{t}$. The target is $Q_t = F(X^{\mathrm{GloFAS}}_{1:t}) + \epsilon_t$, and the predictors are Open-Meteo weather $X^{\mathrm{OM}}_t$, which is *related* to, but not guaranteed identical to, $X^{\mathrm{GloFAS}}_t$. A learned model $\hat f$ that predicts $Q_{t+h}$ from $X^{\mathrm{OM}}_{1:t}$ is therefore approximating $F$ composed with a forcing-mismatch term. Three consequences follow:

1. High skill is expected whenever the forcings are close, and tells us little about the real river.
2. Skill should be *compared to a benchmark that exploits the same structure* (persistence plus recession), not to zero.
3. The right scientific question is not "can we forecast the river?" but "how is the modelled river's response organised, and where does the data-driven approximation break?"

We write the paper in this framing throughout. The A4 finding is a further warning: the identity of the modelled cell is itself a data-processing choice, and a benchmark built on the wrong cell would have produced confident but meaningless numbers.

### 3.6 Structural breaks and non-stationarity

Because "Best Match" is a blend, the underlying weather model may change over time. We will test each weather series for mean shifts using the CUSUM statistic and the Pettitt test, after removing the seasonal cycle by regressing on harmonics. For a series $z_t$, $t=1,\dots,n$, the Pettitt statistic is

$$U_{t} = \sum_{i=1}^{t}\sum_{j=t+1}^{n}\operatorname{sgn}(z_i - z_j), \qquad K = \max_{1\le t<n}|U_t|,$$

with approximate $p$-value $p \approx 2\exp\!\big(-6K^2 / (n^3 + n^2)\big)$. A significant break in precipitation or soil moisture near a model-version date would be a red flag for temporal validation. We ran the Pettitt test on six seasonally adjusted weather series at each of the ten locations (60 tests; Table 132; the CUSUM test was not run).

**Table 132. Pettitt change-point tests on seasonally adjusted weather series (10 locations per variable).** "Largest shift" is the largest change in the mean of the residual, in standard deviations, between the segments before and after the estimated change point.

{{T:pettitt}}

Almost every series is flagged: 46 of the 60 tests remain significant after a Bonferroni correction, including soil moisture at all ten locations (shifts up to 1.4 standard deviations), dew point and relative humidity at 9 and 8, and temperature at 8. **These p-values cannot be taken at face value.** The test assumes independent observations, whereas daily residuals of soil moisture, humidity and temperature are strongly autocorrelated, and with only 3.7 years a wet year followed by a dry one produces an apparent break whatever the cause. What is informative is that the estimated dates *cluster across locations*: the dew-point break falls within 6–8 May 2024 at six of ten locations, the relative-humidity break within 8–9 April 2025 at five, and the temperature break within 27–28 April 2025 at four (and on 29 March 2024 at two more), and the soil-moisture breaks fall between June 2024 and September 2025. Common dates at distant locations are what one would expect from a change in the weather-model blend, but also from a common weather regime, and we cannot distinguish the two without the provider's model-version history, which we did not have. We therefore treat the weather series as possibly non-stationary in distribution. The chronological test period (September 2025–August 2026) lies after most of the estimated breaks, so a model trained on earlier data may meet different input distributions at test time; the ablation, which shows that humidity, soil moisture and temperature contribute little (Section 8.7), limits the consequence for the forecasts reported here, but not for any analysis that uses those variables directly.


### 3.7 Exploratory results

![Figure 2. Modelled daily discharge, log scale, one panel per location.](figures/fig2_discharge.png)

*Figure 2. Corrected modelled daily discharge at each location (log axis). Monsoon peaks dominate, and the plateaus and zero floor of the V2 series (Khokana, Bhada Bridge, Chisapani, Chatara) are gone.*

![Figure 3. Seasonal cycle of discharge and precipitation.](figures/fig4_seasonality.png)

*Figure 3. Left: mean discharge by month normalised by the annual mean. Right: monthly share of total precipitation. All locations peak in June–September, but the discharge peak lags the rainfall peak at several sites.*

![Figure 4. Distribution of discharge by location (log10).](figures/fig6_qbox.png)

*Figure 4. Distribution of log10 corrected discharge at each location, ordered as in Table 1. The locations still differ by roughly three orders of magnitude (the three large rivers carry 10³ m³/s, the others 10¹–10²), which is why we normalise per location before pooling.*

Discharge is extremely persistent. Lag-1 autocorrelation of $\log(1+Q)$ ranges from 0.968 (Khokana) to 0.999 (Chameliya/Nayalbadi, Chisapani and Rasuwagadhi), and lag-7 from 0.821 (Khokana) to 0.983 (Rasuwagadhi). This is why persistence is a very hard baseline at every location and why Khokana, the flashiest, is the most informative test.

---

