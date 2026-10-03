### 4.8 A worked example: Khokana, 24 September – 2 October 2024

To make the formulation concrete, Table 5 shows the quantities defined above for the most responsive location during the late-September 2024 storm, from the corrected series. The antecedent index uses a recession constant $k=0.8$ chosen for illustration (the study estimates $k$ per site).

**Table 5. Khokana during the 2024 storm (corrected discharge).** $y=\log(1+Q)$, $\Delta y$ is the one-day log increment, "persistence error" is $Q_{t+1}-Q_t$ in m³/s, and API is Eq. 4.3 with $k=0.8$.

| Date | P (mm) | θ (m³/m³) | Q (m³/s) | y | Δy | Persistence error (next day) | API(0.8) |
|---|---|---|---|---|---|---|---|
| 24 Sep | 9.4 | 0.375 | 11.1 | 2.492 | — | +37.9 | 25.2 |
| 25 Sep | 5.3 | 0.377 | 49.0 | 3.912 | +1.420 | +37.3 | 25.4 |
| 26 Sep | 86.5 | 0.389 | 86.3 | 4.469 | +0.557 | +354.3 | 106.9 |
| 27 Sep | 164.9 | 0.419 | 440.6 | 6.090 | +1.621 | +1,191.5 | 250.4 |
| 28 Sep | 76.4 | 0.429 | 1,632.1 | 7.398 | +1.308 | −487.5 | 276.7 |
| 29 Sep | 4.0 | 0.424 | 1,144.5 | 7.044 | −0.355 | −842.1 | 225.4 |
| 30 Sep | 0.5 | 0.418 | 302.4 | 5.715 | −1.329 | −166.6 | 180.8 |
| 1 Oct | 1.9 | 0.413 | 135.8 | 4.918 | −0.797 | −57.3 | 146.5 |
| 2 Oct | 18.3 | 0.411 | 78.4 | 4.375 | −0.543 | — | 135.5 |

Four features of this table drive the modelling choices.

1. **The response is fast and asymmetric.** Discharge rises 19-fold in two days (86 to 1,632 m³/s between 26 and 28 September, 147-fold from 24 September) and falls to 302 m³/s two days after the peak. The recession ratio on 29→30 September is $302/1{,}145=0.26$, far steeper than the linear-reservoir constants of the large rivers. A one-day persistence forecast errs by +1,191 m³/s on the rising limb and −842 m³/s on the falling limb, and the sum of squared one-day persistence errors over 26–30 September (about $2.5\times10^{6}$ (m³/s)²) dwarfs the variance of an ordinary day (median flow 6.4 m³/s). This is why NSE at this location is dominated by a handful of days.
2. **The increment target is informative.** The log-increment $\Delta y$ is positive on 25–28 September when rain is falling and negative afterwards, and it is large ($|\Delta y|>1.3$) on four of the nine days. A model that predicts $\Delta y_{t+1}$ from $P_t$, $P_{t-1}$ and the API has a direct, nearly monotone signal on the rising limb, which is the mechanism behind the gains of the learned models at Khokana (Section 8.4).
3. **Soil moisture moves slowly and late.** Soil moisture rises from 0.375 to 0.429 (+0.054) over the event while discharge changes by a factor of about 150. The 0–100 cm layer integrates rainfall and is a *state* rather than a trigger, and the ablation in Section 8.8 asks whether it adds skill beyond the rainfall history.
4. **The API saturates and decays.** With $k=0.8$ the index is 250–277 on 27–28 September and decays by about 20 % per day.

### 4.9 Master recession constants

The recession model of Section 4.5 requires pairs of consecutive days with no rainfall and a falling hydrograph. Applying the filter $P_t=P_{t+1}=0,\ Q_{t+1}\le Q_t,\ Q_t>0.5$ m³/s on the training period gives 103 to 479 qualifying pairs at every location of the corrected panel (Table 6), so a per-site constant is identifiable everywhere. The median day-to-day ratios are 0.968–0.997 at nine locations, i.e. storage constants of 31 to 320 days, which describe slowly receding baseflow, and 0.772 at Khokana, i.e. about 4 days, consistent with its flashy response. (In the V2 series, the same filter gave only 3 to 37 pairs at five locations and a degenerate ratio of exactly 1.0 at Devghat, because the plateaued values made the diagnostic uninformative; the correction removes this problem.)

**Table 6. Recession constants estimated on the training period (corrected discharge).** $\kappa=-1/\ln c$ is the implied storage constant in days.

{{T:recession}}

