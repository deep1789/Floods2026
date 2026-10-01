### 4.8 A worked example: Khokana, 24 September – 2 October 2024

To make the formulation concrete, Table 4a shows the quantities defined above for the most responsive location during the late-September 2024 storm. All values come from the CSV; the antecedent index uses a recession constant $k=0.8$ chosen for illustration (the study estimates $k$ per site).

**Table 4a. Khokana during the 2024 storm.** $y=\log(1+Q)$, $\Delta y$ is the one-day log increment, "persistence error" is $Q_{t+1}-Q_t$ in m³/s, and API is Eq. 4.3 with $k=0.8$.

| Date | P (mm) | θ (m³/m³) | Q (m³/s) | y | Δy | Persistence error (next day) | API(0.8) |
|---|---|---|---|---|---|---|---|
| 24 Sep | 9.4 | 0.375 | 0.33 | 0.285 | — | +0.31 | 25.2 |
| 25 Sep | 5.3 | 0.377 | 0.64 | 0.495 | +0.210 | +2.07 | 25.4 |
| 26 Sep | 86.5 | 0.389 | 2.71 | 1.311 | +0.816 | +11.67 | 106.9 |
| 27 Sep | 164.9 | 0.419 | 14.38 | 2.733 | +1.422 | +43.23 | 250.4 |
| 28 Sep | 76.4 | 0.429 | 57.61 | 4.071 | +1.338 | −18.80 | 276.7 |
| 29 Sep | 4.0 | 0.424 | 38.81 | 3.684 | −0.387 | −30.61 | 225.4 |
| 30 Sep | 0.5 | 0.418 | 8.20 | 2.219 | −1.465 | −4.85 | 180.8 |
| 1 Oct | 1.9 | 0.413 | 3.35 | 1.470 | −0.749 | −1.14 | 146.5 |
| 2 Oct | 18.3 | 0.411 | 2.21 | 1.166 | −0.304 | — | 135.5 |

Four features of this table drive the modelling choices.

1. **The response is fast and asymmetric.** Discharge rises 22-fold in two days (2.71 to 57.61 m³/s between 26 and 28 September) and falls to 8.20 m³/s within two days of the peak. The recession ratio on 29→30 September is $8.20/38.81=0.21$, far steeper than the linear-reservoir constants of the slower sites. For such a site a one-day persistence forecast has errors of +43 m³/s on the rising limb and −31 m³/s on the falling limb. The sum of squared one-day persistence errors over 26–30 September is about 3,300 (m³/s)², the same order as the total variance of ordinary dry-season flow multiplied over many months, which is why NSE at this site is dominated by a handful of days.
2. **The increment target is informative.** The log-increment $\Delta y$ is positive on 25–28 September when rain is falling and negative afterwards, and it is large (|Δy| > 1.3) on four of the nine days. A model that predicts $\Delta y_{t+1}$ from $P_t$, $P_{t-1}$ and the API has a direct, nearly monotone signal on the rising limb. This is the mechanism behind the boosted model's one-day gain at Khokana (NSE 0.141 → 0.632 on the test period, Section 8.4).
3. **Soil moisture moves slowly and late.** Soil moisture rises from 0.375 to 0.429 (+0.054) over the event while discharge changes by a factor of about 170 (0.33 to 57.61 m³/s). The 0–100 cm layer integrates rainfall and is a *state* rather than a trigger; its value on 26 September (0.389) is already elevated relative to the dry-season level, and the ablation in Section 6.5 asks whether it adds skill beyond the rainfall history. This is an empirical question, not an assumption.
4. **The API saturates and decays.** With $k=0.8$ the index is 250–277 on 27–28 September and decays by about 20 % per day. If the best $k$ differs markedly by site, that is itself a measure of storage.

### 4.9 Master recession constants (preliminary)

The recession model of Section 4.5 requires pairs of consecutive days with no rainfall and a falling hydrograph. Applying the filter $P_t=P_{t+1}=0,\ Q_{t+1}\le Q_t,\ Q_t>0.5$ m³/s on the training period gives wildly different sample sizes: 21 pairs at Rasuwagadhi (median ratio $c=0.957$, i.e. a storage constant $\kappa=-1/\ln c\approx 23$ days), 23 at Chisapani ($c=0.951$, $\kappa\approx 20$ days), and only 3 at Khokana ($c=0.41$, unreliable). At Devghat the filter returns 489 pairs but the median ratio is exactly 1.0, because the series is quantised or plateaued at the 2-decimal resolution (audit A5), which makes the linear-reservoir constant degenerate. We draw two conclusions. First, a per-site recession constant is only identifiable at some sites; the others need a pooled or regularised estimate. Second, this is a concrete example of how the audit findings propagate: a quantised target makes a standard hydrological diagnostic uninformative, and the study must say so instead of reporting a spurious $\kappa$.
