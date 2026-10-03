"""Plausibility check without gauge data: catchment area implied by mean discharge, local precipitation and a runoff coefficient c
(A = Qbar * seconds_per_year / (P_annual * c)); c = 0.5 central, 0.3-0.7 range. Run from corrected/ (needs panel_corrected_2023_2026.csv)."""
import pandas as pd
df = pd.read_csv('panel_corrected_2023_2026.csv', parse_dates=['date']); rows = []
for s, g in df.groupby('location'):
    P = g.precipitation_mm.sum() / (len(g) / 365.25)
    for lab, col in (('V2 file', 'river_discharge_m3s_original_cell'), ('corrected', 'river_discharge_m3s')):
        Qm = g[col].mean(); vol = Qm * 86400 * 365.25; a = [vol / ((P / 1000) * c) / 1e6 for c in (0.7, 0.5, 0.3)]
        rows.append(dict(location=s, series=lab, mean_Q=Qm, P_mm_yr=P, area_c05=a[1], area_lo=a[0], area_hi=a[2]))
pd.DataFrame(rows).to_csv('results/implied_area.csv', index=False)
