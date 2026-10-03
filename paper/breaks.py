"""Pettitt change-point test (Sec. 3.6) on seasonally adjusted weather series. Rank-based: U_t = 2*sum_{i<=t} r_i - t(n+1), K = max|U_t|, p ~ 2 exp(-6 K^2 / (n^3 + n^2))."""
import numpy as np, pandas as pd
from scipy import stats
from floodlab import data, features as F
df = data.load(); order = data.site_order(df)
VARS = {'precipitation_mm': 'precipitation', 'temperature_mean_c': 'temperature', 'soil_moisture_0_100cm_m3m3': 'soil moisture', 'relative_humidity_mean_pct': 'relative humidity', 'dew_point_mean_c': 'dew point', 'wind_speed_max_kmh': 'max wind speed'}
rows = []
for s in order:
    g = df[df.location == s].sort_values('date'); X = F._design(g.date.dt.dayofyear.values)
    for v, lab in VARS.items():
        y = g[v].values.astype(float); res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; n = len(res); r = stats.rankdata(res)
        U = 2 * np.cumsum(r) - np.arange(1, n + 1) * (n + 1); k = np.argmax(np.abs(U[:-1])); K = abs(U[k]); p = min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
        rows.append(dict(location=s, variable=lab, change_date=str(g.date.iloc[k + 1].date()), K=K, p=p, mean_before=res[:k + 1].mean(), mean_after=res[k + 1:].mean(), sd=res.std()))
R = pd.DataFrame(rows); R['p_bonf'] = np.minimum(1, R.p * len(R)); R['shift_in_sd'] = (R.mean_after - R.mean_before) / R.sd; R.to_csv('results/pettitt.csv', index=False)
sig = R[R.p_bonf < 0.05]; print(len(R), 'tests;', len(sig), 'significant after Bonferroni'); print(sig.round(3).to_string() if len(sig) else '')
sm = R.groupby('variable').agg(tests=('p', 'size'), min_p=('p', 'min'), n_sig_raw=('p', lambda x: int((x < 0.05).sum())), n_sig_bonf=('p_bonf', lambda x: int((x < 0.05).sum())), max_abs_shift_sd=('shift_in_sd', lambda x: float(np.abs(x).max()))).reset_index()
h = '| ' + ' | '.join(['Variable', 'Tests', 'Smallest p', 'p<0.05 (raw)', 'p<0.05 (Bonferroni, 60 tests)', 'Largest |mean shift| (sd)']) + ' |\n|' + '|'.join(['---'] * 6) + '|\n'
open('tables2/pettitt.md', 'w').write(h + '\n'.join(f'| {r.variable} | {r.tests} | {r.min_p:.4f} | {r.n_sig_raw} | {r.n_sig_bonf} | {r.max_abs_shift_sd:.2f} |' for r in sm.itertuples()) + '\n'); print(sm.round(4).to_string())
