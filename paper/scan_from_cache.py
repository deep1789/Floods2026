"""Analyse the GloFAS cell scan from cached API responses only (no network). Complete locations only."""
import os, json, hashlib, urllib.parse, numpy as np, pandas as pd
from floodlab import data
F_API = 'https://flood-api.open-meteo.com/v1/flood'; CACHE = 'results/extended/cache'
df = data.load(); sites = df.groupby('location')[['latitude', 'longitude']].first(); rows = []
for s, (la, lo) in sites.iterrows():
    ref = df[df.location == s].set_index('date').river_discharge_m3s; cells = {}
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = dict(latitude=round(la + i * 0.05, 4), longitude=round(lo + j * 0.05, 4), daily='river_discharge', start_date='2023-01-01', end_date='2026-08-31')
            fn = f"{CACHE}/{hashlib.md5((F_API + '?' + urllib.parse.urlencode(p)).encode()).hexdigest()}.json"
            if os.path.exists(fn): cells[(i, j)] = json.load(open(fn))
    if len(cells) < 25: print(s, 'incomplete', len(cells), '/25'); continue
    for (i, j), jd in cells.items():
        q = pd.Series(jd['daily']['river_discharge'], index=pd.to_datetime(jd['daily']['time'])).reindex(ref.index)
        rows.append(dict(location=s, di=i, dj=j, snapped_lat=jd['latitude'], snapped_lon=jd['longitude'], mean_Q=q.mean(), max_Q=q.max(), corr_with_dataset=np.corrcoef(q.fillna(0), ref)[0, 1], max_abs_diff=float((q - ref).abs().max())))
R = pd.DataFrame(rows); R.to_csv('results/extended/cell_scan_cached.csv', index=False)
out = []
for s, g in R.groupby('location'):
    c = g[(g.di == 0) & (g.dj == 0)].iloc[0]; b = g.loc[g.mean_Q.idxmax()]; match = g[g.max_abs_diff < 0.02]
    out.append(dict(location=s, dataset_mean=df[df.location == s].river_discharge_m3s.mean(), center_cell_reproduces_dataset=bool(c.max_abs_diff < 0.02), center_mean_Q=c.mean_Q, best_cell=f'({int(b.di)},{int(b.dj)})', best_mean_Q=b.mean_Q, ratio_best_to_center=b.mean_Q / c.mean_Q, cells_with_mean_gt_10x_center=int((g.mean_Q > 10 * c.mean_Q).sum())))
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv('results/extended/cell_scan_summary.csv', index=False)
