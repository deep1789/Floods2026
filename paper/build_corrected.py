"""Build the corrected-discharge panel for 2023-2026 from cached GloFAS cell scans (no network). Weather/soil are unchanged (V2)."""
import os, json, hashlib, urllib.parse, numpy as np, pandas as pd
from floodlab import data
F_API = 'https://flood-api.open-meteo.com/v1/flood'; CACHE = 'results/extended/cache'
df = data.load(); summ = pd.read_csv('results/extended/cell_scan_summary.csv').set_index('location'); out = []; log = []
for s, g in df.groupby('location', sort=False):
    la, lo = float(g.latitude.iloc[0]), float(g.longitude.iloc[0]); use = summ.loc[s, 'ratio_best_to_center'] > 10; di, dj = eval(summ.loc[s, 'best_cell']) if use else (0, 0)
    p = dict(latitude=round(la + di * 0.05, 4), longitude=round(lo + dj * 0.05, 4), daily='river_discharge', start_date='2023-01-01', end_date='2026-08-31')
    fn = f"{CACHE}/{hashlib.md5((F_API + '?' + urllib.parse.urlencode(p)).encode()).hexdigest()}.json"
    if not os.path.exists(fn): print('MISSING', s, p, use, (di, dj)); continue
    j = json.load(open(fn))
    q = pd.Series(j['daily']['river_discharge'], index=pd.to_datetime(j['daily']['time'])).reindex(g.date).values
    h = g.copy(); h['river_discharge_m3s_original_cell'] = h.river_discharge_m3s; h['river_discharge_m3s'] = q; h['corrected_cell'] = bool(use); out.append(h)
    log.append(dict(location=s, corrected=bool(use), cell_offset=(di, dj), snapped_lat=j['latitude'], snapped_lon=j['longitude'], mean_original=g.river_discharge_m3s.mean(), mean_corrected=np.nanmean(q), nan=int(np.isnan(q).sum())))
P = pd.concat(out); P.to_csv('corrected/panel_corrected_2023_2026.csv', index=False); L = pd.DataFrame(log); L.to_csv('results/extended/corrected_cells.csv', index=False); print(L.round(2).to_string()); print(P.shape)
