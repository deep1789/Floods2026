"""RESUMING: responses already fetched are saved in extended/api_cache.tar.gz. Before running, restore them with
    mkdir -p results/extended && tar -xzf extended/api_cache.tar.gz -C results/extended
The script then skips every request already cached. Run it when the Open-Meteo daily quota has reset (the API answers "Daily API request limit exceeded" otherwise).

Build the corrected + extended panel (needs network: archive-api.open-meteo.com and flood-api.open-meteo.com).
 - Discharge: for locations where a neighbouring GloFAS cell has >10x the mean flow of the dataset's cell (results/extended/cell_scan_summary.csv),
   use that cell ('corrected'); otherwise keep the dataset's cell. The original-cell series is kept as river_discharge_m3s_original_cell.
 - Weather and soil moisture: Open-Meteo archive (default 'best match'), 2010-01-01..2022-12-31; the dataset's V2 values are kept for 2023-01-01 onwards.
Writes extended/panel_2010_2026.csv (same columns as the original file plus the original-cell discharge)."""
import os, sys, json, time, hashlib, urllib.request, urllib.parse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from floodlab import data
START, END_EXT = int(os.environ.get('EXT_START', 2010)), 2022
CACHE = 'results/extended/cache'; os.makedirs(CACHE, exist_ok=True); os.makedirs('extended', exist_ok=True)
W_API = 'https://archive-api.open-meteo.com/v1/archive'; F_API = 'https://flood-api.open-meteo.com/v1/flood'
DAILY = 'precipitation_sum,rain_sum,temperature_2m_mean,dew_point_2m_mean,precipitation_hours,relative_humidity_2m_mean,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant'
MAP = {'precipitation_sum': 'precipitation_mm', 'rain_sum': 'rain_mm', 'temperature_2m_mean': 'temperature_mean_c', 'dew_point_2m_mean': 'dew_point_mean_c', 'precipitation_hours': 'precipitation_hours',
       'relative_humidity_2m_mean': 'relative_humidity_mean_pct', 'wind_speed_10m_max': 'wind_speed_max_kmh', 'wind_gusts_10m_max': 'wind_gusts_max_kmh', 'wind_direction_10m_dominant': 'wind_direction_dominant_deg'}
def get(url, params, tries=8):
    q = url + '?' + urllib.parse.urlencode(params); fn = f"{CACHE}/{hashlib.md5(q.encode()).hexdigest()}.json"
    if os.path.exists(fn): return json.load(open(fn))
    for i in range(tries):
        try:
            time.sleep(4)
            with urllib.request.urlopen(q, timeout=120) as r: j = json.load(r)
            if j.get('error'): raise RuntimeError(j.get('reason'))
            json.dump(j, open(fn, 'w')); return j
        except Exception as e:
            print('   retry', i, str(e)[:90], flush=True)
            if i == tries - 1: raise
            time.sleep(min(120, 15 * 2 ** i))
def flood(la, lo, a, b):
    j = get(F_API, dict(latitude=la, longitude=lo, daily='river_discharge', start_date=a, end_date=b)); return pd.Series(j['daily']['river_discharge'], index=pd.to_datetime(j['daily']['time']))
def weather(la, lo, y0, y1):
    """One-year requests: large requests exceed the API's per-minute weighted limit and are rejected with 429."""
    parts, sm = [], []
    for y in range(y0, y1 + 1):
        j = get(W_API, dict(latitude=la, longitude=lo, start_date=f'{y}-01-01', end_date=f'{y}-12-31', daily=DAILY)); parts.append(pd.DataFrame(j['daily']))
        h = get(W_API, dict(latitude=la, longitude=lo, start_date=f'{y}-01-01', end_date=f'{y}-12-31', hourly='soil_moisture_0_to_100cm'))
        sm.append(pd.Series(h['hourly']['soil_moisture_0_to_100cm'], index=pd.to_datetime(h['hourly']['time'])).resample('D').mean())
    w = pd.concat(parts); w['time'] = pd.to_datetime(w.time); w = w.set_index('time').rename(columns=MAP); w['soil_moisture_0_100cm_m3m3'] = pd.concat(sm); return w
if __name__ == '__main__':
    df = data.load(); summ = pd.read_csv('results/extended/cell_scan_summary.csv').set_index('location'); sites = df.groupby('location').first(); out = []; check = []
    for s in sites.index:
        la, lo = sites.loc[s, 'latitude'], sites.loc[s, 'longitude']; o = df[df.location == s].set_index('date'); print(s, flush=True)
        use_best = summ.loc[s, 'ratio_best_to_center'] > 10; di, dj = eval(summ.loc[s, 'best_cell']) if use_best else (0, 0)
        # discharge: corrected cell and original cell for the full period
        best_lat, best_lon = round(la + di * 0.05, 4), round(lo + dj * 0.05, 4)
        q_corr = flood(best_lat, best_lon, f'{START}-01-01', '2026-08-31'); q_orig = flood(la, lo, f'{START}-01-01', '2026-08-31')
        # weather: archive for the extension; keep V2 values from 2023 on
        w = weather(la, lo, START, END_EXT)
        # consistency check on the overlap (2023-2024) between the archive and the V2 file
        if s in ('Khokana', 'Devghat'):
            wo = weather(la, lo, 2023, 2024); 
            for c in list(MAP.values()) + ['soil_moisture_0_100cm_m3m3']: check.append(dict(location=s, variable=c, max_abs_diff=float((wo[c] - o[c].reindex(wo.index)).abs().max()), mean_abs_diff=float((wo[c] - o[c].reindex(wo.index)).abs().mean())))
        full_idx = pd.date_range(f'{START}-01-01', '2026-08-31'); p = pd.DataFrame(index=full_idx)
        for c in list(MAP.values()) + ['soil_moisture_0_100cm_m3m3']: p[c] = pd.concat([w[c], o[c]]).reindex(full_idx)
        p['river_discharge_m3s'] = q_corr.reindex(full_idx).values; p['river_discharge_m3s_original_cell'] = q_orig.reindex(full_idx).values
        for c in ('river', 'basin', 'dhm_station', 'latitude', 'longitude', 'elevation_m'): p[c] = sites.loc[s, c]
        p['location'] = s; p['corrected_cell'] = bool(use_best); p.index.name = 'date'; out.append(p.reset_index())
    P = pd.concat(out); cols = list(df.columns) + ['river_discharge_m3s_original_cell', 'corrected_cell']
    P = P[cols]; P.to_csv('extended/panel_2010_2026.csv', index=False); pd.DataFrame(check).to_csv('results/extended/archive_vs_v2_check.csv', index=False)
    print(P.shape, P.isna().sum()[P.isna().sum() > 0].to_dict()); print(pd.DataFrame(check).round(3).to_string())
