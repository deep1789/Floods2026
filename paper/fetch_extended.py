"""UNTESTED in this environment (the sandbox network policy blocks open-meteo.com hosts). Run where those hosts are reachable.
Purpose: fix the two data blockers identified in the audit
  (1) discharge scale / grid-cell snapping (audit A4): scan a (2k+1)x(2k+1) neighbourhood of GloFAS cells around each coordinate;
  (2) short record: extend weather + modelled discharge back in time (GloFAS v4 reanalysis starts 1984; ERA5-based weather longer).
Usage:  python fetch_extended.py scan                 # step (1): writes results/extended/cell_scan.csv
        python fetch_extended.py extend 1990 2022     # step (2): writes results/extended/<site>.csv (weather + discharge, default and best cell)
Variable names follow the Open-Meteo docs as understood at writing time; verify them against the current API reference before relying on the output."""
import sys, os, time, json, urllib.request, urllib.parse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from floodlab import data
W_API = 'https://archive-api.open-meteo.com/v1/archive'; F_API = 'https://flood-api.open-meteo.com/v1/flood'
DAILY = 'precipitation_sum,rain_sum,temperature_2m_mean,dew_point_2m_mean,precipitation_hours,relative_humidity_2m_mean,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant'
OUT = 'results/extended'; os.makedirs(OUT, exist_ok=True)
def get(url, params, tries=4):
    q = url + '?' + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=60) as r: return json.load(r)
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2 ** (i + 1))
def flood(lat, lon, a, b):
    j = get(F_API, dict(latitude=lat, longitude=lon, daily='river_discharge', start_date=a, end_date=b)); return pd.Series(j['daily']['river_discharge'], index=pd.to_datetime(j['daily']['time']))
def scan(k=2, step=0.05):
    df = data.load(); sites = df.groupby('location')[['latitude', 'longitude']].first(); rows = []
    for s, (la, lo) in sites.iterrows():
        ref = df[df.location == s].set_index('date').river_discharge_m3s
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                q = flood(round(la + i * step, 4), round(lo + j * step, 4), '2023-01-01', '2026-08-31'); c = q.reindex(ref.index)
                rows.append(dict(location=s, di=i, dj=j, mean_Q=c.mean(), max_Q=c.max(), corr_with_dataset=np.corrcoef(c.fillna(0), ref)[0, 1], max_abs_diff_vs_dataset=float((c - ref).abs().max())))
        print(s, 'scanned', flush=True)
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/cell_scan.csv', index=False); print(r.groupby('location').apply(lambda g: g.loc[g.mean_Q.idxmax()]).to_string())
def extend(y0, y1):
    df = data.load(); sites = df.groupby('location')[['latitude', 'longitude']].first(); sc = pd.read_csv(f'{OUT}/cell_scan.csv') if os.path.exists(f'{OUT}/cell_scan.csv') else None
    for s, (la, lo) in sites.iterrows():
        best = (0, 0) if sc is None else tuple(sc[sc.location == s].loc[lambda g: g.mean_Q.idxmax(), ['di', 'dj']].astype(int))
        j = get(W_API, dict(latitude=la, longitude=lo, start_date=f'{y0}-01-01', end_date=f'{y1}-12-31', daily=DAILY)); w = pd.DataFrame(j['daily']).rename(columns={'time': 'date'}); w['date'] = pd.to_datetime(w.date)
        h = get(W_API, dict(latitude=la, longitude=lo, start_date=f'{y0}-01-01', end_date=f'{y1}-12-31', hourly='soil_moisture_0_to_100cm'))   # NOTE: verify variable name / availability
        sm = pd.Series(h['hourly']['soil_moisture_0_to_100cm'], index=pd.to_datetime(h['hourly']['time'])).resample('D').mean(); w = w.merge(sm.rename('soil_moisture_0_100cm_m3m3'), left_on='date', right_index=True, how='left')
        w['river_discharge_m3s_dataset_cell'] = flood(la, lo, f'{y0}-01-01', f'{y1}-12-31').reindex(w.date).values
        w['river_discharge_m3s_best_cell'] = flood(round(la + best[0] * 0.05, 4), round(lo + best[1] * 0.05, 4), f'{y0}-01-01', f'{y1}-12-31').reindex(w.date).values
        w.insert(1, 'location', s); w.to_csv(f"{OUT}/{s.replace('/', '_')}.csv", index=False); print(s, 'written', len(w), flush=True)
if __name__ == '__main__':
    if sys.argv[1] == 'scan': scan()
    elif sys.argv[1] == 'extend': extend(int(sys.argv[2]), int(sys.argv[3]))
