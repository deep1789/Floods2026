import os, numpy as np, pandas as pd
CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv')
Q, P, TH = 'river_discharge_m3s', 'precipitation_mm', 'soil_moisture_0_100cm_m3m3'

def load(path=CSV):
    df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date'])
    return df.sort_values(['location', 'date']).reset_index(drop=True)

def site_order(df):
    return df.groupby('location').elevation_m.first().sort_values().index.tolist()

def audit(df):
    """Machine-checkable audit tests A1, A2, A3 (paper Sec. 3.4). Returns dict of pass/fail."""
    out = {}
    out['A1_complete'] = bool((df.groupby('location').size() == 1339).all() and not df.duplicated(['location', 'date']).any() and df.notna().all().all())
    out['A2_zero_consistent'] = bool((((df[P] == 0) == (df.precipitation_hours == 0))).all())
    out['A3_rain_le_precip'] = bool(((df[P] - df.rain_mm) > -1e-6).all())
    return out
