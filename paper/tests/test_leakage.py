"""Leakage tests (paper Sec. 10 step 2): perturbing data after cutoff t0 must not change
(a) any feature at origins <= t0, (b) predictions at origins <= t0 of a model fitted on data before t0."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd
from floodlab import data, features as F, models as M
df = data.load(); print('audit', data.audit(df))
t0 = pd.Timestamp('2025-03-01'); train_end = pd.Timestamp('2024-12-31')
def feats(d):
    fit = F.Fit(d, d.date <= train_end); f = F.build(d, fit); f['sid'] = M.sid(f); return fit, f
fit1, f1 = feats(df)
rng = np.random.default_rng(1); d2 = df.copy(); d2 = d2.astype({c: float for c in ['precipitation_mm','rain_mm','river_discharge_m3s','soil_moisture_0_100cm_m3m3','temperature_mean_c','relative_humidity_mean_pct']}); late = d2.date > t0
for c in ['precipitation_mm', 'rain_mm', 'river_discharge_m3s', 'soil_moisture_0_100cm_m3m3', 'temperature_mean_c', 'relative_humidity_mean_pct']:
    d2.loc[late, c] = d2.loc[late, c] * rng.uniform(0.2, 3, late.sum())
fit2, f2 = feats(d2)
a, b = f1[f1.date <= t0].reset_index(drop=True), f2[f2.date <= t0].reset_index(drop=True)
bad = [c for c in F.FEATS if not np.allclose(a[c].values, b[c].values, equal_nan=True)]
assert not bad, f'features leak future info: {bad}'
print('PASS (a): all', len(F.FEATS), 'features at origins <= t0 are unchanged by future perturbation')
h = 3
def preds(f):
    tr, te = F.split_rows(f, h, lambda d: d.date_tgt <= train_end, lambda d: d.date <= t0)
    te = te[te.date > train_end]; p, _ = M.hgb_fit_predict(tr, te, h); return te.date.values, p
d1_, p1 = preds(f1); d2_, p2 = preds(f2)
assert (d1_ == d2_).all() and np.allclose(p1, p2), 'model predictions changed when future data were perturbed'
print('PASS (b): HGB predictions for', len(p1), 'origins unchanged by future perturbation')
# negative control: the test must be able to fail. A deliberately leaky feature (centered rolling mean) must be detected.
g = df[df.location == 'Khokana'].copy(); lk = g.river_discharge_m3s.rolling(5, center=True).mean()
g2 = g.copy(); g2.loc[g2.date > t0, 'river_discharge_m3s'] *= 2; lk2 = g2.river_discharge_m3s.rolling(5, center=True).mean()
assert not np.allclose(lk[g.date <= t0].fillna(0), lk2[g2.date <= t0].fillna(0)), 'negative control failed'
print('PASS (c): negative control - a centered rolling feature is detected as leaking')
