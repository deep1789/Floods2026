"""Algorithm 1: leakage-safe feature construction.
Rule: a feature at origin t may use data dated <= t only. Every fitted quantity
(climatology, recession constants, thresholds, scalers) is estimated on the training mask only."""
import numpy as np, pandas as pd
from .data import Q, P, TH

K_HARM = 3
def _design(doy, k=K_HARM):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    cols = [np.ones_like(w)] + [f(j * w) for j in range(1, k + 1) for f in (np.sin, np.cos)]
    return np.column_stack(cols)

class Fit:
    """Quantities estimated on the training mask only (Algorithm 1, steps 3-6)."""
    def __init__(self, df, train_mask):
        self.th, self.thsd, self.yclim, self.c, self.thr, self.ymu, self.ysd = {}, {}, {}, {}, {}, {}, {}
        tr = df[train_mask]
        for s, g in tr.groupby('location'):
            doy = g.date.dt.dayofyear.values
            X = _design(doy)
            b = np.linalg.lstsq(X, g[TH].values, rcond=None)[0]; self.th[s] = b
            self.thsd[s] = float(np.std(g[TH].values - X @ b) + 1e-6)
            y = np.log1p(g[Q].values); self.yclim[s] = np.linalg.lstsq(X, y, rcond=None)[0]
            self.ymu[s], self.ysd[s] = float(y.mean()), float(y.std() + 1e-6)
            self.thr[s] = {q: float(g[Q].quantile(q)) for q in (0.90, 0.95, 0.99)}
            z = g.assign(Qn=g[Q].shift(-1), Pn=g[P].shift(-1))
            d = z[(z[P] == 0) & (z.Pn == 0) & (z.Qn <= z[Q]) & (z[Q] > 0.5)]
            self.c[s] = (float(np.clip(np.median(d.Qn / d[Q]), 0.5, 1.0)), len(d))
        pooled = np.median([c for c, n in self.c.values() if n >= 10]) if any(n >= 10 for _, n in self.c.values()) else 0.95
        # sites with too few dry pairs fall back to the pooled constant (paper Sec. 4.9)
        self.c = {s: (c if n >= 10 else pooled) for s, (c, n) in self.c.items()}

    def clim_theta(self, s, doy): return _design(doy) @ self.th[s]
    def clim_y(self, s, doy): return _design(doy) @ self.yclim[s]

LAGP = list(range(1, 15))
def build(df, fit):
    """Return a copy of df with causal features. No look-ahead columns except explicit targets (added separately)."""
    out = []
    for s, g in df.groupby('location', sort=False):
        g = g.sort_values('date').copy()
        doy = g.date.dt.dayofyear.values
        g['y'] = np.log1p(g[Q]); g['lp'] = np.log1p(g[P])
        for k in (1, 2, 3, 7): g[f'y_l{k}'] = g.y.shift(k)
        for k in LAGP: g[f'lp_l{k}'] = g.lp.shift(k)
        for w in (3, 7, 14, 30): g[f'P_s{w}'] = g[P].rolling(w).sum()
        for k in (0.8, 0.95):
            a = 1 - k; g[f'API{int(k*100)}'] = g[P].ewm(alpha=a, adjust=False).mean() / a
        g['th_star'] = (g[TH] - fit.clim_theta(s, doy)) / fit.thsd[s]
        g['th_d7'] = g[TH] - g[TH].shift(7)
        g['dtemp'] = g.temperature_mean_c - g.temperature_mean_c.shift(1)
        g['pdd7'] = g.temperature_mean_c.clip(lower=0).rolling(7).sum()
        g['snowfrac'] = np.where(g[P] > 0, (g[P] - g.rain_mm) / g[P].where(g[P] > 0, 1), 0.0)
        wet = (g[P] > 0).astype(int); grp = (wet == 0).cumsum(); g['wet_spell'] = wet.groupby(grp).cumsum()
        g['intens'] = g[P] / g.precipitation_hours.clip(lower=1)
        g['dy'] = g.y - g.y_l1
        g['sin_doy'] = np.sin(2 * np.pi * doy / 365.25); g['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
        g['y_clim'] = fit.clim_y(s, doy)
        g['y_anom'] = g.y - g.y_clim
        out.append(g)
    return pd.concat(out).sort_values(['location', 'date']).reset_index(drop=True)

FEATS = ['y', 'y_l1', 'y_l2', 'y_l3', 'y_l7', 'dy', 'y_anom', 'lp', 'lp_l1', 'lp_l2', 'lp_l3',
         'P_s3', 'P_s7', 'P_s14', 'P_s30', 'API80', 'API95', 'th_star', 'th_d7',
         'temperature_mean_c', 'dtemp', 'pdd7', 'snowfrac', 'relative_humidity_mean_pct', 'wet_spell', 'intens',
         'sin_doy', 'cos_doy']
STATIC = ['site_id', 'elevation_m']

def add_targets(feat, h):
    g = feat.groupby('location', sort=False)
    feat = feat.copy()
    feat['y_tgt'] = g.y.shift(-h); feat['Q_tgt'] = g[Q].shift(-h)
    feat['date_tgt'] = g.date.shift(-h)
    return feat

def split_rows(feat, h, train_mask_fn, test_mask_fn, purge_fn=None):
    """Return (train, test) frames of complete rows. Masks act on origin date; purge_fn drops
    train origins whose feature window or target overlaps the test block."""
    d = add_targets(feat, h).dropna(subset=FEATS + ['y_tgt'])
    tr = d[train_mask_fn(d)]
    if purge_fn is not None: tr = tr[~purge_fn(tr)]
    te = d[test_mask_fn(d)]
    return tr, te
