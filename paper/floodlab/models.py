"""Baselines B0-B3, pooled HGB (point + quantile), joint LSTM. All return predictions of y_{t+h}=log(1+Q_{t+h})."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import RidgeCV
from .features import FEATS, LAGP, _design
warnings.filterwarnings('ignore')

def sid(feat):
    cats = sorted(feat.location.unique()); return feat.location.map({c: i for i, c in enumerate(cats)}).values

def b0_persistence(te, h, fit=None): return te.y.values
def b1_recession(te, h, fit):
    c = te.location.map({s: fit.c[s] for s in fit.c}).values
    dry = ((te.precipitation_mm == 0) & (te.lp_l1 == 0)).values   # no rain today or yesterday -> recession; else persist
    q = np.expm1(te.y.values); qh = np.where(dry, q * c ** h, q); return np.log1p(qh)
def b2_climatology(te, h, fit):
    out = np.empty(len(te))
    for s in te.location.unique():
        m = (te.location == s).values; doy = (te.date_tgt[m].dt.dayofyear.values); out[m] = fit.clim_y(s, doy)
    return out
def b3_arx(tr, te, h, fit):
    """Per-site ridge distributed-lag ARX on the increment (Sec. 5.3)."""
    cols = ['lp'] + [f'lp_l{k}' for k in LAGP] + ['th_star', 'dtemp', 'y_anom']
    out = np.empty(len(te))
    for s in te.location.unique():
        a = tr[tr.location == s]; m = (te.location == s).values
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(a[cols], a.y_tgt - a.y)
        out[m] = te.y.values[m] + mdl.predict(te.loc[m, cols])
    return out

def hgb_params(): return dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
def design(d, use_site=True, use_elev=True):
    X = d[FEATS].copy()
    if use_elev: X['elevation_m'] = d.elevation_m.values
    if use_site: X['site_id'] = d.sid.values
    return X
def hgb_fit_predict(tr, te, h, use_site=True, use_elev=True, loss='squared_error', q=None, params=None):
    p = dict(hgb_params()); p.update(params or {})
    cat = ['site_id'] if use_site else None
    mdl = HGB(loss=loss, quantile=q, categorical_features=cat, **p) if loss == 'quantile' else HGB(categorical_features=cat, **p)
    Xtr, Xte = design(tr, use_site, use_elev), design(te, use_site, use_elev)
    mdl.fit(Xtr, tr.y_tgt - tr.y); return te.y.values + mdl.predict(Xte), mdl

def back(yhat, smear=1.0): return np.clip(smear * np.exp(yhat) - 1, 0, None)
