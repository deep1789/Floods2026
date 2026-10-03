"""Feature-group ablation (Sec. 6.5): leave-one-monsoon-out pooled HGB, drop one feature group at a time (and 'lags only')."""
import numpy as np, pandas as pd, warnings
from floodlab import data, features as F, models as M, metrics as Mx
warnings.filterwarnings('ignore')
from floodlab.config import ABL_YEARS
df = data.load(); order = data.site_order(df)
GROUPS = {'soil (θ*, Δθ7)': ['th_star', 'th_d7'], 'antecedent rain (API, 14/30-d sums)': ['API80', 'API95', 'P_s14', 'P_s30'], 'current rain (lags 0-3, 3/7-d sums, intensity, wet spell)': ['lp', 'lp_l1', 'lp_l2', 'lp_l3', 'P_s3', 'P_s7', 'intens', 'wet_spell'],
          'temperature/humidity/snow (τ, Δτ, PDD, RH, snow frac.)': ['temperature_mean_c', 'dtemp', 'pdd7', 'snowfrac', 'relative_humidity_mean_pct'], 'season (sin/cos doy)': ['sin_doy', 'cos_doy', 'y_anom'],
          'site & elevation': ['__site__']}
LAGS_ONLY = ['y', 'y_l1', 'y_l2', 'y_l3', 'y_l7', 'dy']
def run(drop=None, only=None, hs=(1, 3)):
    res = []
    for yr in ABL_YEARS:
        vs, ve = pd.Timestamp(f'{yr}-06-01'), min(pd.Timestamp(f'{yr}-09-30'), df.date.max()); ex = (df.date >= vs - pd.Timedelta(days=7)) & (df.date <= ve + pd.Timedelta(days=30))
        fit = F.Fit(df, ~ex); f = F.build(df, fit); f['sid'] = M.sid(f)
        for h in hs:
            tr, te = F.split_rows(f, h, lambda d: ~((d.date >= vs - pd.Timedelta(days=7)) & (d.date <= ve + pd.Timedelta(days=30))), lambda d: (d.date >= vs) & (d.date <= ve))
            cols = list(F.FEATS) if only is None else list(only); site = True
            if drop:
                for c in GROUPS[drop]:
                    if c == '__site__': site = False
                    elif c in cols: cols.remove(c)
            if only is not None: site = False
            Xtr = tr[cols].copy(); Xte = te[cols].copy()
            if site: Xtr['elevation_m'] = tr.elevation_m.values; Xte['elevation_m'] = te.elevation_m.values; Xtr['site_id'] = tr.sid.values; Xte['site_id'] = te.sid.values
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            p = M.hgb_params(); mdl = H(categorical_features=['site_id'] if site else None, **p).fit(Xtr, tr.y_tgt - tr.y)
            res.append(pd.DataFrame(dict(h=h, location=te.location.values, y_obs=te.y_tgt.values, yhat=te.y.values + mdl.predict(Xte))))
    r = pd.concat(res); out = {}
    for h in hs:
        g = r[r.h == h].groupby('location').apply(lambda x: Mx.nse(x.y_obs, x.yhat)); gq = r[r.h == h].groupby('location').apply(lambda x: Mx.nse(np.expm1(x.y_obs), np.clip(np.expm1(x.yhat), 0, None)))
        out[h] = (g.median(), gq.median(), g.reindex(order).values)
    return out
base = run(); rows = [dict(Variant='all features (reference)', **{f'h={h} log-NSE': base[h][0] for h in (1, 3)}, **{f'h={h} NSE': base[h][1] for h in (1, 3)}, **{f'h={h} Δlog-NSE': 0.0 for h in (1, 3)})]
for g in GROUPS:
    o = run(drop=g); rows.append(dict(Variant='drop ' + g, **{f'h={h} log-NSE': o[h][0] for h in (1, 3)}, **{f'h={h} NSE': o[h][1] for h in (1, 3)}, **{f'h={h} Δlog-NSE': o[h][0] - base[h][0] for h in (1, 3)}))
o = run(only=LAGS_ONLY); rows.append(dict(Variant='flow lags only (no weather, no site)', **{f'h={h} log-NSE': o[h][0] for h in (1, 3)}, **{f'h={h} NSE': o[h][1] for h in (1, 3)}, **{f'h={h} Δlog-NSE': o[h][0] - base[h][0] for h in (1, 3)}))
A = pd.DataFrame(rows)[['Variant', 'h=1 log-NSE', 'h=1 Δlog-NSE', 'h=1 NSE', 'h=3 log-NSE', 'h=3 Δlog-NSE', 'h=3 NSE']]; A.to_csv('results/ablation.csv', index=False)
h = '| ' + ' | '.join(A.columns) + ' |\n|' + '|'.join(['---'] * len(A.columns)) + '|\n'
open('tables2/ablation.md', 'w').write(h + '\n'.join('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r) + ' |' for r in A.values) + '\n')
print(A.round(3).to_string())
# simultaneous extremes
P = df.pivot(index='date', columns='location', values=data.P); ex = P.gt(P.quantile(.95)); n = ex.sum(1)
print('sites above own q95 precip: 2024-07-06', n['2024-07-06'], ' 2024-09-27', n['2024-09-27'], ' 2024-09-28', n['2024-09-28'], ' max days', n.nlargest(5).to_dict())
Qp = df.pivot(index='date', columns='location', values=data.Q); exq = Qp.gt(Qp.quantile(.95)); nq = exq.sum(1)
print('sites above own q95 discharge: 2024-09-28', nq['2024-09-28'], '2024-09-29', nq['2024-09-29'], nq.nlargest(5).to_dict())
