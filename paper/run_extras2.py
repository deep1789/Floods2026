"""Scenario analysis (7.6), isolation-forest anomaly detector (7.8), grouped permutation importance (6.5),
Duan smearing check (4.2), full-period leave-one-location-out. Run alone."""
import numpy as np, pandas as pd, warnings, json
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor as H
from floodlab import data, features as F, models as M, metrics as Mx
warnings.filterwarnings('ignore')
from floodlab.config import T0
df = data.load(); order = data.site_order(df); Q, P = data.Q, data.P
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
w = lambda n, t: open(f'tables2/{n}.md', 'w').write(t)
# ===== 1. scenario analysis: model trained WITHOUT the 2024 monsoon, applied to origins in the Sept-2024 storm
vs, ve = pd.Timestamp('2024-06-01'), pd.Timestamp('2024-09-30'); ex = (df.date >= vs - pd.Timedelta(days=7)) & (df.date <= ve + pd.Timedelta(days=30))
fit = F.Fit(df, ~ex); f = F.build(df, fit); f['sid'] = M.sid(f); h = 1
tr, te = F.split_rows(f, h, lambda d: ~((d.date >= vs - pd.Timedelta(days=7)) & (d.date <= ve + pd.Timedelta(days=30))), lambda d: (d.date >= '2024-09-25') & (d.date <= '2024-09-28'))
cat = ['site_id']; mdl = H(categorical_features=cat, **M.hgb_params()).fit(M.design(tr), tr.y_tgt - tr.y)
def scen(d, lam=1.0, dth=0.0):
    d = d.copy()
    for c in ['lp'] + [f'lp_l{k}' for k in (1, 2, 3)]: d[c] = np.log1p(lam * np.expm1(d[c]))
    for c in ['P_s3', 'P_s7', 'P_s14', 'P_s30', 'API80', 'API95', 'intens']: d[c] = d[c] * lam
    d['th_star'] = d['th_star'] + dth; return d
rows = []
for s in ('Khokana', 'Kusum', 'Devghat', 'Chatara'):
    for date in ('2024-09-26', '2024-09-27'):
        row = te[(te.location == s) & (te.date == date)]
        if len(row) == 0: continue
        base = np.expm1(row.y.values + mdl.predict(M.design(row)))[0]
        for name, lam, dth in (('baseline', 1, 0), ('soil −1σ', 1, -1), ('soil +1σ', 1, 1), ('rain ×0.8', .8, 0), ('rain ×1.2', 1.2, 0), ('rain ×1.2 & soil +1σ', 1.2, 1)):
            q = np.expm1(row.y.values + mdl.predict(M.design(scen(row, lam, dth))))[0]
            rows.append(dict(Location=s, Origin=date, Scenario=name, Q_pred=q, change_vs_baseline_pct=100 * (q / base - 1), Q_observed_next_day=row.Q_tgt.values[0]))
SC = pd.DataFrame(rows); SC.to_csv('results/scenarios.csv', index=False); w('scenarios', md(SC, '{:.2f}'))
print(SC.round(2).to_string())
# ===== 2. isolation forest (per site, unsupervised) on the rainfall-flow relationship; rank documented event days
fit_all = F.Fit(df, df.date >= '2000-01-01'); fa = F.build(df, fit_all); feats = ['dy', 'lp', 'lp_l1', 'P_s3', 'th_star', 'y_anom']
ev = [('Kusum', '2024-09-28', 'Sep-2024 storm (rain-driven)'), ('Khokana', '2024-09-28', 'Sep-2024 storm (rain-driven)'), ('Devghat', '2024-09-29', 'Sep-2024 storm (rain-driven)'), ('Chatara', '2024-08-16', 'Thame GLOF, nearest location'), ('Bahrabise', '2024-08-16', 'Thame GLOF, nearest Koshi site'), ('Rasuwagadhi', '2025-07-08', 'Bhote Koshi flash flood')]
rows = []
for s, d, lab in ev:
    g = fa[fa.location == s].dropna(subset=feats).reset_index(drop=True); Z = (g[feats] - g[feats].mean()) / g[feats].std()
    sc = -IsolationForest(n_estimators=300, random_state=0).fit(Z).score_samples(Z); g['a'] = sc; t = pd.Timestamp(d); win = g[(g.date >= t - pd.Timedelta(days=1)) & (g.date <= t + pd.Timedelta(days=1))]
    rows.append(dict(Location=s, Date=d, Event=lab, max_score=win.a.max(), percentile=(g.a < win.a.max()).mean() * 100))
IFD = pd.DataFrame(rows); IFD.to_csv('results/isolation_forest.csv', index=False); w('isoforest', md(IFD.rename(columns={'percentile': 'percentile within site record'}), '{:.2f}')); print(IFD.round(2).to_string())
# ===== 3. grouped permutation importance (chronological split, pooled HGB default)
GR = {'flow lags': ['y', 'y_l1', 'y_l2', 'y_l3', 'y_l7', 'dy', 'y_anom'], 'current rain': ['lp', 'lp_l1', 'lp_l2', 'lp_l3', 'P_s3', 'P_s7', 'intens', 'wet_spell'], 'antecedent rain': ['API80', 'API95', 'P_s14', 'P_s30'],
      'soil moisture': ['th_star', 'th_d7'], 'temperature, humidity, snow': ['temperature_mean_c', 'dtemp', 'pdd7', 'snowfrac', 'relative_humidity_mean_pct'], 'season': ['sin_doy', 'cos_doy'], 'site & elevation': ['elevation_m', 'site_id']}
fit = F.Fit(df, df.date < T0); f = F.build(df, fit); f['sid'] = M.sid(f); rng = np.random.default_rng(0); rows = []
for h in (1, 3, 7):
    tr, te = F.split_rows(f, h, lambda d: d.date_tgt < T0, lambda d: d.date >= T0); m = H(categorical_features=['site_id'], **M.hgb_params()).fit(M.design(tr), tr.y_tgt - tr.y)
    X = M.design(te); y = te.y_tgt.values - te.y.values; base = np.mean((y - m.predict(X)) ** 2)
    for g_, cols in GR.items():
        d_ = []
        for _ in range(10):
            Xp = X.copy(); ix = rng.permutation(len(X))
            for c in cols: Xp[c] = X[c].values[ix]
            d_.append(np.mean((y - m.predict(Xp)) ** 2) - base)
        rows.append(dict(h=h, Group=g_, dMSE_mean=np.mean(d_), dMSE_sd=np.std(d_), rel_increase_pct=100 * np.mean(d_) / base))
PI = pd.DataFrame(rows); PI.to_csv('results/perm_importance.csv', index=False)
w('perm_importance', md(PI.pivot(index='Group', columns='h', values='rel_increase_pct').add_prefix('h=').add_suffix(' (% MSE increase)').reset_index().sort_values('h=3 (% MSE increase)', ascending=False), '{:.1f}')); print(PI.pivot(index='Group', columns='h', values='rel_increase_pct').round(1))
# ===== 4. Duan smearing (chronological split)
rows = []
for h in (1, 3, 7):
    tr, te = F.split_rows(f, h, lambda d: d.date_tgt < T0, lambda d: d.date >= T0); yh, mdl2 = M.hgb_fit_predict(tr, te, h); ytr, _ = M.hgb_fit_predict(tr, tr, h)
    sm = np.mean(np.exp(tr.y_tgt.values - ytr)); a = te.assign(yh=yh)
    for s, g_ in a.groupby('location'):
        rows.append(dict(h=h, location=s, NSE_plain=Mx.nse(g_.Q_tgt, np.clip(np.expm1(g_.yh), 0, None)), NSE_smear=Mx.nse(g_.Q_tgt, np.clip(sm * np.exp(g_.yh) - 1, 0, None)), smear=sm))
SM = pd.DataFrame(rows); t = SM.groupby('h')[['NSE_plain', 'NSE_smear', 'smear']].agg({'NSE_plain': 'median', 'NSE_smear': 'median', 'smear': 'first'}).reset_index(); t.columns = ['h', 'median NSE (expm1)', 'median NSE (Duan smearing)', 'smearing factor']; w('smearing', md(t)); print(t)
# ===== 5. LOLO full-period (spatial only): nine sites all dates -> tenth site all dates, no site id / elevation
fit = F.Fit(df, df.date >= '2000-01-01'); f = F.build(df, fit); f['sid'] = M.sid(f); rows = []
for h in (1, 3, 7):
    for s in order:
        tr, te = F.split_rows(f, h, lambda d, s=s: d.location != s, lambda d, s=s: d.location == s); yh, _ = M.hgb_fit_predict(tr, te, h, use_site=False, use_elev=False)
        rows.append(dict(h=h, location=s, logNSE_persist=Mx.nse(te.y_tgt, te.y), logNSE_lolo=Mx.nse(te.y_tgt, yh), NSE_persist=Mx.nse(te.Q_tgt, np.expm1(te.y)), NSE_lolo=Mx.nse(te.Q_tgt, np.clip(np.expm1(yh), 0, None))))
LF = pd.DataFrame(rows); LF.to_csv('results/lolo_full.csv', index=False)
t = LF.groupby('h')[['logNSE_persist', 'logNSE_lolo', 'NSE_persist', 'NSE_lolo']].median().reset_index(); t['sites better than persistence (log-NSE)'] = LF.groupby('h').apply(lambda x: f'{(x.logNSE_lolo > x.logNSE_persist).sum()}/10').values; w('lolo_full', md(t)); print(t)
