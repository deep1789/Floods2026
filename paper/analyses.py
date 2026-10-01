"""Process analyses (paper Sec. 7.1, 7.2, 7.5, 7.7, 4.9). Descriptive: fitted on the full record, uses no train/test split."""
import numpy as np, pandas as pd, warnings, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from floodlab import data, features as F, metrics as Mx, events as Ev
warnings.filterwarnings('ignore')
df = data.load(); order = data.site_order(df); Q, P = data.Q, data.P
fit = F.Fit(df, df.date >= '2000-01-01'); f = F.build(df, fit)     # full-record fit: descriptive analysis only
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
rng = np.random.default_rng(0); out = {}
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
w = lambda n, t: open(f'tables2/{n}.md', 'w').write(t)

# ================= 7.1 distributed-lag weights, prewhitened =================
J = 10
def lag_design(g):
    doy = g.date.dt.dayofyear.values; H = F._design(doy)[:, 1:]                      # seasonal harmonics
    X = np.column_stack([g.lp.values] + [g[f'lp_l{k}'].values for k in range(1, J + 1)])
    ec = g.y_anom.shift(1).values                                                  # error-correction (AR) term
    Z = np.column_stack([X, ec, H]); y = g.dy.values
    ok = ~np.isnan(Z).any(1) & ~np.isnan(y); return Z[ok], y[ok]
def pen_matrix(J, lam):
    D = np.zeros((J - 1, J + 1))
    for i in range(J - 1): D[i, i:i + 3] = [1, -2, 1]
    return lam * D.T @ D
def ridge_w(Z, y, lam):
    p = Z.shape[1]; Pm = np.zeros((p, p)); Pm[:J + 1, :J + 1] = pen_matrix(J, lam); Pm += 1e-6 * np.eye(p)
    Zc = np.column_stack([np.ones(len(Z)), Z]); Pm2 = np.zeros((p + 1, p + 1)); Pm2[1:, 1:] = Pm
    b = np.linalg.solve(Zc.T @ Zc + Pm2, Zc.T @ y); return b[1:J + 2], b
def pick_lambda(g, grid=(0.1, 1, 10, 100, 1000)):
    Z, y = lag_design(g); d = g.date.values[~np.isnan(np.column_stack([lag_design(g)[0]]).sum(1))] if False else None
    n = len(y); blocks = np.array_split(np.arange(n), 4); best = (1e18, None)
    for lam in grid:
        e = 0
        for b in blocks:
            tr = np.setdiff1d(np.arange(n), np.arange(max(0, b[0] - 14), min(n, b[-1] + 15)))
            wv, bb = ridge_w(Z[tr], y[tr], lam); Zc = np.column_stack([np.ones(len(b)), Z[b]]); e += ((y[b] - Zc @ bb) ** 2).sum()
        if e < best[0]: best = (e, lam)
    return best[1]
rows, curves = [], {}
for s in order:
    g = f[f.location == s].reset_index(drop=True); lam = pick_lambda(g); Z, y = lag_design(g)
    wv, _ = ridge_w(Z, y, lam); pos = np.clip(wv, 0, None); cen = (np.arange(J + 1) * pos).sum() / pos.sum() if pos.sum() > 0 else np.nan
    bs = []
    for _ in range(200):
        ix = Mx.stationary_bootstrap_idx(len(y), 10, rng); wb, _ = ridge_w(Z[ix], y[ix], lam); pb = np.clip(wb, 0, None)
        bs.append((int(np.argmax(wb)), (np.arange(J + 1) * pb).sum() / pb.sum() if pb.sum() > 0 else np.nan, wb.sum(), wb))
    cen_ci = np.nanpercentile([b[1] for b in bs], [2.5, 97.5]); cum_ci = np.percentile([b[2] for b in bs], [2.5, 97.5])
    curves[s] = (wv, np.percentile([b[3] for b in bs], [2.5, 97.5], axis=0))
    rows.append(dict(Location=s, lam=lam, peak_lag=int(np.argmax(wv)), centroid=cen, centroid_lo=cen_ci[0], centroid_hi=cen_ci[1], cum_response=wv.sum(), cum_lo=cum_ci[0], cum_hi=cum_ci[1], w0=wv[0], w1=wv[1], w2=wv[2], n=len(y)))
LG = pd.DataFrame(rows); LG.to_csv('results/lag_weights.csv', index=False)
t = LG[['Location', 'lam', 'peak_lag', 'centroid', 'centroid_lo', 'centroid_hi', 'cum_response', 'cum_lo', 'cum_hi']].copy()
t.columns = ['Location', 'λ', 'Peak lag (d)', 'Centroid j̄ (d)', 'j̄ 2.5%', 'j̄ 97.5%', 'Σw', 'Σw 2.5%', 'Σw 97.5%']; w('lagweights', md(t, '{:.2f}'))
fig, axs = plt.subplots(2, 5, figsize=(11, 4), sharex=True)
for ax, s in zip(axs.ravel(), order):
    wv, ci = curves[s]; ax.fill_between(range(J + 1), ci[0], ci[1], alpha=.25); ax.plot(range(J + 1), wv, marker='o', ms=3); ax.axhline(0, c='k', lw=.5); ax.set_title(s, fontsize=8)
fig.supxlabel('rain lag j (days)'); fig.supylabel('weight w_j on log(1+P_{t-j}) in Δlog(1+Q)_t'); fig.tight_layout(); fig.savefig('figures/fig8_lagweights.png', dpi=160); plt.close()

# ================= 7.2 soil moisture and event amplification =================
ev = []
for s in order:
    g = f[f.location == s].reset_index(drop=True); u = g.P_s3.quantile(0.90); q50 = g[Q].median()
    for (a, pk, b, zp) in Ev.detect_events(g.P_s3.fillna(0).values, u, r=3):
        if a < 6 or b + 5 >= len(g): continue
        qpre = g[Q].iloc[a - 5:a].median(); qpk = g[Q].iloc[a:b + 6].max()
        ev.append(dict(location=s, date=g.date.iloc[a], Pe=zp, qpre=qpre, dQ=max(qpk - qpre, 0), q50=q50, th_pre=g.th_star.iloc[a - 1], api_pre=g.API95.iloc[a - 1], th_raw=g.soil_moisture_0_100cm_m3m3.iloc[a - 1], monsoon=int(g.date.iloc[a].month in (6, 7, 8, 9))))
EV = pd.DataFrame(ev).dropna(); EV['resp'] = np.log((EV.dQ + 0.01 * EV.q50) / EV.q50); EV['lPe'] = np.log(EV.Pe); EV['lqpre'] = np.log((EV.qpre + 1e-3) / EV.q50)
EV.to_csv('results/rain_events.csv', index=False)
mm = smf.mixedlm('resp ~ th_pre + api_pre_s + lPe + lqpre + monsoon', EV.assign(api_pre_s=EV.api_pre / EV.api_pre.std()), groups=EV['location']).fit(reml=True)
beta = mm.params['th_pre']; perm = []
for _ in range(500):
    e2 = EV.copy(); e2['th_pre'] = e2.groupby('location').th_pre.transform(lambda x: rng.permutation(x.values))
    perm.append(smf.mixedlm('resp ~ th_pre + api_pre_s + lPe + lqpre + monsoon', e2.assign(api_pre_s=e2.api_pre / e2.api_pre.std()), groups=e2['location']).fit(reml=True).params['th_pre'])
pperm = (np.abs(perm) >= abs(beta)).mean()
tt = pd.DataFrame({'Term': mm.params.index[:-1], 'Estimate': mm.params.values[:-1], 'SE': mm.bse.values[:-1], 'z': mm.tvalues.values[:-1], 'p': mm.pvalues.values[:-1]})
tt.loc[len(tt)] = ['**permutation p for θ\\* (500 within-site shuffles)**', np.nan, np.nan, np.nan, pperm]; w('soilmodel', md(tt, '{:.4f}'))
out['n_events'] = len(EV); out['beta_theta'] = beta; out['perm_p'] = pperm; out['sigma_u'] = float(mm.cov_re.iloc[0, 0] ** .5)
# also per-site Spearman between theta* and amplification residual
sp = []
for s, g in EV.groupby('location'):
    if len(g) >= 8: r, p = stats.spearmanr(g.th_pre, g.resp - smf.ols('resp ~ lPe + lqpre', g).fit().predict(g)); sp.append((s, len(g), r, p))
SP = pd.DataFrame(sp, columns=['Location', 'events', 'Spearman rho (θ*, resid.)', 'p']); w('soil_site', md(SP.set_index('Location').loc[[o for o in order if o in set(SP.Location)]].reset_index(), '{:.3f}'))
fig, ax = plt.subplots(figsize=(5, 3.6)); sc = ax.scatter(EV.th_pre, EV.resp - smf.ols('resp ~ lPe + lqpre + C(location)', EV).fit().predict(EV).values + EV.resp.mean(), c=EV.monsoon, s=14, cmap='coolwarm')
ax.set_xlabel('pre-event soil-moisture anomaly θ*'); ax.set_ylabel('log amplification, adjusted for P_e, Q_pre, site'); ax.set_title(f'{len(EV)} rain events; mixed-model β(θ*)={beta:.3f}'); fig.tight_layout(); fig.savefig('figures/fig9_soil.png', dpi=160); plt.close()

# ================= 7.5 extremes: POT/GPD, leave-event-out, joint extremes =================
def gpd_fit(x):
    c, loc, sc = stats.genpareto.fit(x, floc=0); return c, sc
rows = []; EVT = (pd.Timestamp('2024-09-24'), pd.Timestamp('2024-09-30'))
for var, lab in ((P, 'P'), (Q, 'Q')):
    for s in order:
        g = df[df.location == s].reset_index(drop=True); z = g[var].values.astype(float)
        if lab == 'Q': z = z / max(np.median(z), 1e-3)
        u = np.quantile(z, 0.95); evs = Ev.detect_events(z, u, r=3); pk = np.array([e[3] for e in evs]); ex = pk - u
        mask = ((g.date >= EVT[0]) & (g.date <= EVT[1])).values
        zk = z.copy(); zk[mask] = -1                     # drop the event window, keep the series length
        evs_k = [e for e in Ev.detect_events(zk, u, r=3)]; exk = np.array([e[3] for e in evs_k]) - u
        win_pk = z[mask].max()
        if len(exk) < 8: continue
        xi, sg = gpd_fit(exk[exk > 0]); xs = []
        for _ in range(300):
            b = rng.choice(exk, len(exk)); c, sc_ = gpd_fit(b); xs.append(c)
        yrs = (g.date.max() - g.date.min()).days / 365.25 - (EVT[1] - EVT[0]).days / 365.25
        rate = len(exk) / yrs; sf = stats.genpareto.sf(max(win_pk - u, 1e-9), xi, 0, sg)
        rows.append(dict(Variable=lab, Location=s, u=u, n_peaks=len(exk), xi=xi, xi_lo=np.percentile(xs, 2.5), xi_hi=np.percentile(xs, 97.5), sigma=sg, event_peak=win_pk, above_u=win_pk > u, return_period_yr=(1 / (rate * sf) if sf > 0 else np.inf) if win_pk > u else np.nan, rate_per_yr=rate))
GP = pd.DataFrame(rows); GP.to_csv('results/gpd.csv', index=False)
for lab in ('P', 'Q'):
    t = GP[GP.Variable == lab][['Location', 'n_peaks', 'xi', 'xi_lo', 'xi_hi', 'event_peak', 'u', 'return_period_yr']].copy()
    t['event_peak'] = t.event_peak.round(1); t['u'] = t.u.round(1); t.columns = ['Location', 'peaks (event-excluded)', 'ξ̂', 'ξ 2.5%', 'ξ 97.5%', 'event max (P mm or Q/median)', 'threshold u', 'implied return period (yr)']
    t['implied return period (yr)'] = [('beyond fitted upper bound' if np.isinf(v) else ('below threshold' if np.isnan(v) else f'{v:.1f}')) for v in t['implied return period (yr)']]
    w(f'gpd_{lab}', md(t, '{:.2f}'))
# joint extremes
piv = {v: df.pivot(index='date', columns='location', values=v)[order] for v in (P, Q)}
for v, lab in ((P, 'P'), (Q, 'Q')):
    ex = piv[v].gt(piv[v].quantile(0.95)); ext = ex.sum(1)
    chi = pd.DataFrame(index=order, columns=order, dtype=float)
    for a in order:
        for b in order: chi.loc[a, b] = (ex[a] & ex[b]).sum() / ex[a].sum()
    off = chi.values[~np.eye(len(order), dtype=bool)]
    out[f'chi_{lab}_mean'] = off.mean(); out[f'chi_{lab}_max'] = off.max(); out[f'days_ge5_{lab}'] = int((ext >= 5).sum()); out[f'days_ge3_{lab}'] = int((ext >= 3).sum()); out[f'max_sites_{lab}'] = int(ext.max())
    out[f'maxdate_{lab}'] = str(ext.idxmax().date())
    chi.round(2).to_csv(f'results/chi_{lab}.csv')
    if lab == 'P': chiP = chi
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, v, lab in zip(axs, (P, Q), ('P', 'Q')):
    c = pd.read_csv(f'results/chi_{lab}.csv', index_col=0); im = ax.imshow(c.values, vmin=0, vmax=1, cmap='viridis'); ax.set_xticks(range(10)); ax.set_xticklabels(order, rotation=90, fontsize=6); ax.set_yticks(range(10)); ax.set_yticklabels(order, fontsize=6); ax.set_title(f'empirical χ(0.95), {lab}')
fig.colorbar(im, ax=axs, shrink=.8); fig.savefig('figures/fig10_chi.png', dpi=160, bbox_inches='tight'); plt.close()
w('chi_summary', md(pd.DataFrame([{'Variable': 'P', 'mean off-diag χ': out['chi_P_mean'], 'max χ': out['chi_P_max'], 'days with ≥3 sites >q95': out['days_ge3_P'], 'days ≥5 sites': out['days_ge5_P'], 'max sites on one day': out['max_sites_P'], 'date': out['maxdate_P']},
    {'Variable': 'Q', 'mean off-diag χ': out['chi_Q_mean'], 'max χ': out['chi_Q_max'], 'days with ≥3 sites >q95': out['days_ge3_Q'], 'days ≥5 sites': out['days_ge5_Q'], 'max sites on one day': out['max_sites_Q'], 'date': out['maxdate_Q']}]), '{:.3f}'))

# ================= 7.7 upstream-downstream: prewhitened cross-correlation conditioning on local rain =================
def resid(s):
    g = f[f.location == s].reset_index(drop=True); Z, y = lag_design(g)
    wv, b = ridge_w(Z, y, 10); Zc = np.column_stack([np.ones(len(Z)), Z]); r = y - Zc @ b
    d = g.dropna(subset=['dy']); idx = np.where(~(np.isnan(np.column_stack([lag_design(g)[0]])).any(1)))[0]
    # align by date
    ok = ~np.isnan(np.column_stack([g.lp.values] + [g[f'lp_l{k}'].values for k in range(1, J + 1)] + [g.y_anom.shift(1).values])).any(1) & ~g.dy.isna().values
    return pd.Series(r, index=g.date[ok].values)
pairs = [('Rasuwagadhi', 'Devghat', 'upstream→downstream (Trishuli→Narayani)'), ('Bahrabise', 'Chatara', 'upstream→downstream (Sun Koshi→Saptakoshi)'), ('Rasuwagadhi', 'Kusum', 'control: different basins'), ('Bahrabise', 'Chisapani', 'control: different basins')]
rows = []
for a, b_, lab in pairs:
    ra, rb = resid(a), resid(b_); j = pd.concat([ra, rb], axis=1, keys=['a', 'b']).dropna(); r = {}
    for k in range(0, 6):
        x, yv = j.a.values[:len(j) - k], j.b.values[k:]; r[k] = np.corrcoef(x, yv)[0, 1]
        if k == 3:   # bootstrap CI at lag 3 and best lag
            pass
    kb = max(r, key=lambda k: abs(r[k])); x, yv = j.a.values[:len(j) - kb], j.b.values[kb:]
    ci = Mx.boot_ci(lambda p, q: np.corrcoef(p, q)[0, 1], [x, yv], b=10, R=300)
    rows.append(dict(Pair=f'{a} → {b_}', Relation=lab, **{f'r(k={k})': r[k] for k in range(6)}, best_k=kb, ci_lo=ci[0], ci_hi=ci[1]))
SY = pd.DataFrame(rows); SY.to_csv('results/synchrony.csv', index=False); w('synchrony', md(SY, '{:.3f}'))

# ================= 4.9 recession constants (training period, chrono) =================
fitc = F.Fit(df, df.date < '2025-09-01'); rows = []
for s in order:
    g = df[(df.location == s) & (df.date < '2025-09-01')]; z = g.assign(Qn=g[Q].shift(-1), Pn=g[P].shift(-1))
    d = z[(z[P] == 0) & (z.Pn == 0) & (z.Qn <= z[Q]) & (z[Q] > 0.5)]; n = len(d); c = float(np.median(d.Qn / d[Q])) if n else np.nan
    rows.append(dict(Location=s, dry_pairs=n, raw_median_ratio=c, used_c=fitc.c[s], kappa_days=(-1 / np.log(fitc.c[s])) if fitc.c[s] < 1 else np.inf, fallback='pooled' if n < 10 else 'site'))
RC = pd.DataFrame(rows); RC.to_csv('results/recession.csv', index=False); w('recession', md(RC, '{:.3f}'))
import json; json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}, open('results/analyses_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1, default=float))
