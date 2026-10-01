"""LOLO table + regime similarity, residual-based anomaly check on documented events, figures."""
import numpy as np, pandas as pd, warnings, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from floodlab import data, metrics as Mx
warnings.filterwarnings('ignore')
df = data.load(); order = data.site_order(df); Q, P = data.Q, data.P
M = pd.read_csv('results/metrics_site.csv'); S = pd.read_csv('results/skill_tests.csv')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
w = lambda n, t: open(f'tables2/{n}.md', 'w').write(t)
# ------------- regime descriptors from the training window (< 2025-09-01), standardised across sites
tr = df[df.date < '2025-09-01']; rows = {}
for s, g in tr.groupby('location'):
    q = g[Q].values; med = max(np.median(q), 1e-3); y = np.log1p(q)
    rows[s] = dict(fdc5=np.log10((np.quantile(q, .05) + .01) / med), fdc25=np.log10((np.quantile(q, .25) + .01) / med), fdc75=np.log10(np.quantile(q, .75) / med),
                   fdc95=np.log10(np.quantile(q, .95) / med), monsoonP=g[g.date.dt.month.between(6, 9)][P].sum() / g[P].sum(), ac1=pd.Series(y).autocorr(1), sd_dy=np.std(np.diff(y)))
Rg = pd.DataFrame(rows).T.loc[order]; Z = (Rg - Rg.mean()) / Rg.std(); D = pd.DataFrame(np.sqrt(((Z.values[:, None] - Z.values[None]) ** 2).sum(-1)), index=order, columns=order)
dist = D.sum(1) / (len(order) - 1)
def get(sc, h, m): return M[(M.scheme == sc) & (M.h == h) & (M.model == m)].set_index('location')
tab = []
for h in (1, 3):
    wi, lo, pe, ll = get('chrono', h, 'HGB'), get('lolo', h, 'HGB'), get('chrono', h, 'B0_persistence'), get('lolo', h, 'LSTM')
    for s in order: tab.append(dict(h=h, Location=s, persist=pe.loc[s, 'logNSE'], within=wi.loc[s, 'logNSE'], lolo_hgb=lo.loc[s, 'logNSE'], lolo_lstm=ll.loc[s, 'logNSE'], change_hgb=lo.loc[s, 'logNSE'] - wi.loc[s, 'logNSE'],
                    change_lstm=ll.loc[s, 'logNSE'] - wi.loc[s, 'logNSE'], dist=dist[s], nse_persist=pe.loc[s, 'NSE'], nse_within=wi.loc[s, 'NSE'], nse_lolo_hgb=lo.loc[s, 'NSE'], nse_lolo_lstm=ll.loc[s, 'NSE']))
T = pd.DataFrame(tab); T.to_csv('results/lolo_table.csv', index=False)
for h in (1, 3):
    t = T[T.h == h][['Location', 'persist', 'within', 'lolo_hgb', 'lolo_lstm', 'change_hgb', 'dist']].copy()
    t.columns = ['Location', 'Persistence log-NSE', 'HGB within-site log-NSE', 'HGB LOLO log-NSE', 'LSTM LOLO log-NSE', 'Δ HGB (LOLO − within)', 'Regime distance']
    t.loc[len(t)] = ['**Median**'] + list(t.iloc[:, 1:].median()); w(f'lolo_h{h}', md(t))
corr = {}
for h in (1, 3):
    t = T[T.h == h]
    for c in ('change_hgb', 'change_lstm'):
        r, p = stats.spearmanr(t['dist'], t[c]); rs = []
        rng = np.random.default_rng(0)
        for _ in range(2000): ix = rng.integers(0, 10, 10); rs.append(stats.spearmanr(t['dist'].values[ix], t[c].values[ix])[0])
        corr[(h, c)] = (r, p, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5))
CR = pd.DataFrame([dict(h=k[0], model=k[1].replace('change_', ''), rho=v[0], p=v[1], ci_lo=v[2], ci_hi=v[3]) for k, v in corr.items()]); w('lolo_corr', md(CR)); CR.to_csv('results/lolo_corr.csv', index=False)
Rg.round(2).assign(dist=dist.round(2)).reset_index().rename(columns={'index': 'Location'}).pipe(lambda d: w('regime', md(d, '{:.2f}')))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, h in zip(axs, (1, 3)):
    t = T[T.h == h]; ax.scatter(t.dist, t.change_hgb, label='HGB', c='C0'); ax.scatter(t.dist, t.change_lstm, label='LSTM (5 seeds avg)' if False else 'LSTM', c='C3', marker='s')
    for _, r in t.iterrows(): ax.annotate(r.Location, (r.dist, r.change_hgb), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, c='k', lw=.5); ax.set_xlabel('regime distance to other sites'); ax.set_ylabel('Δ log-NSE vs within-site HGB'); ax.set_title(f'h = {h} d'); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('figures/fig11_lolo.png', dpi=160); plt.close()

# ------------- skill by site figure (chronological and LOMO, h = 3)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, sc in zip(axs, ('chrono', 'lomo')):
    d = S[(S.scheme == sc) & (S.h == 3)]
    for i, m in enumerate(['B3_ARX', 'HGB', 'LSTM']):
        g = d[d.model == m].set_index('location').reindex(order[::-1])
        if g.skill_vs_B0.isna().all(): continue
        y = np.arange(len(order)) + (i - 1) * 0.22
        ax.errorbar(g.skill_vs_B0.values, y, fmt='o', ms=3, label=m, capsize=0)
    ax.axvline(0, c='k', lw=.6); ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1], fontsize=7); ax.set_xlabel('MSE skill vs persistence (log space), h = 3'); ax.set_title({'chrono': 'chronological split', 'lomo': 'leave-one-monsoon-out'}[sc]); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('figures/fig12_skill_sites.png', dpi=160); plt.close()

# ------------- prediction intervals (chrono h=1) for two contrasting sites
R = pd.read_csv('results/preds_tree.csv', parse_dates=['date', 'date_tgt']); R = R[(R.scheme == 'chrono') & (R.h == 1)]
fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
for ax, s in zip(axs, ('Khokana', 'Devghat')):
    g = {m: R[(R.location == s) & (R.model == m)].set_index('date_tgt').sort_index() for m in ('HGB_q05', 'HGB_q50', 'HGB_q95')}
    obs = g['HGB_q50'].y_obs; ax.fill_between(obs.index, np.expm1(g['HGB_q05'].yhat), np.expm1(g['HGB_q95'].yhat), alpha=.3, label='5-95% interval'); ax.plot(obs.index, np.expm1(g['HGB_q50'].yhat), lw=.8, label='median')
    ax.plot(obs.index, np.expm1(obs), 'k', lw=.6, label='observed (modelled Q)'); ax.set_yscale('log'); ax.set_title(s + ', h = 1 d, test window'); ax.set_ylabel('Q (m³/s)'); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('figures/fig13_intervals.png', dpi=160); plt.close()

# ------------- residual-based anomaly check on documented events (LOMO out-of-fold, monsoon only, h = 1)
Rl = pd.read_csv('results/preds_tree.csv', parse_dates=['date', 'date_tgt']); Rl = Rl[(Rl.scheme == 'lomo') & (Rl.h == 1) & Rl.model.isin(['B3_ARX', 'HGB'])]
Rl['res'] = Rl.y_obs - Rl.yhat; Rl['z'] = Rl.groupby(['model', 'location']).res.transform(lambda x: (x - x.mean()) / x.std())
ev = [('Kusum', '2024-09-28', 'rain-driven flood (positive)', 'Sep-2024 storm'), ('Khokana', '2024-09-28', 'rain-driven flood (positive)', 'Sep-2024 storm'), ('Devghat', '2024-09-29', 'rain-driven flood (positive)', 'Sep-2024 storm'),
      ('Chatara', '2024-08-16', 'GLOF, nearest location (negative control)', 'Thame GLOF'), ('Bahrabise', '2024-08-16', 'GLOF, nearest Koshi site (negative control)', 'Thame GLOF'), ('Rasuwagadhi', '2025-07-08', 'flash flood (negative control)', 'Bhote Koshi flash flood')]
rows = []
for s, d, lab, ename in ev:
    for m in ('B3_ARX', 'HGB'):
        g = Rl[(Rl.location == s) & (Rl.model == m)]; t = pd.Timestamp(d); win = g[(g.date_tgt >= t - pd.Timedelta(days=1)) & (g.date_tgt <= t + pd.Timedelta(days=1))]
        if len(win) == 0: continue
        zmax = win.z.max(); pct = (g.z < zmax).mean() * 100
        rows.append(dict(Event=ename, Location=s, Date=d, Role=lab, Model=m, max_z_pm1d=zmax, percentile=pct, n_oof=len(g)))
AN = pd.DataFrame(rows); AN.to_csv('results/anomaly_events.csv', index=False)
w('anomaly', md(AN[['Event', 'Location', 'Date', 'Role', 'Model', 'max_z_pm1d', 'percentile']].rename(columns={'max_z_pm1d': 'max standardised residual (±1 d)', 'percentile': 'percentile within site monsoon OOF residuals'}), '{:.2f}'))
# base rate: fraction of monsoon days with z above 2 and 3 for reference
z2 = (Rl[Rl.model == 'HGB'].z > 2).mean(); z3 = (Rl[Rl.model == 'HGB'].z > 3).mean()
print('base rates HGB z>2: %.3f, z>3: %.3f' % (z2, z3)); print(AN.round(2).to_string()); print(CR.round(3).to_string()); print(dist.round(2).to_string())
