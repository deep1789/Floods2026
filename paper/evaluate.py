"""Metrics, statistical tests and tables for the full benchmark (Sec. 6.3-6.4)."""
import numpy as np, pandas as pd, warnings, os
from statsmodels.stats.multitest import multipletests
from floodlab import data, metrics as Mx, events as Ev
warnings.filterwarnings('ignore')
df = data.load(); order = data.site_order(df)
frames = [pd.read_csv('results/preds_tree.csv', parse_dates=['date', 'date_tgt'])]
if os.path.exists('results/preds_lstm.csv'): frames.append(pd.read_csv('results/preds_lstm.csv', parse_dates=['date', 'date_tgt']))
if os.path.exists('results/preds_extra.csv'): frames.append(pd.read_csv('results/preds_extra.csv', parse_dates=['date', 'date_tgt']))
R = pd.concat(frames, ignore_index=True)
R['Q_hat'] = np.clip(np.expm1(R.yhat), 0, None); R['Q_obs'] = np.expm1(R.y_obs)
R['Q_per'] = np.expm1(R.y0)
NEW = ['HGB_tuned', 'LSTM_tuned', 'TFT_lite', 'GRAPH_none', 'GRAPH_phys', 'GRAPH_learned']
MAIN = ['B0_persistence', 'B1_recession', 'B2_climatology', 'B3_ARX', 'HGB', 'LSTM'] + NEW
# tuned models exist only for the chronological split and the 2026 forward-chaining fold: keep them out of the 2-fold 'fc' pool and report 'fc26' separately
R = R[~((R.scheme == 'fc') & R.model.isin(['HGB_tuned', 'LSTM_tuned']))]
R26 = R[(R.scheme == 'fc') & (R.fold.astype(str) == '2026')].copy(); R26['scheme'] = 'fc26'; R26 = pd.concat([R26, pd.concat(frames, ignore_index=True).query("scheme=='fc' and model in ['HGB_tuned','LSTM_tuned'] and fold.astype('str')=='2026'").assign(scheme='fc26')]); R26['Q_hat'] = np.clip(np.expm1(R26.yhat), 0, None); R26['Q_obs'] = np.expm1(R26.y_obs); R26['Q_per'] = np.expm1(R26.y0)
R = pd.concat([R, R26], ignore_index=True)
os.makedirs('tables2', exist_ok=True)

def site_metrics(g):
    o, p = g.Q_obs.values, g.Q_hat.values; yo, yp = g.y_obs.values, g.yhat.values
    return pd.Series(dict(n=len(g), NSE=Mx.nse(o, p), logNSE=Mx.nse(yo, yp), KGE=Mx.kge(o, p), RMSE_log=np.sqrt(np.mean((yo - yp) ** 2))))

base = R[R.model.isin(MAIN)]
M = base.groupby(['scheme', 'h', 'model', 'location']).apply(site_metrics).reset_index()
M.to_csv('results/metrics_site.csv', index=False)

# skill vs persistence, DM test per site, bootstrap CI of delta logNSE
per = R[R.model == 'B0_persistence'].set_index(['scheme', 'fold', 'h', 'location', 'date'])
rec = []
for (sc, h, m, s), g in base[base.model != 'B0_persistence'].groupby(['scheme', 'h', 'model', 'location']):
    gp = per.loc[g.set_index(['scheme', 'fold', 'h', 'location', 'date']).index.intersection(per.index)].reset_index()
    gg = g.set_index(['scheme', 'fold', 'h', 'location', 'date']).loc[gp.set_index(['scheme', 'fold', 'h', 'location', 'date']).index].reset_index()
    if len(gg) < 30: continue
    eA, eB = (gg.y_obs - gg.yhat).values, (gp.y_obs - gp.yhat).values
    st, pv = Mx.dm_test(eA, eB)
    ss = Mx.skill(gg.y_obs.values, gg.yhat.values, gp.yhat.values)
    ci = Mx.boot_ci(lambda o, a, b: Mx.nse(o, a) - Mx.nse(o, b), [gg.y_obs.values, gg.yhat.values, gp.yhat.values], b=10, R=200)
    rec.append(dict(scheme=sc, h=h, model=m, location=s, skill_vs_B0=ss, dlogNSE_lo=ci[0], dlogNSE_hi=ci[1], DM_stat=st, DM_p=pv))
S = pd.DataFrame(rec)
S['q_BH'] = np.nan
for k, g in S.groupby(['scheme']):   # BH across sites x horizons x models within a split scheme
    S.loc[g.index, 'q_BH'] = multipletests(g.DM_p.fillna(1), method='fdr_bh')[1]
S.to_csv('results/skill_tests.csv', index=False)

def md(d, fmt='{:.3f}'):
    d = d.copy(); h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    f = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(f(v) for v in r) + ' |' for r in d.values) + '\n'
w = lambda n, t: open(f'tables2/{n}.md', 'w').write(t)

# ---- main skill table: median across sites per scheme / h / model
def main_table(scheme, hs=(1, 3, 7)):
    rows = []
    for m in MAIN:
        r = {'Model': m}
        for h in hs:
            g = M[(M.scheme == scheme) & (M.h == h) & (M.model == m)]
            if len(g) == 0: r[f'h={h} NSE'] = r[f'h={h} logNSE'] = np.nan; continue
            r[f'h={h} NSE'] = g.NSE.median(); r[f'h={h} logNSE'] = g.logNSE.median()
        g3 = M[(M.scheme == scheme) & (M.h == 3) & (M.model == m)]; r['KGE h=3'] = g3.KGE.median() if len(g3) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
for sc in ('chrono', 'lomo', 'fc', 'fc26'):
    t = main_table(sc); t.to_csv(f'results/main_{sc}.csv', index=False); w(f'main_{sc}', md(t))

# beat-persistence counts
rows = []
for sc in ('chrono', 'lomo', 'fc', 'fc26', 'lolo'):
    for m in ['B1_recession', 'B3_ARX', 'HGB', 'LSTM'] + NEW:
        r = {'Split': sc, 'Model': m}
        for h in (1, 3, 7):
            g = S[(S.scheme == sc) & (S.model == m) & (S.h == h)]
            if len(g) == 0: r[f'h={h}'] = ''; continue
            r[f'h={h}'] = f"{(g.skill_vs_B0 > 0).sum()}/{len(g)} better; {((g.skill_vs_B0 > 0) & (g.q_BH < .05)).sum()} sig.; {((g.skill_vs_B0 < 0) & (g.q_BH < .05)).sum()} sig. worse"
        rows.append(r)
BC = pd.DataFrame(rows); BC = BC[(BC[['h=1', 'h=3', 'h=7']] != '').any(axis=1)]; w('beat_persistence', md(BC))

# ---- per-site detail, chronological
def per_site(scheme, h, models):
    t = pd.DataFrame({'Location': order}).set_index('Location')
    for m in models:
        g = M[(M.scheme == scheme) & (M.h == h) & (M.model == m)].set_index('location')
        t[f'{m} NSE'] = g.NSE; t[f'{m} logNSE'] = g.logNSE
    t.loc['**Median**'] = t.median(); return t.reset_index()
for h in (1, 3, 7): w(f'site_chrono_h{h}', md(per_site('chrono', h, ['B0_persistence', 'B3_ARX', 'HGB', 'LSTM'])))
for h in (1, 3): w(f'site_lomo_h{h}', md(per_site('lomo', h, ['B0_persistence', 'B3_ARX', 'HGB'])))

# ---- by-fold (LOMO and forward chaining), median over sites of logNSE and NSE
rows = []
for sc in ('lomo', 'fc'):
    for fo in sorted(R[R.scheme == sc].fold.unique()):
        for h in (1, 3, 7):
            r = {'Split': sc, 'Held-out monsoon': fo, 'h': h}
            for m in ['B0_persistence', 'B3_ARX', 'HGB', 'LSTM']:
                g = R[(R.scheme == sc) & (R.fold == fo) & (R.h == h) & (R.model == m)]
                if len(g) == 0: continue
                v = g.groupby('location').apply(lambda x: pd.Series(dict(l=Mx.nse(x.y_obs, x.yhat), n=Mx.nse(x.Q_obs, x.Q_hat))))
                r[m + ' logNSE'] = v.l.median(); r[m + ' NSE'] = v.n.median()
            rows.append(r)
BF = pd.DataFrame(rows); BF.to_csv('results/by_fold.csv', index=False); w('by_fold', md(BF[['Split', 'Held-out monsoon', 'h'] + [c for c in BF.columns if 'logNSE' in c]]))

# ---- LSTM seed variability (chrono)
seedrows = []
for sd in range(20):
    nm = f'LSTM_seed{sd}'
    g = R[(R.scheme == 'chrono') & (R.model == nm)]
    if len(g) == 0: break
    for h in (1, 3, 7):
        x = g[g.h == h].groupby('location').apply(lambda z: Mx.nse(z.y_obs, z.yhat)); seedrows.append(dict(seed=sd, h=h, med_logNSE=x.median()))
SD = pd.DataFrame(seedrows)
if len(SD):
    ens = M[(M.scheme == 'chrono') & (M.model == 'LSTM')].groupby('h').logNSE.median()
    t = SD.groupby('h').med_logNSE.agg(['mean', 'std', 'min', 'max']).join(ens.rename('ensemble')).reset_index(); t.columns = ['h', 'seed mean', 'seed sd', 'seed min', 'seed max', 'ensemble (10 seeds)']
    t.to_csv('results/lstm_seeds.csv', index=False); w('lstm_seeds', md(t, '{:.4f}'))

# ---- event metrics (chrono), threshold = training q95 per site
rows = []
for m in ['B0_persistence', 'B3_ARX', 'HGB', 'LSTM', 'HGB_tuned', 'LSTM_tuned', 'TFT_lite', 'GRAPH_phys']:
    for h in (1, 3):
        if len(R[(R.scheme == 'chrono') & (R.model == m) & (R.h == h)]) == 0: continue
        a = dict(hits=0, false_alarms=0, misses=0); ne = nh = 0; te = []; pe = []
        for s, g in R[(R.scheme == 'chrono') & (R.model == m) & (R.h == h)].groupby('location'):
            g = g.sort_values('date'); u = g.thr95.iloc[0]
            c = Mx.contingency((g.Q_obs > u).values, (g.Q_hat > u).values)
            for k in a: a[k] += c[k]
            es = Ev.event_scores(g.Q_obs.values, g.Q_hat.values, u)
            ne += es['n_events']; nh += es['event_hits']
            if not np.isnan(es['timing_mae']): te.append(es['timing_mae']); pe.append(es['peak_rel_err'])
        if a['hits'] + a['misses'] == 0: continue
        pod = a['hits'] / (a['hits'] + a['misses']); far = a['false_alarms'] / (a['hits'] + a['false_alarms']) if a['hits'] + a['false_alarms'] else np.nan
        csi = a['hits'] / (a['hits'] + a['false_alarms'] + a['misses'])
        rows.append({'Model': m, 'h': h, 'day POD': pod, 'day FAR': far, 'day CSI': csi, 'events': ne, 'event POD': nh / ne, 'timing MAE (d)': np.mean(te), 'median peak rel. err': np.median(pe)})
EV = pd.DataFrame(rows); EV.to_csv('results/events.csv', index=False); w('events', md(EV))

# ---- probabilistic (chrono): coverage, width, CRPS approx
rows = []
Rq = R[(R.scheme == 'chrono') & R.model.str.startswith('HGB_q')]
for h in (1, 3):
    g = Rq[Rq.h == h].pivot_table(index=['location', 'date', 'y_obs'], columns='model', values='yhat').reset_index()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]; cols = [f'HGB_q{int(q*100):02d}' for q in qs]
    pl = np.mean([Mx.pinball(g.y_obs, g[c], q) for c, q in zip(cols, qs)])
    cov = ((g.y_obs >= g.HGB_q05) & (g.y_obs <= g.HGB_q95)).mean(); wid = (g.HGB_q95 - g.HGB_q05).mean()
    cov_s = g.assign(ok=(g.y_obs >= g.HGB_q05) & (g.y_obs <= g.HGB_q95)).groupby('location').ok.mean()
    # high-flow subset: observed above site training q95 -> coverage conditional on event days
    ev = R[(R.scheme == 'chrono') & (R.model == 'B0_persistence') & (R.h == h)][['location', 'date', 'thr95', 'Q_obs']]
    g2 = g.merge(ev, on=['location', 'date']); hi = g2.Q_obs > g2.thr95
    cov_hi = ((g2.y_obs >= g2.HGB_q05) & (g2.y_obs <= g2.HGB_q95))[hi].mean()
    rows.append({'h': h, 'mean pinball (5 levels)': pl, 'approx CRPS (=2x mean pinball)': 2 * pl, '90% interval coverage': cov, 'coverage on training-q95 high-flow days': cov_hi, 'mean width (log units)': wid, 'min site coverage': cov_s.min(), 'max site coverage': cov_s.max()})
PR = pd.DataFrame(rows); PR.to_csv('results/prob.csv', index=False); w('prob', md(PR))
pers = R[(R.scheme == 'chrono') & (R.model == 'B0_persistence') & (R.h.isin((1, 3)))]
# graph ablation: sites where phys / learned adjacency beat 'none' (site-level log-NSE), and tuning effect
rows = []
for sc in ('chrono', 'fc', 'fc26'):
    for h in (1, 3, 7):
        g = lambda m: M[(M.scheme == sc) & (M.h == h) & (M.model == m)].set_index('location').logNSE
        n0 = g('GRAPH_none')
        for a, b, lab in (('GRAPH_phys', 'GRAPH_none', 'physical adjacency vs none'), ('GRAPH_learned', 'GRAPH_none', 'learned adjacency vs none'), ('GRAPH_none', 'LSTM', 'graph(no edges) vs LSTM'), ('HGB_tuned', 'HGB', 'tuned vs default HGB'), ('LSTM_tuned', 'LSTM', 'tuned vs default LSTM'), ('TFT_lite', 'LSTM', 'TFT-lite vs LSTM')):
            x, y = g(a), g(b)
            if len(x) == 0 or len(y) == 0: continue
            d = (x - y).dropna(); rows.append({'Split': sc, 'h': h, 'Comparison': lab, 'median Δ log-NSE': d.median(), 'sites better': f'{(d > 0).sum()}/{len(d)}'})
AB = pd.DataFrame(rows); AB.to_csv('results/arm_comparisons.csv', index=False); w('arm_comparisons', md(AB))
print('written tables2:', sorted(os.listdir('tables2')))
