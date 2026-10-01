"""Full benchmark (paper Sec. 6): chronological, leave-one-monsoon-out (LOMO) and leave-one-location-out (LOLO) splits.
Writes results/preds_tree.csv (long format: one row per scheme/fold/h/model/site/origin-date)."""
import sys, time, numpy as np, pandas as pd
from floodlab import data, features as F, models as M
df = data.load(); T0 = pd.Timestamp('2025-09-01'); HS = (1, 3, 7); QS = (0.05, 0.25, 0.5, 0.75, 0.95); rows = []

def prep(train_mask):
    fit = F.Fit(df, train_mask); f = F.build(df, fit); f['sid'] = M.sid(f); return fit, f

def record(scheme, fold, h, name, te, yhat, fit):
    thr = te.location.map({s: fit.thr[s][0.95] for s in fit.thr}).values
    rows.append(pd.DataFrame(dict(scheme=scheme, fold=fold, h=h, model=name, location=te.location.values, date=te.date.values,
        date_tgt=te.date_tgt.values, y_obs=te.y_tgt.values, Q_obs=te.Q_tgt.values, yhat=yhat, y0=te.y.values, thr95=thr)))

def run(scheme, fold, fit, f, tr_fn, te_fn, purge=None, site_models=True, tag_site=True, models=('B0', 'B1', 'B2', 'B3', 'HGB', 'HGBq'), hs=HS):
    for h in hs:
        tr, te = F.split_rows(f, h, tr_fn, te_fn, purge)
        if len(te) == 0: continue
        if 'B0' in models: record(scheme, fold, h, 'B0_persistence', te, M.b0_persistence(te, h), fit)
        if 'B1' in models: record(scheme, fold, h, 'B1_recession', te, M.b1_recession(te, h, fit), fit)
        if 'B2' in models: record(scheme, fold, h, 'B2_climatology', te, M.b2_climatology(te, h, fit), fit)
        if 'B3' in models: record(scheme, fold, h, 'B3_ARX', te, M.b3_arx(tr, te, h, fit), fit)
        if 'HGB' in models:
            yh, _ = M.hgb_fit_predict(tr, te, h, use_site=tag_site, use_elev=tag_site); record(scheme, fold, h, 'HGB', te, yh, fit)
        if 'HGBq' in models and h in (1, 3):
            qp = []
            for q in QS:
                p, _ = M.hgb_fit_predict(tr, te, h, use_site=tag_site, use_elev=tag_site, loss='quantile', q=q); qp.append(p)
            qp = np.sort(np.column_stack(qp), 1)
            for i, q in enumerate(QS): record(scheme, fold, h, f'HGB_q{int(q*100):02d}', te, qp[:, i], fit)

t = time.time()
# ---- 1. chronological
tm = df.date < T0; fit, f = prep(tm)
run('chrono', 'all', fit, f, lambda d: d.date_tgt < T0, lambda d: d.date >= T0)
print('chrono done', round(time.time() - t), flush=True)
# ---- 2. leave-one-monsoon-out with purge (target within h days before block, features 30 days after)
for yr in (2023, 2024, 2025, 2026):
    vs, ve = pd.Timestamp(f'{yr}-06-01'), min(pd.Timestamp(f'{yr}-09-30'), df.date.max())
    for h in HS:
        pass
    ex = (df.date >= vs - pd.Timedelta(days=7)) & (df.date <= ve + pd.Timedelta(days=30))
    fit, f = prep(~ex)
    run('lomo', str(yr), fit, f,
        lambda d, vs=vs, ve=ve: ~((d.date >= vs - pd.Timedelta(days=7)) & (d.date <= ve + pd.Timedelta(days=30))),
        lambda d, vs=vs, ve=ve: (d.date >= vs) & (d.date <= ve), models=('B0', 'B1', 'B2', 'B3', 'HGB'))
    print('lomo', yr, round(time.time() - t), flush=True)
# ---- 2b. forward chaining (train strictly before the held-out monsoon; comparable with the LSTM)
for yr in (2025, 2026):
    vs, ve = pd.Timestamp(f'{yr}-06-01'), min(pd.Timestamp(f'{yr}-09-30'), df.date.max()); cut = vs - pd.Timedelta(days=7)
    fit, f = prep(df.date < cut)
    run('fc', str(yr), fit, f, lambda d, cut=cut: d.date_tgt < cut, lambda d, vs=vs, ve=ve: (d.date >= vs) & (d.date <= ve), models=('B0', 'B1', 'B2', 'B3', 'HGB'))
    print('fc', yr, flush=True)
# ---- 3. leave-one-location-out (chronological variant): model trained on 9 sites, applied to the 10th
fit, f = prep(tm)
for s in data.site_order(df):
    run('lolo', s, fit, f, lambda d, s=s: (d.location != s) & (d.date_tgt < T0), lambda d, s=s: (d.location == s) & (d.date >= T0),
        models=('B0', 'HGB'), tag_site=False)
    print('lolo', s, round(time.time() - t), flush=True)
out = pd.concat(rows, ignore_index=True); out.to_csv('results/preds_tree.csv', index=False); print(out.shape)
