"""Joint multi-site LSTM (Sec. 5.5). Seeds: 10 for chronological and forward-chaining, SEEDS_LOLO for leave-one-location-out."""
import sys, time, numpy as np, pandas as pd, torch
from floodlab import data, features as F, models as M, lstm as L
torch.set_num_threads(4)
df = data.load(); T0 = pd.Timestamp('2025-09-01'); HS = L.HZ
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10; NSL = int(sys.argv[2]) if len(sys.argv) > 2 else 3; EP = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sites = data.site_order(df); sidmap = {s: i for i, s in enumerate(sites)}; rows = []
def seqs_for(train_mask):
    fit = F.Fit(df, train_mask); f = F.build(df, fit); stats = L.norm_stats(f, train_mask)
    return fit, L.make_seqs(f, stats, sidmap)
def record(scheme, fold, name, X, fit, dates, names, y0, T, pred, hmask):
    for k, h in enumerate(HS):
        m = hmask & ~np.isnan(T[:, k])
        if m.sum() == 0: continue
        dt = pd.to_datetime(dates[m]); y_obs = y0[m] + T[m, k]
        rows.append(pd.DataFrame(dict(scheme=scheme, fold=fold, h=h, model=name, location=names[m], date=dt, date_tgt=dt + pd.Timedelta(days=h),
            y_obs=y_obs, Q_obs=np.expm1(y_obs), yhat=y0[m] + pred[m, k], y0=y0[m],
            thr95=pd.Series(names[m]).map({s: fit.thr[s][0.95] for s in fit.thr}).values)))
def mask_targets(T, dates, cutoff):
    T = T.copy()
    for k, h in enumerate(HS): T[pd.to_datetime(dates) + pd.Timedelta(days=h) >= cutoff, k] = np.nan
    return T
def ens(scheme, fold, tr_mask_dates_fn, te_fn, nseed, use_site=True, tag='LSTM', trmask=None, train_cutoff=None, keep_seed_rows=False, lolo_site=None):
    fit, (X, S, D, N, Y0, T) = seqs_for(trmask)
    Dd = pd.to_datetime(D)
    trm = tr_mask_dates_fn(Dd, N); tem = te_fn(Dd, N)
    Ttr = mask_targets(T, D, train_cutoff)[trm]
    dtr = Dd[trm]; val = (dtr >= dtr.min() + (dtr.max() - dtr.min()) * 0.85)
    print(scheme, fold, 'train', trm.sum(), 'val', val.sum(), 'val dates', dtr[val].min().date(), '->', dtr[val].max().date(), flush=True)
    preds = []
    for sd in range(nseed):
        t = time.time(); p = L.fit_predict(X[trm], S[trm], Ttr, X[tem], S[tem], len(sites), sd, EP, use_site, val=val)
        preds.append(p); print(scheme, fold, 'seed', sd, round(time.time() - t), 's', flush=True)
    P = np.mean(preds, 0)
    sub = lambda a: a[tem]
    record(scheme, fold, tag, X[tem], fit, sub(D), sub(N), sub(Y0), sub(T), P, np.ones(tem.sum(), bool))
    if keep_seed_rows:
        for sd, p in enumerate(preds): record(scheme, fold, f'{tag}_seed{sd}', X[tem], fit, sub(D), sub(N), sub(Y0), sub(T), p, np.ones(tem.sum(), bool))
t = time.time()
# chronological
tm = (df.date < T0)
ens('chrono', 'all', lambda d, n: d < T0, lambda d, n: d >= T0, NS, trmask=tm.values, train_cutoff=T0, keep_seed_rows=True)
pd.concat(rows).to_csv('results/preds_lstm.csv', index=False)
# forward chaining
for yr in (2025, 2026):
    vs, ve = pd.Timestamp(f'{yr}-06-01'), min(pd.Timestamp(f'{yr}-09-30'), df.date.max()); cut = vs - pd.Timedelta(days=7)
    ens('fc', str(yr), lambda d, n, cut=cut: d < cut, lambda d, n, vs=vs, ve=ve: (d >= vs) & (d <= ve), NS, trmask=(df.date < cut).values, train_cutoff=cut)
    pd.concat(rows).to_csv('results/preds_lstm.csv', index=False)
# LOLO (no site embedding)
for s in sites:
    ens('lolo', s, lambda d, n, s=s: (d < T0) & (n != s), lambda d, n, s=s: (d >= T0) & (n == s), NSL, use_site=False, trmask=tm.values, train_cutoff=T0)
    pd.concat(rows).to_csv('results/preds_lstm.csv', index=False)
print('done', round(time.time() - t))
