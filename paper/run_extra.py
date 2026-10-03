"""Tuned HGB and LSTM (nested tuning), TFT-lite, graph networks. Writes results/preds_extra.csv. Run AFTER run_tune.py, alone (CPU)."""
import sys, json, time, numpy as np, pandas as pd, torch, warnings
from floodlab import data, features as F, models as M, lstm as L, nets as N, metrics as Mx
warnings.filterwarnings('ignore'); torch.set_num_threads(4)
from floodlab.config import T0, EXTRA_FC, TUNE_YEAR
df = data.load(); HS = L.HZ; sites = data.site_order(df); sidmap = {s: i for i, s in enumerate(sites)}
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10; rows = []
def rec(scheme, fold, name, h, loc, dates, y0, yobs, yhat, thr):
    dt = pd.to_datetime(dates); rows.append(pd.DataFrame(dict(scheme=scheme, fold=fold, h=h, model=name, location=loc, date=dt, date_tgt=dt + pd.Timedelta(days=h), y_obs=yobs, Q_obs=np.expm1(yobs), yhat=yhat, y0=y0,
        thr95=pd.Series(loc).map(thr).values)))
tune = json.load(open('results/tune_hgb.json')); t = time.time()
# ---------------- 1. tuned HGB (chronological and 2026 forward-chaining only: inner tuning used seasons < 2025-09)
LAST = max(EXTRA_FC); vs_ = lambda y: pd.Timestamp(f'{y}-06-01'); ve_ = lambda y: min(pd.Timestamp(f'{y}-09-30'), df.date.max())
for scheme, fold, cut_tr, te_lo, te_hi in (('chrono', 'all', T0, T0, None), ('fc', str(LAST), vs_(LAST) - pd.Timedelta(days=7), vs_(LAST), ve_(LAST))):
    fit = F.Fit(df, df.date < cut_tr); f = F.build(df, fit); f['sid'] = M.sid(f)
    for h in HS:
        p = tune['best'][str(h)]['params']
        tr, te = F.split_rows(f, h, lambda d: d.date_tgt < cut_tr, lambda d: (d.date >= te_lo) & ((d.date <= te_hi) if te_hi is not None else True))
        yh, _ = M.hgb_fit_predict(tr, te, h, params=p)
        rec(scheme, fold, 'HGB_tuned', h, te.location.values, te.date.values, te.y.values, te.y_tgt.values, yh, {s: fit.thr[s][0.95] for s in fit.thr})
print('HGB_tuned', round(time.time() - t), flush=True)
# ---------------- sequence arrays
def seqs(train_mask):
    fit = F.Fit(df, train_mask); f = F.build(df, fit); st = L.norm_stats(f, train_mask); return fit, L.make_seqs(f, st, sidmap)
def mask_targets(T, D, cutoff):
    T = T.copy()
    for k, h in enumerate(HS): T[pd.to_datetime(D) + pd.Timedelta(days=h) >= cutoff, k] = np.nan
    return T
def val_mask(dtr): return (dtr >= dtr.min() + (dtr.max() - dtr.min()) * 0.85)
def emit(scheme, fold, name, fit, D, Nn, Y0, T, P, tem):
    for k, h in enumerate(HS):
        m = ~np.isnan(T[tem][:, k])
        if m.sum(): rec(scheme, fold, name, h, Nn[tem][m], D[tem][m], Y0[tem][m], Y0[tem][m] + T[tem][m][:, k], Y0[tem][m] + P[m, k], {s: fit.thr[s][0.95] for s in fit.thr})
SCH = [('chrono', 'all', T0, T0, None)] + [('fc', str(y), vs_(y) - pd.Timedelta(days=7), vs_(y), ve_(y)) for y in EXTRA_FC]
TUNE_SCH = ('fc', str(TUNE_YEAR), vs_(TUNE_YEAR) - pd.Timedelta(days=7), vs_(TUNE_YEAR), ve_(TUNE_YEAR))
cache = {}; TUNE_KEY = TUNE_SCH[:2] if TUNE_SCH[:2] in [x[:2] for x in SCH] else ('tune', TUNE_SCH[1])
for scheme, fold, cut, lo, hi in SCH + [TUNE_SCH]:
    key = (scheme, fold) if (scheme, fold) != TUNE_SCH[:2] or TUNE_SCH[:2] in [x[:2] for x in SCH] else ('tune', fold)
    fit, (X, S, D, Nn, Y0, T) = seqs((df.date < cut).values); Dd = pd.to_datetime(D)
    trm = Dd < cut; tem = (Dd >= lo) & ((Dd <= hi) if hi is not None else True); cache[key] = (fit, X, S, D, Nn, Y0, T, Dd, trm, tem, cut)
# ---------------- 2. LSTM tuning (inner forward-chaining: train < 2025-05-24, validate Jun-Sep 2025 origins), 12 of 16 configs, 3 seeds
rng = np.random.default_rng(7); grid = [dict(hid=a, drop=b, lr=c, wd=d) for a in (32, 64) for b in (0.2, 0.4) for c in (1e-3, 3e-3) for d in (1e-3, 1e-2)]
cfgs = [dict(hid=48, drop=0.3, lr=2e-3, wd=1e-3)] + [grid[i] for i in rng.permutation(len(grid))[:11]]
fit, X, S, D, Nn, Y0, T, Dd, trm, tem, cut = cache[TUNE_KEY]; Ttr = mask_targets(T, D, cut)[trm]; val = val_mask(Dd[trm]); sc = []
for ci, c in enumerate(cfgs):
    P = np.mean([L.fit_predict(X[trm], S[trm], Ttr, X[tem], S[tem], len(sites), sd, 60, True, val=val, **c) for sd in range(3)], 0)
    ok = ~np.isnan(T[tem]); s = []
    for k in range(3):
        m = ok[:, k]; g = pd.DataFrame(dict(n=Nn[tem][m], o=Y0[tem][m] + T[tem][m][:, k], p=Y0[tem][m] + P[m, k])); s.append(g.groupby('n').apply(lambda x: Mx.nse(x.o, x.p)).median())
    sc.append(float(np.mean(s))); print('lstm cfg', ci, c, round(sc[-1], 4), round(time.time() - t), flush=True)
best = cfgs[int(np.argmax(sc))]; json.dump(dict(configs=cfgs, scores=sc, best=best), open('results/tune_lstm.json', 'w'), indent=1); print('LSTM best', best, flush=True)
# ---------------- 3. final fits: tuned LSTM (chrono, fc 2026), TFT-lite (chrono, fc 2025, fc 2026), graph nets (same)
tft_w = []
for scheme, fold, *_ in SCH:
    fit, X, S, D, Nn, Y0, T, Dd, trm, tem, cut = cache[(scheme, fold)]; Ttr = mask_targets(T, D, cut)[trm]; val = val_mask(Dd[trm])
    if scheme == 'chrono' or fold == str(LAST):
        P = np.mean([L.fit_predict(X[trm], S[trm], Ttr, X[tem], S[tem], len(sites), sd, 60, True, val=val, **best) for sd in range(NS)], 0); emit(scheme, fold, 'LSTM_tuned', fit, D, Nn, Y0, T, P, tem); print('LSTM_tuned', scheme, fold, round(time.time() - t), flush=True)
    ps = []
    for sd in range(NS):
        p, m = N.fit_predict_tft(X[trm], S[trm], Ttr, X[tem], S[tem], len(sites), sd, val); ps.append(p)
        if scheme == 'chrono':
            with torch.no_grad(): m(torch.tensor(X[tem][:512]), torch.tensor(S[tem][:512])); tft_w.append(m.last_w.mean((0, 1)).numpy())
    emit(scheme, fold, 'TFT_lite', fit, D, Nn, Y0, T, np.mean(ps, 0), tem); print('TFT', scheme, fold, round(time.time() - t), flush=True)
    # graph: arrange (date, node)
    dates = np.array(sorted(set(Dd))); di = {d: i for i, d in enumerate(dates)}; G = np.zeros((len(dates), 10) + X.shape[1:], np.float32); GT = np.full((len(dates), 10, 3), np.nan, np.float32); GY0 = np.zeros((len(dates), 10)); present = np.zeros((len(dates), 10), bool)
    for i in range(len(X)): j = di[Dd[i]]; G[j, S[i]] = X[i]; GT[j, S[i]] = T[i]; GY0[j, S[i]] = Y0[i]; present[j, S[i]] = True
    gtr = dates < cut; gte = (dates >= (cut if scheme == 'chrono' else pd.Timestamp(f'{fold}-06-01'))) & (dates <= (pd.Timestamp('2100-01-01') if scheme == 'chrono' else pd.Timestamp(f'{fold}-09-30')))
    GTm = GT.copy()
    for k, h in enumerate(HS): GTm[dates + pd.Timedelta(days=h) >= cut, :, k] = np.nan
    gval = val_mask(pd.DatetimeIndex(dates[gtr]))
    for adj in ('none', 'phys', 'learned'):
        ps = [N.fit_predict_graph(G[gtr], GTm[gtr], G[gte], 10, sd, gval, adj)[0] for sd in range(NS)]; P = np.mean(ps, 0)       # (Dte,10,3)
        for k, h in enumerate(HS):
            for node, s in enumerate(sites):
                ok = ~np.isnan(GT[gte][:, node, k]) & present[gte][:, node]
                if ok.sum(): rec(scheme, fold, f'GRAPH_{adj}', h, np.array([s] * ok.sum()), dates[gte][ok], GY0[gte][ok, node], GY0[gte][ok, node] + GT[gte][ok, node, k], GY0[gte][ok, node] + P[ok, node, k], {s_: fit.thr[s_][0.95] for s_ in fit.thr})
        print('GRAPH', adj, scheme, fold, round(time.time() - t), flush=True)
    pd.concat(rows).to_csv('results/preds_extra.csv', index=False)
pd.DataFrame(dict(feature=L.DYN, mean_selection_weight=np.mean(tft_w, 0))).to_csv('results/tft_selection_weights.csv', index=False)
print('done', round(time.time() - t))
