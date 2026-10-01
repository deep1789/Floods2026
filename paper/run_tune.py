"""Bounded hyperparameter search, nested so the final test period is never touched.
Inner folds: the 2023, 2024 and 2025 monsoons inside the TRAINING period (< 2025-09-01), each held out in turn with 7-day pre / 30-day post purge.
Criterion: median across sites of log-NSE, averaged over the folds. 30 random configurations + the default."""
import sys, json, time, itertools, numpy as np, pandas as pd, warnings
from floodlab import data, features as F, models as M, metrics as Mx
warnings.filterwarnings('ignore')
df = data.load(); T0 = pd.Timestamp('2025-09-01'); rng = np.random.default_rng(42)
SPACE = dict(learning_rate=[0.02, 0.05, 0.1], max_leaf_nodes=[7, 15, 31], min_samples_leaf=[10, 20, 50], l2_regularization=[0.0, 1.0, 10.0], max_iter=[200, 400, 600])
cands = [M.hgb_params()] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} | {'random_state': 0} for _ in range(30)]
folds = []
for yr in (2023, 2024, 2025):
    vs, ve = pd.Timestamp(f'{yr}-06-01'), pd.Timestamp(f'{yr}-09-30'); ex = (df.date >= vs - pd.Timedelta(days=7)) & (df.date <= ve + pd.Timedelta(days=30)) | (df.date >= T0)
    fit = F.Fit(df, ~ex); f = F.build(df, fit); f['sid'] = M.sid(f); folds.append((vs, ve, f))
res = {h: np.zeros((len(cands), len(folds))) for h in (1, 3, 7)}; t = time.time()
for h in (1, 3, 7):
    data_h = []
    for vs, ve, f in folds:
        tr, te = F.split_rows(f, h, lambda d, vs=vs, ve=ve: ~(((d.date >= vs - pd.Timedelta(days=7)) & (d.date <= ve + pd.Timedelta(days=30))) | (d.date_tgt >= T0)), lambda d, vs=vs, ve=ve: (d.date >= vs) & (d.date <= ve))
        data_h.append((tr, te))
    for ci, p in enumerate(cands):
        for fi, (tr, te) in enumerate(data_h):
            yh, _ = M.hgb_fit_predict(tr, te, h, params=p)
            res[h][ci, fi] = te.assign(yh=yh).groupby('location').apply(lambda x: Mx.nse(x.y_tgt, x.yh)).median()
    print('h', h, 'done', round(time.time() - t), flush=True)
out = {'candidates': cands, 'inner_scores': {h: res[h].tolist() for h in res}, 'best': {}}
for h in res:
    m = res[h].mean(1); b = int(np.argmax(m)); out['best'][h] = dict(index=b, params=cands[b], inner_mean=float(m[b]), default_inner_mean=float(m[0]), spread=[float(m.min()), float(np.median(m)), float(m.max())])
    print(h, 'best idx', b, cands[b], 'inner mean %.4f (default %.4f; min/med/max %.4f/%.4f/%.4f)' % (m[b], m[0], m.min(), np.median(m), m.max()))
json.dump(out, open('results/tune_hgb.json', 'w'), indent=1, default=int)
