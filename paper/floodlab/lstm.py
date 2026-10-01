"""Joint LSTM (Sec. 5.5): one set of weights for all sites, per-site input normalisation, multi-horizon increment heads."""
import numpy as np, torch, torch.nn as nn
from .data import Q, P, TH
SEQ = 30; HZ = (1, 3, 7)
DYN = ['y', 'lp', 'th_star', 'temperature_mean_c', 'relative_humidity_mean_pct', 'dew_point_mean_c', 'wind_speed_max_kmh',
       'precipitation_hours', 'snowfrac', 'sin_doy', 'cos_doy', 'dy']

class Net(nn.Module):
    def __init__(s, nin, nsite, hid=48, emb=4, drop=0.3):
        super().__init__(); s.emb = nn.Embedding(nsite, emb) if nsite else None
        s.lstm = nn.LSTM(nin + (emb if nsite else 0), hid, batch_first=True); s.drop = nn.Dropout(drop); s.head = nn.Linear(hid, len(HZ))
    def forward(s, x, sid=None):
        if s.emb is not None: x = torch.cat([x, s.emb(sid).unsqueeze(1).expand(-1, x.shape[1], -1)], 2)
        o, _ = s.lstm(x); return s.head(s.drop(o[:, -1]))

def make_seqs(feat, stats, sidmap):
    """Return arrays X (N,SEQ,F), site ids, origin dates, site names, y0 and targets (N,3) of y_{t+h}-y_t (NaN where unavailable)."""
    X, S, D, N, Y0, T = [], [], [], [], [], []
    for s, g in feat.groupby('location', sort=False):
        g = g.sort_values('date'); mu, sd = stats[s]
        A = ((g[DYN].values - mu) / sd).astype(np.float32); y = g.y.values
        for i in range(SEQ - 1, len(g)):
            w = A[i - SEQ + 1:i + 1]
            if np.isnan(w).any(): continue
            X.append(w); S.append(sidmap[s]); D.append(g.date.values[i]); N.append(s); Y0.append(y[i])
            T.append([y[i + h] - y[i] if i + h < len(g) else np.nan for h in HZ])
    return np.array(X), np.array(S), np.array(D), np.array(N), np.array(Y0), np.array(T, dtype=np.float32)

def norm_stats(feat, train_mask):
    st = {}
    for s, g in feat[train_mask].groupby('location'):
        v = g[DYN].values; st[s] = (np.nanmean(v, 0), np.nanstd(v, 0) + 1e-6)
    return st

def fit_predict(Xtr, Str, Ttr, Xte, Ste, nsite, seed, epochs=60, use_site=True, device='cpu', val=None, hid=48, drop=0.3, lr=2e-3, wd=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(Xtr.shape[2], nsite if use_site else 0, hid=hid, drop=drop).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    # early stopping on the most recent 15% of training DATES (val = boolean mask over training rows, chronological), never on test data
    if val is None: raise ValueError('pass a chronological validation mask')
    tr_i, va_i = np.where(~val)[0], np.where(val)[0]
    Xt, St, Tt = [torch.tensor(a) for a in (Xtr, Str, Ttr)]
    best, bs, bad = 1e9, None, 0
    msk = ~torch.isnan(Tt)
    for ep in range(epochs):
        m.train(); idx = torch.tensor(np.random.permutation(tr_i))
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; opt.zero_grad(); p = m(Xt[j], St[j] if use_site else None)
            loss = (((p - torch.nan_to_num(Tt[j])) ** 2) * msk[j]).sum() / msk[j].sum(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(Xt[va_i], St[va_i] if use_site else None); v = (((p - torch.nan_to_num(Tt[va_i])) ** 2) * msk[va_i]).sum() / msk[va_i].sum()
        if v < best - 1e-5: best, bs, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return m(torch.tensor(Xte), torch.tensor(Ste) if use_site else None).numpy()
