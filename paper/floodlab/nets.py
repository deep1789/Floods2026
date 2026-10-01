"""Additional neural arms: a simplified Temporal-Fusion-style network ('TFT-lite') and a multi-site graph network.
These are NOT the reference TFT implementation (no quantile heads, no known-future inputs): variable-selection gating (GRN),
an LSTM encoder, one self-attention layer over the window, and a gated output head."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from .lstm import HZ, SEQ

class GRN(nn.Module):
    def __init__(s, d_in, d, d_out=None, drop=0.1):
        super().__init__(); d_out = d_out or d
        s.f1 = nn.Linear(d_in, d); s.f2 = nn.Linear(d, d_out); s.gate = nn.Linear(d, 2 * d_out); s.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity(); s.ln = nn.LayerNorm(d_out); s.drop = nn.Dropout(drop)
    def forward(s, x):
        h = s.f2(Fn.elu(s.f1(x))); h = s.drop(h); a, b = s.gate(Fn.elu(s.f1(x))).chunk(2, -1)
        return s.ln(s.skip(x) + torch.sigmoid(a) * b + 0 * h)

class TFTLite(nn.Module):
    def __init__(s, nin, nsite, d=32, heads=2, drop=0.2):
        super().__init__(); s.nin = nin; s.emb = nn.ModuleList([nn.Linear(1, d) for _ in range(nin)]); s.sel = GRN(nin, d, nin, drop)
        s.site = nn.Embedding(nsite, d); s.lstm = nn.LSTM(d, d, batch_first=True); s.att = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        s.ln = nn.LayerNorm(d); s.post = GRN(d, d, d, drop); s.head = nn.Linear(d, len(HZ)); s.last_w = None
    def forward(s, x, sid):
        w = torch.softmax(s.sel(x), -1); s.last_w = w.detach()                       # (B,T,F) variable-selection weights
        e = torch.stack([s.emb[j](x[..., j:j + 1]) for j in range(s.nin)], -2)       # (B,T,F,d)
        v = (w.unsqueeze(-1) * e).sum(-2) + s.site(sid).unsqueeze(1)
        o, _ = s.lstm(v); a, _ = s.att(o, o, o, need_weights=False); o = s.ln(o + a)
        return s.head(s.post(o[:, -1]))

class GraphNet(nn.Module):
    """Shared LSTM encoder per node -> one graph-convolution layer -> per-node head. adj in {'none','phys','learned'}."""
    def __init__(s, nin, nsite, adj='phys', hid=48, drop=0.3, edges=((7, 4), (8, 0))):
        super().__init__(); s.adj = adj; s.n = nsite
        s.enc = nn.LSTM(nin, hid, batch_first=True); s.site = nn.Embedding(nsite, hid); s.gc = nn.Linear(hid, hid); s.self_ = nn.Linear(hid, hid)
        s.drop = nn.Dropout(drop); s.head = nn.Linear(hid, len(HZ)); s.E = nn.Parameter(torch.randn(nsite, 8) * 0.1)
        A = torch.eye(nsite)
        for u, dn in edges: A[dn, u] = 1.0                                           # downstream node receives from upstream node
        s.register_buffer('A', A / A.sum(1, keepdim=True))
    def forward(s, x, sid=None):                                                      # x: (B, N, T, F)
        B, N, T, F_ = x.shape; o, _ = s.enc(x.reshape(B * N, T, F_)); h = o[:, -1].reshape(B, N, -1) + s.site.weight.unsqueeze(0)
        if s.adj == 'none': msg = torch.zeros_like(h)
        else:
            A = s.A if s.adj == 'phys' else torch.softmax(s.E @ s.E.T, -1)
            msg = s.gc(torch.einsum('ij,bjh->bih', A, h))
        h = Fn.relu(s.self_(h) + msg); return s.head(s.drop(h))                      # (B,N,3)

def _loop(m, step_loss, n_tr, val_idx, tr_idx, epochs, lr, wd, bs, seed):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); best, bs_, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tr_idx)
        for b in range(0, len(perm), bs):
            j = torch.tensor(perm[b:b + bs]); opt.zero_grad(); loss = step_loss(j); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): v = step_loss(torch.tensor(val_idx))
        if v < best - 1e-5: best, bs_, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    m.load_state_dict(bs_); m.eval(); return m

def fit_predict_tft(Xtr, Str, Ttr, Xte, Ste, nsite, seed, val, epochs=60, lr=2e-3, wd=1e-3, d=32):
    torch.manual_seed(seed); np.random.seed(seed); m = TFTLite(Xtr.shape[2], nsite, d=d)
    Xt, St, Tt = [torch.tensor(a) for a in (Xtr, Str, Ttr)]; msk = ~torch.isnan(Tt); Tz = torch.nan_to_num(Tt)
    def loss(j): p = m(Xt[j], St[j]); return (((p - Tz[j]) ** 2) * msk[j]).sum() / msk[j].sum()
    _loop(m, loss, len(Xtr), np.where(val)[0], np.where(~val)[0], epochs, lr, wd, 128, seed)
    with torch.no_grad(): p = m(torch.tensor(Xte), torch.tensor(Ste)).numpy()
    return p, m

def fit_predict_graph(Xtr, Ttr, Xte, nsite, seed, val, adj='phys', epochs=80, lr=2e-3, wd=1e-3):
    """Xtr: (D,N,T,F), Ttr: (D,N,3). One sample = one date with all nodes."""
    torch.manual_seed(seed); np.random.seed(seed); m = GraphNet(Xtr.shape[3], nsite, adj)
    Xt, Tt = torch.tensor(Xtr), torch.tensor(Ttr); msk = ~torch.isnan(Tt); Tz = torch.nan_to_num(Tt)
    def loss(j): p = m(Xt[j]); return (((p - Tz[j]) ** 2) * msk[j]).sum() / msk[j].sum()
    _loop(m, loss, len(Xtr), np.where(val)[0], np.where(~val)[0], epochs, lr, wd, 32, seed)
    with torch.no_grad(): return m(torch.tensor(Xte)).numpy(), m
