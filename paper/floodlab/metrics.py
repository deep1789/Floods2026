import numpy as np
from scipy import stats

def nse(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float); d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((o - p) ** 2).sum() / d) if d > 0 else np.nan

def kge(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    if o.std() == 0 or p.std() == 0: return np.nan
    r = np.corrcoef(o, p)[0, 1]; a = p.std() / o.std(); b = p.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))

def skill(o, p, pref):
    """MSE skill score vs reference forecast."""
    o = np.asarray(o, float); return float(1 - ((o - p) ** 2).mean() / ((o - pref) ** 2).mean())

def pinball(o, q, a): u = np.asarray(o) - np.asarray(q); return float(np.mean(u * (a - (u < 0))))

def contingency(obs, pred):
    a = int((obs & pred).sum()); b = int((~obs & pred).sum()); c = int((obs & ~pred).sum())
    pod = a / (a + c) if a + c else np.nan; far = b / (a + b) if a + b else np.nan
    csi = a / (a + b + c) if a + b + c else np.nan
    return dict(hits=a, false_alarms=b, misses=c, POD=pod, FAR=far, CSI=csi)

def stationary_bootstrap_idx(n, b, rng):
    """Politis-Romano stationary bootstrap indices, geometric block lengths with mean b."""
    idx = np.empty(n, int); i = 0; p = 1.0 / b
    while i < n:
        s = rng.integers(0, n); L = rng.geometric(p)
        for k in range(min(L, n - i)): idx[i + k] = (s + k) % n
        i += min(L, n - i)
    return idx

def boot_ci(fn, arrays, b=10, R=300, seed=0):
    """Algorithm 4: CI of a statistic fn(*arrays) with paired stationary bootstrap."""
    rng = np.random.default_rng(seed); n = len(arrays[0]); v = []
    for _ in range(R):
        ix = stationary_bootstrap_idx(n, b, rng); v.append(fn(*[a[ix] for a in arrays]))
    v = np.array(v); return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))

def dm_test(eA, eB):
    """Diebold-Mariano on squared-error loss differential with Newey-West HAC variance."""
    d = np.asarray(eA) ** 2 - np.asarray(eB) ** 2; n = len(d); dm = d - d.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g0 = (dm @ dm) / n; var = g0
    for l in range(1, L + 1):
        var += 2 * (1 - l / (L + 1)) * (dm[l:] @ dm[:-l]) / n
    stat = d.mean() / np.sqrt(var / n) if var > 0 else 0.0
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))
