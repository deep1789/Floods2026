import numpy as np

def detect_events(z, u, r=3):
    """Algorithm 3: declustered threshold exceedance events. Returns list of (start, peak, end, zpeak) as indices."""
    z = np.asarray(z, float); ex = np.where(z > u)[0]
    if len(ex) == 0: return []
    groups = [[ex[0]]]
    for i in ex[1:]:
        if i - groups[-1][-1] < r: groups[-1].append(i)
        else: groups.append([i])
    ev = []
    for g in groups:
        lo, hi = g[0], g[-1]; pk = lo + int(np.argmax(z[lo:hi + 1]))
        s = lo
        while s > 0 and z[s - 1] > u: s -= 1
        e = hi
        while e < len(z) - 1 and z[e + 1] > u: e += 1
        ev.append((s, pk, e, float(z[pk])))
    return ev

def event_scores(obs, pred, u, r=3, tol=1):
    """Event-level POD (observed event hit if pred exceeds u within +-tol days of its window),
    peak timing error (days) and relative peak error for hit events."""
    ev = detect_events(obs, u, r); hits = 0; terr = []; perr = []
    for s, pk, e, zp in ev:
        lo, hi = max(0, s - tol), min(len(obs) - 1, e + tol)
        w = pred[lo:hi + 1]
        if (w > u).any():
            hits += 1; pp = lo + int(np.argmax(w)); terr.append(pp - pk); perr.append((pred[pp] - zp) / zp)
    return dict(n_events=len(ev), event_hits=hits, event_POD=hits / len(ev) if ev else np.nan,
                timing_mae=float(np.mean(np.abs(terr))) if terr else np.nan,
                peak_rel_err=float(np.median(perr)) if perr else np.nan)
