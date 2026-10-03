"""Previously unrun items: CUSUM test, profile-likelihood GPD intervals, TreeSHAP (LightGBM), LSTM autoencoder detector. Run from corrected/ with FLOOD_CSV set."""
import numpy as np, pandas as pd, warnings, torch, torch.nn as nn
from scipy import stats, optimize
import lightgbm as lgb, shap
from floodlab import data, features as F, models as M, events as Ev, lstm as L
from floodlab.config import T0
warnings.filterwarnings('ignore'); torch.set_num_threads(4)
df = data.load(); order = data.site_order(df); Q, P = data.Q, data.P
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
w = lambda n, t: open(f'tables2/{n}.md', 'w').write(t)
# ======== A. CUSUM test on the seasonally adjusted weather series (same residuals as breaks.py)
VARS = {'precipitation_mm': 'precipitation', 'temperature_mean_c': 'temperature', 'soil_moisture_0_100cm_m3m3': 'soil moisture', 'relative_humidity_mean_pct': 'relative humidity', 'dew_point_mean_c': 'dew point', 'wind_speed_max_kmh': 'max wind speed'}
pet = pd.read_csv('results/pettitt.csv'); rows = []
for s in order:
    g = df[df.location == s].sort_values('date'); X = F._design(g.date.dt.dayofyear.values)
    for v, lab in VARS.items():
        y = g[v].values.astype(float); r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; n = len(r); S = np.cumsum(r - r.mean()) / (r.std() * np.sqrt(n)); k = int(np.argmax(np.abs(S))); x = abs(S[k])
        p = float(min(1, 2 * sum((-1) ** (j - 1) * np.exp(-2 * j * j * x * x) for j in range(1, 100))))
        pd_ = pet[(pet.location == s) & (pet.variable == lab)].change_date.iloc[0]
        rows.append(dict(location=s, variable=lab, cusum_date=str(g.date.iloc[k].date()), stat=x, p=p, days_from_pettitt=abs((g.date.iloc[k] - pd.Timestamp(pd_)).days)))
CU = pd.DataFrame(rows); CU['p_bonf'] = np.minimum(1, CU.p * len(CU)); CU.to_csv('results/cusum.csv', index=False)
sm = CU.groupby('variable').agg(tests=('p', 'size'), sig_raw=('p', lambda x: int((x < .05).sum())), sig_bonf=('p_bonf', lambda x: int((x < .05).sum())), median_days_from_pettitt=('days_from_pettitt', 'median')).reset_index()
w('cusum', md(sm.rename(columns={'variable': 'Variable', 'tests': 'Tests', 'sig_raw': 'p<0.05 (raw)', 'sig_bonf': 'p<0.05 (Bonferroni, 60 tests)', 'median_days_from_pettitt': 'Median days from Pettitt date'}), '{:.0f}')); print(sm.to_string())
# ======== B. profile-likelihood intervals for the GPD shape (events excluded as in analyses.py)
def gll(xi, sg, x):
    if sg <= 0: return -np.inf
    if abs(xi) < 1e-8: return -len(x) * np.log(sg) - x.sum() / sg
    t = 1 + xi * x / sg
    return -np.inf if (t <= 0).any() else -len(x) * np.log(sg) - (1 + 1 / xi) * np.log(t).sum()
def profile(x, grid=np.linspace(-1.5, 1.5, 301)):
    m = x.mean(); xs = x / m; pl = []
    for xi in grid:
        lo = max(1e-4, -xi * xs.max() * (1 + 1e-6)) if xi < 0 else 1e-4; hi = 1e3
        res = optimize.minimize_scalar(lambda ls: -gll(xi, np.exp(ls), xs), bounds=(np.log(lo), np.log(hi)), method='bounded'); pl.append(-res.fun if np.isfinite(res.fun) else -np.inf)
    pl = np.array(pl); k = int(np.argmax(pl)); ok = grid[pl >= pl[k] - 1.92]; return grid[k], ok.min(), ok.max(), (ok.min() <= grid[0] + 1e-9) or (ok.max() >= grid[-1] - 1e-9)
EVT = (pd.Timestamp('2024-09-24'), pd.Timestamp('2024-09-30')); gp = pd.read_csv('results/gpd.csv'); rows = []
for var, lab in ((P, 'P'), (Q, 'Q')):
    for s in order:
        g = df[df.location == s].reset_index(drop=True); z = g[var].values.astype(float)
        if lab == 'Q': z = z / max(np.median(z), 1e-3)
        u = np.quantile(z, .95); zk = z.copy(); zk[((g.date >= EVT[0]) & (g.date <= EVT[1])).values] = -1
        ex = np.array([e[3] for e in Ev.detect_events(zk, u, r=3)]) - u; ex = ex[ex > 0]
        if len(ex) < 8: continue
        xi, lo, hi, edge = profile(ex); b = gp[(gp.Variable == lab) & (gp.Location == s)]
        rows.append(dict(Variable=lab, Location=s, peaks=len(ex), xi_hat=xi, profile_lo=lo, profile_hi=hi, hit_grid_edge=edge, boot_lo=float(b.xi_lo.iloc[0]) if len(b) else np.nan, boot_hi=float(b.xi_hi.iloc[0]) if len(b) else np.nan, contains_zero=bool(lo <= 0 <= hi)))
PR = pd.DataFrame(rows); PR.to_csv('results/gpd_profile.csv', index=False)
t = PR.copy(); t['hit_grid_edge'] = t.hit_grid_edge.map({True: 'yes', False: ''}); t['contains_zero'] = t.contains_zero.map({True: 'yes', False: 'no'})
t.columns = ['Variable', 'Location', 'Peaks', 'ξ̂ (profile max)', 'Profile 2.5%', 'Profile 97.5%', 'Interval hits grid edge (±1.5)', 'Bootstrap 2.5%', 'Bootstrap 97.5%', 'Profile interval contains 0']; w('gpd_profile', md(t, '{:.2f}')); print(PR.round(2).to_string())
# ======== C. TreeSHAP on a LightGBM model with the boosted model's features (chronological split)
fit = F.Fit(df, df.date < T0); f = F.build(df, fit); f['sid'] = M.sid(f)
GR = {'flow lags': ['y', 'y_l1', 'y_l2', 'y_l3', 'y_l7', 'dy', 'y_anom'], 'current rain': ['lp', 'lp_l1', 'lp_l2', 'lp_l3', 'P_s3', 'P_s7', 'intens', 'wet_spell'], 'antecedent rain': ['API80', 'API95', 'P_s14', 'P_s30'],
      'soil moisture': ['th_star', 'th_d7'], 'temperature, humidity, snow': ['temperature_mean_c', 'dtemp', 'pdd7', 'snowfrac', 'relative_humidity_mean_pct'], 'season': ['sin_doy', 'cos_doy'], 'site & elevation': ['elevation_m', 'site_id']}
rows = []
for h in (1, 3, 7):
    tr, te = F.split_rows(f, h, lambda d: d.date_tgt < T0, lambda d: d.date >= T0); Xtr, Xte = M.design(tr), M.design(te)
    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20, reg_lambda=1.0, random_state=0, verbose=-1).fit(Xtr, tr.y_tgt - tr.y, categorical_feature=['site_id'])
    sv = shap.TreeExplainer(mdl).shap_values(Xte); ma = np.abs(sv).mean(0); tot = ma.sum()
    for g_, cols in GR.items(): rows.append(dict(h=h, Group=g_, share_pct=100 * ma[[list(Xte.columns).index(c) for c in cols]].sum() / tot))
SH = pd.DataFrame(rows); SH.to_csv('results/shap_groups.csv', index=False)
w('shap_groups', md(SH.pivot(index='Group', columns='h', values='share_pct').add_prefix('h=').add_suffix(' (% of mean |SHAP|)').reset_index().sort_values('h=1 (% of mean |SHAP|)', ascending=False), '{:.1f}')); print(SH.pivot(index='Group', columns='h', values='share_pct').round(1))
# ======== D. LSTM autoencoder detector (unsupervised, full record)
sites = order; sidmap = {s: i for i, s in enumerate(sites)}; fa = F.build(df, F.Fit(df, df.date >= '2000-01-01')); st = L.norm_stats(fa, (df.date >= '2000-01-01').values)
X, S_, D, N, Y0, T = L.make_seqs(fa, st, sidmap); X = np.nan_to_num(X)
class AE(nn.Module):
    def __init__(s, nin, hid=16):
        super().__init__(); s.enc = nn.LSTM(nin, hid, batch_first=True); s.dec = nn.LSTM(hid, hid, batch_first=True); s.out = nn.Linear(hid, nin)
    def forward(s, x):
        _, (h, _) = s.enc(x); z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1); o, _ = s.dec(z); return s.out(o)
torch.manual_seed(0); m = AE(X.shape[2]); opt = torch.optim.Adam(m.parameters(), lr=3e-3); Xt = torch.tensor(X)
for ep in range(40):
    perm = torch.randperm(len(Xt)); m.train()
    for b in range(0, len(Xt), 256):
        j = perm[b:b + 256]; opt.zero_grad(); loss = ((m(Xt[j]) - Xt[j]) ** 2).mean(); loss.backward(); opt.step()
m.eval()
with torch.no_grad(): rec = (m(Xt) - Xt).numpy(); err = (rec[:, -3:, :] ** 2).mean((1, 2))        # error on the last 3 days of each window
A = pd.DataFrame(dict(location=N, date=pd.to_datetime(D), err=err))
ev = [('Kusum', '2024-09-28', 'Sep-2024 storm (rain-driven)'), ('Khokana', '2024-09-28', 'Sep-2024 storm (rain-driven)'), ('Devghat', '2024-09-29', 'Sep-2024 storm (rain-driven)'), ('Chatara', '2024-08-16', 'Thame GLOF, nearest location'), ('Bahrabise', '2024-08-16', 'Thame GLOF, nearest Koshi site'), ('Rasuwagadhi', '2025-07-08', 'Bhote Koshi flash flood')]
rows = []
for s, d, lab in ev:
    g = A[A.location == s]; t0 = pd.Timestamp(d); win = g[(g.date >= t0 - pd.Timedelta(days=1)) & (g.date <= t0 + pd.Timedelta(days=1))]
    rows.append(dict(Location=s, Date=d, Event=lab, max_error=win.err.max(), percentile=(g.err < win.err.max()).mean() * 100))
AEd = pd.DataFrame(rows); AEd.to_csv('results/autoencoder.csv', index=False); w('autoencoder', md(AEd.rename(columns={'percentile': 'percentile within location record'}), '{:.2f}')); print(AEd.round(2).to_string())
