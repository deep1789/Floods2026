"""Automated cross-check of numbers quoted in corrected/PAPER.md against the result files. Run from paper/: python check_numbers.py"""
import json, re, numpy as np, pandas as pd
B = 'corrected/results/'; txt = open('corrected/PAPER.md').read(); fails = []; n = 0
def chk(name, val, quoted, tol=0.0006, fmt='{:.3f}'):
    global n; n += 1; ok = abs(val - quoted) <= tol
    if not ok: fails.append(f'{name}: file {fmt.format(val)} vs text {quoted}')
def inpaper(s, name):
    global n; n += 1
    if s not in txt: fails.append(f'{name}: string not found in paper: {s[:60]}')
mc = pd.read_csv(B + 'main_chrono.csv').set_index('Model'); ml = pd.read_csv(B + 'main_lomo.csv').set_index('Model'); mf = pd.read_csv(B + 'main_fc.csv').set_index('Model')
for m, vals in {'B0_persistence': (.984, .996, .949, .981, .893, .944), 'HGB': (.990, .997, .904, .980, .878, .947), 'HGB_tuned': (.989, .997, .965, .984, .922, .951), 'LSTM': (.987, .997, .963, .987, .905, .964), 'LSTM_tuned': (.987, .997, .943, .984, .854, .958), 'TFT_lite': (.987, .997, .964, .988, .912, .966)}.items():
    for c, q in zip(['h=1 NSE', 'h=1 logNSE', 'h=3 NSE', 'h=3 logNSE', 'h=7 NSE', 'h=7 logNSE'], vals): chk(f'chrono {m} {c}', mc.loc[m, c], q)
for m, c, q in (('B0_persistence', 'h=3 NSE', .802), ('B3_ARX', 'h=3 NSE', .801), ('HGB', 'h=3 NSE', .802), ('B3_ARX', 'h=7 NSE', .553), ('B0_persistence', 'h=7 NSE', .585), ('B3_ARX', 'h=3 logNSE', .917), ('HGB', 'h=3 logNSE', .926), ('B0_persistence', 'h=3 logNSE', .897), ('B3_ARX', 'h=7 logNSE', .742), ('HGB', 'h=7 logNSE', .789), ('B0_persistence', 'h=7 logNSE', .672)): chk(f'lomo {m} {c}', ml.loc[m, c], q)
for m, c, q in (('HGB', 'h=7 NSE', .517), ('LSTM', 'h=7 NSE', .512), ('B0_persistence', 'h=7 NSE', .621), ('TFT_lite', 'h=7 NSE', .724), ('GRAPH_learned', 'h=7 NSE', .708)): chk(f'fc {m} {c}', mf.loc[m, c], q)
sd = pd.read_csv(B + 'lstm_seeds.csv').set_index('h'); chk('lstm seed mean h7', sd.loc[7, 'seed mean'], .952, .0006); chk('lstm seed sd h7', sd.loc[7, 'seed sd'], .011, .0006); chk('lstm seed min h7', sd.loc[7, 'seed min'], .934, .0006); chk('lstm seed max h7', sd.loc[7, 'seed max'], .965, .0006); chk('lstm ensemble h7', sd.loc[7, 'ensemble (10 seeds)'], .964, .0006)
sk = pd.read_csv(B + 'skill_tests.csv'); 
def cnt(sc, m, h): g = sk[(sk.scheme == sc) & (sk.model == m) & (sk.h == h)]; return int((g.skill_vs_B0 > 0).sum()), int(((g.skill_vs_B0 > 0) & (g.q_BH < .05)).sum())
for (sc, m, h), (b, s) in {('lomo', 'B3_ARX', 1): (9, 8), ('lomo', 'B3_ARX', 3): (10, 8), ('lomo', 'B3_ARX', 7): (10, 3), ('lomo', 'HGB', 1): (9, 6), ('lomo', 'HGB', 3): (9, 2), ('lomo', 'HGB', 7): (10, 3), ('chrono', 'B1_recession', 1): (10, 8), ('chrono', 'B1_recession', 3): (9, 6), ('chrono', 'GRAPH_learned', 7): (10, 5), ('chrono', 'B3_ARX', 7): (10, 0), ('chrono', 'HGB', 3): (7, 1)}.items():
    got = cnt(sc, m, h); n += 1
    if got != (b, s): fails.append(f'beat counts {sc} {m} h{h}: file {got} vs text {(b, s)}')
pr = pd.read_csv(B + 'prob.csv'); chk('coverage h1', pr.iloc[0]['90% interval coverage'], .865); chk('coverage h3', pr.iloc[1]['90% interval coverage'], .848); chk('hi-flow cov h1', pr.iloc[0]['coverage on training-q95 high-flow days'], .847); chk('hi-flow cov h3', pr.iloc[1]['coverage on training-q95 high-flow days'], .815)
sy = pd.read_csv(B + 'synchrony.csv'); chk('sync Rasu->Dev r0', sy.iloc[0]['r(k=0)'], .524); chk('sync Bahr->Chat r0', sy.iloc[1]['r(k=0)'], .524); chk('sync control1 r0', sy.iloc[2]['r(k=0)'], .117); chk('sync control2 r0', sy.iloc[3]['r(k=0)'], .114)
js = json.load(open(B + 'analyses_summary.json')); chk('soil beta', js['beta_theta'], -.025, .0006); chk('soil perm p', js['perm_p'], .80, .006); chk('n events', js['n_events'], 310, 0)
ab = pd.read_csv(B + 'ablation.csv').set_index('Variant'); chk('abl drop soil h3', ab.loc['drop soil (θ*, Δθ7)', 'h=3 Δlog-NSE'], -.003, .0006); chk('abl flow lags only h3', ab.loc['flow lags only (no weather, no site)', 'h=3 Δlog-NSE'], -.020, .0006); chk('abl drop season h3', ab.loc['drop season (sin/cos doy)', 'h=3 Δlog-NSE'], -.029, .0006)
pi = pd.read_csv(B + 'perm_importance.csv'); chk('perm flow lags h1', pi[(pi.Group == 'flow lags') & (pi.h == 1)].rel_increase_pct.iloc[0], 106, 0.6, '{:.1f}'); chk('perm current rain h1', pi[(pi.Group == 'current rain') & (pi.h == 1)].rel_increase_pct.iloc[0], 92, 0.6, '{:.1f}')
cs = pd.read_csv('results/extended/cell_scan_summary.csv').set_index('location'); chk('Chisapani best mean', cs.loc['Chisapani', 'best_mean_Q'], 1325, 1, '{:.0f}'); chk('Devghat best mean', cs.loc['Devghat', 'best_mean_Q'], 1609, 1, '{:.0f}'); chk('Chatara best mean', cs.loc['Chatara', 'best_mean_Q'], 1829, 1, '{:.0f}'); chk('ratio min among replaced', cs[cs.ratio_best_to_center > 10].ratio_best_to_center.min(), 17, 0.5, '{:.1f}'); chk('ratio max', cs.ratio_best_to_center.max(), 1565, 1, '{:.0f}')
tl = json.load(open(B + 'tune_hgb.json'))['best']; chk('tune inner h3 best', tl['3']['inner_mean'], .9086, .0006, '{:.4f}'); chk('tune inner h3 default', tl['3']['default_inner_mean'], .8913, .0006, '{:.4f}')
lg = pd.read_csv(B + 'lag_weights.csv').set_index('Location'); chk('Khokana centroid', lg.loc['Khokana', 'centroid'], 1.20, .006, '{:.2f}'); chk('Rasuwagadhi centroid', lg.loc['Rasuwagadhi', 'centroid'], 1.15, .006, '{:.2f}'); chk('Bhada centroid', lg.loc['Bhada Bridge', 'centroid'], 3.72, .006, '{:.2f}'); chk('Khokana cum', lg.loc['Khokana', 'cum_response'], .44, .006, '{:.2f}')
pt = pd.read_csv('corrected/results/pettitt.csv') if False else pd.read_csv(B + 'pettitt.csv'); n += 1
if int((pt.p_bonf < .05).sum()) != 46: fails.append(f'pettitt bonf count {(pt.p_bonf<.05).sum()} vs 46')
for s in ('0.984, 0.949 and 0.893', '8,649 m³/s', '38-fold', '0.52 at lag 0 for Rasuwagadhi', '46 of the 60 tests'): inpaper(s, 'string')
print(f'{n} checks, {len(fails)} mismatches'); [print(' -', f) for f in fails]
