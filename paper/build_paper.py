"""Assemble corrected/PAPER.md from corrected/parts (new sections + patched carry-over sections) and the generated tables."""
import re, os, pandas as pd, numpy as np
B = 'corrected'
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
# --- generated tables
rows = []
for sc, lab in (('chrono', 'Chronological'), ('lomo', 'Leave-one-monsoon-out'), ('fc', 'Forward chaining (2025 and 2026 folds)'), ('fc26', 'Forward chaining, 2026 fold only')):
    t = pd.read_csv(f'{B}/results/main_{sc}.csv'); t.insert(0, 'Split', lab); rows.append(t)
open(f'{B}/tables2/main_all.md', 'w').write(md(pd.concat(rows)))
S = pd.read_csv('results/extended/cell_scan_summary.csv')
S['off'] = S.best_cell; T = pd.DataFrame({'Location': S.location, 'Dataset cell mean Q': S.dataset_mean.round(1), 'Dataset cell reproduces file': S.center_cell_reproduces_dataset.map({True: 'yes', False: 'no'}), 'Best neighbouring cell offset': S.off,
    'Best-cell mean Q': S.best_mean_Q.round(1), 'Ratio best / dataset': S.ratio_best_to_center.round(1), 'Cells >10×': S.cells_with_mean_gt_10x_center, 'Replaced in corrected panel': (S.ratio_best_to_center > 10).map({True: 'yes', False: 'no'})})
order = pd.read_csv(f'{B}/tables/sites.md', sep='|', skiprows=2, header=None).iloc[:, 1].str.strip().tolist() if False else None
ia = pd.read_csv(f'{B}/results/implied_area.csv'); v2 = ia[ia.series == 'V2 file'].set_index('location'); co = ia[ia.series == 'corrected'].set_index('location')
IA = pd.DataFrame({'Location': co.index, 'Annual precipitation at coordinate (mm)': co.P_mm_yr.round(0).values, 'V2 mean Q (m³/s)': v2.mean_Q.round(1).values, 'V2 implied area (km²)': v2.area_c05.round(0).values, 'Corrected mean Q (m³/s)': co.mean_Q.round(1).values,
    'Corrected implied area (km²)': co.area_c05.round(0).values, 'Range for c = 0.7–0.3 (km²)': [f'{lo:,.0f}–{hi:,.0f}' for lo, hi in zip(co.area_lo, co.area_hi)]}).sort_values('Corrected implied area (km²)', ascending=False)
open(f'{B}/tables2/implied_area.md', 'w').write(md(IA, '{:,.0f}'))
open(f'{B}/tables2/cellscan.md', 'w').write(md(T.sort_values('Ratio best / dataset', ascending=False), '{:.1f}'))
# --- carry-over sections with patches
def patch(s, pairs):
    for a, b in pairs:
        assert a in s, a[:80]; s = s.replace(a, b)
    return s
P = lambda n: open(f'{B}/parts/{n}').read()
q1 = P('q1.md')
p2 = P('p2.md'); sec4 = p2[p2.index('## 4. Mathematical Formulation'):p2.index('### 4.8 A worked example')]
sec4 = patch(sec4, [("(this is relevant for locations where absolute magnitudes are unreliable, per audit A4)", "(this was relevant when absolute magnitudes were unreliable in the uncorrected series, audit A4)")])
p3 = patch(P('p3.md'), [("| T4 Zero-flow | $\\mathbb{1}[Q_{t+h}=0]$ | 1 d | Khokana only; handles audit A6 |", "| T4 Zero-flow | — | — | dropped: the corrected series has no zero flow (audit A6) |"),
    ("*In the reported results no search was run: fixed configurations were used (Section 8.6), and the TFT and graph arms were not run.*", "*Boosting and the LSTM were tuned in a nested, bounded search (Section 8.5); the ridge model has built-in selection; TFT-lite and the graph networks use default settings and 5 seeds; the reference TFT was not used.*"),
    ("between 0.49 and 0.94; Section 8.2", "between 0.60 and 0.98; Section 8.2")])
p4 = P('p4.md'); sec7 = p4[:p4.index('## 8. ')]; refs = p4[p4.index('## References'):]
sec7 = patch(sec7, [("**Status: done (Section 8.8).**", "**Status: done (Section 8.7).**"), ("Raw cross-correlations (Table 9)", "Raw cross-correlations (Table 104)"),
    ("The completed benchmark is reported in Section 8.6.", "The completed benchmark is reported in Sections 8.4 and 8.5."), ("The completed experiment is reported in Section 8.7.", "The completed experiment is reported in Section 8.6."),
    ("exploratory (done; Section 8.10, with a negative methodological result)", "exploratory (done; Section 8.8, with a negative methodological result)"),
    ("exploratory and useful as a diagnostic (done; Sections 8.9–8.10)", "exploratory and useful as a diagnostic (done; Section 8.8)"),
    ("(audit A4)", "(audit A4; the cell scan of Section 3.4 resolves the gross scale error but not the match to gauges)")])
sec7 = re.sub(r'(### 7\.7 [^\n]*— \*exploratory)(\*)', r'\1 (done; Section 8.7)\2', sec7)
sec7 = re.sub(r'followed the precipitation peak by 1 and 3 days \(Table 13\)', 'followed the precipitation peak by 2 days at both (Table 103)', sec7)
txt = '\n'.join([q1, P('new_s3.md'), sec4, P('new_s48.md'), '---\n', p3, sec7, P('new_s8a.md'), P('new_s8b.md'), P('new_s9_10.md'), refs])
def tbl(m):
    n = m.group(1)
    for d in ('tables', 'tables2'):
        if os.path.exists(f'{B}/{d}/{n}.md'): return open(f'{B}/{d}/{n}.md').read()
    raise FileNotFoundError(n)
txt = re.sub(r'\{\{T:(\w+)\}\}', tbl, txt)
cap = re.findall(r'\*\*Table (\d+[a-z]?)[\.\s(]', txt); seen = list(dict.fromkeys(cap)); tm = {o: str(i + 1) for i, o in enumerate(seen)}
fig = re.findall(r'\*Figure (\d+)\.', txt); fm = {o: str(i + 1) for i, o in enumerate(dict.fromkeys(fig))}
unmapped = sorted(set(re.findall(r'Table (\d+[a-z]?)\b', txt)) - set(tm)); figun = sorted(set(re.findall(r'Figure (\d+)\b', txt)) - set(fm))
txt = re.sub(r'Table (\d+[a-z]?)\b', lambda m: 'Table ' + tm.get(m.group(1), m.group(1)), txt); txt = re.sub(r'Figure (\d+)\b', lambda m: 'Figure ' + fm.get(m.group(1), m.group(1)), txt)
open(f'{B}/PAPER.md', 'w').write(txt)
body = re.sub(r'```.*?```', '', txt, flags=re.S); body = re.sub(r'\$\$.*?\$\$', '', body, flags=re.S); body = re.sub(r'\|.*\|', '', body); body = re.sub(r'!\[.*?\]\(.*?\)', '', body)
print('prose words:', len(body.split()), ' total tokens:', len(txt.split())); print('placeholders left:', re.findall(r'\{\{.*?\}\}', txt)); print('unmapped table refs:', unmapped, ' unmapped fig refs:', figun)
