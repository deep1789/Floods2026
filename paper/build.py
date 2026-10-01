import re, os, pandas as pd, numpy as np
def md(d, fmt='{:.3f}'):
    h = '| ' + ' | '.join(map(str, d.columns)) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    fm = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('' if isinstance(v, float) else str(v))
    return h + '\n'.join('| ' + ' | '.join(fm(v) for v in r) + ' |' for r in d.values) + '\n'
# combined main table
rows = []
for sc, lab in (('chrono', 'Chronological'), ('lomo', 'Leave-one-monsoon-out'), ('fc', 'Forward chaining')):
    t = pd.read_csv(f'results/main_{sc}.csv'); t.insert(0, 'Split', lab); rows.append(t)
open('tables2/main_all.md', 'w').write(md(pd.concat(rows)))
txt = '\n'.join(open(f'parts/p{i}.md').read() for i in (1, 2, 3, 4))
def tbl(m):
    n = m.group(1)
    for d in ('tables', 'tables2'):
        if os.path.exists(f'{d}/{n}.md'): return open(f'{d}/{n}.md').read()
    raise FileNotFoundError(n)
txt = re.sub(r'\{\{T:(\w+)\}\}', tbl, txt)
# renumber tables and figures in order of first caption appearance; update every reference
cap = re.findall(r'\*\*Table (\d+[a-z]?)[\.\s(]', txt); seen = []
for c in cap:
    if c not in seen: seen.append(c)
tm = {old: str(i + 1) for i, old in enumerate(seen)}
def trep(m): return 'Table ' + tm.get(m.group(1), m.group(1))
fig = re.findall(r'\*Figure (\d+)\.', txt); fm = {old: str(i + 1) for i, old in enumerate(dict.fromkeys(fig))}
# protect multi-reference forms like "Tables 5 and 10–12" by editing them explicitly beforehand
txt = re.sub(r'Table (\d+[a-z]?)\b', trep, txt); txt = re.sub(r'Figure (\d+)\b', lambda m: 'Figure ' + fm.get(m.group(1), m.group(1)), txt)
txt = txt.replace('Tables 5, 14–20', 'Tables ' + tm['5'] + ', ' + ', '.join(tm[k] for k in ('14', '15', '16', '17', '18', '19', '20') if k in tm)).replace('Tables 6 and 21', 'Tables ' + tm['6'] + ' and ' + tm['21'])
open('PAPER_PLAN.md', 'w').write(txt)
body = re.sub(r'```.*?```', '', txt, flags=re.S); body = re.sub(r'\$\$.*?\$\$', '', body, flags=re.S); body = re.sub(r'\|.*\|', '', body); body = re.sub(r'!\[.*?\]\(.*?\)', '', body)
print('words (prose only):', len(body.split()), ' total whitespace tokens:', len(txt.split()))
print('placeholders left:', re.findall(r'\{\{.*?\}\}', txt)); print('table map', tm); print('figure map', fm)
