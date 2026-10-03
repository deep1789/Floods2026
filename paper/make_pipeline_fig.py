import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(0, 3.4)
def box(x, y, w, h, t, fc='#e8eef7'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03', fc=fc, ec='#3b5b8c', lw=1)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.5, wrap=True)
def arrow(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='#3b5b8c', lw=1))
box(0.1, 1.3, 1.1, 0.8, 'Panel (corrected\ndischarge)'); box(1.5, 1.3, 1.1, 0.8, 'Audit A1–A8\n+ cell scan'); box(2.9, 1.3, 1.2, 0.8, 'Temporal split by\nmonsoon; purge')
box(4.4, 1.3, 1.5, 0.8, 'Fit on TRAIN only:\nlog, scalers, climatology,\nrecession, thresholds')
box(6.2, 1.3, 1.1, 0.8, 'Causal feature\nbuilder')
for i, (t, y) in enumerate([('Baselines B0–B3', 2.75), ('Boosting (default, tuned)', 2.05), ('LSTM / TFT-lite / graph', 1.35), ('Ridge ARX, quantile models', 0.65)]): box(7.7, y - 0.25, 1.6, 0.5, t, '#f3ead7')
box(9.6, 1.3, 1.3, 0.8, 'NSE, KGE, log-NSE,\nevent scores, CRPS;\nDM tests, bootstrap', '#e3f1e3')
for a, b in ((1.2, 1.5), (2.6, 2.9), (4.1, 4.4), (5.9, 6.2)): arrow(a, 1.7, b, 1.7)
for y in (2.75, 2.05, 1.35, 0.65): arrow(7.3, 1.7, 7.7, y); arrow(9.3, y, 9.6, 1.7)
ax.text(5.5, 0.1, 'Every learned quantity (box 4) is estimated on the training period only.', ha='center', fontsize=8, style='italic')
fig.savefig('corrected/figures/fig_pipeline.png', dpi=170, bbox_inches='tight'); print('ok')
