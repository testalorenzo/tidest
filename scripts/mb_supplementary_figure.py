"""
Supplementary figure — per-gene results across all 55 cortical layer markers.
Single-page, two-column layout.

Each column:
  Left:  TIDEST τ̂ ± 95% CI lollipop (alphabetical).
  Right: 5 small colored squares per gene (one per method).
         Significant (q < 0.05): red (superficial effect) or blue (deep effect).
         Non-significant: soft grey.

Output: results_raw/mb_supplementary_figure.pdf
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':       6,
    'axes.labelsize':  6,
    'xtick.labelsize': 5.5,
    'ytick.labelsize': 6,
    'axes.linewidth':  0.5,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# ── Colors ───────────────────────────────────────────────────────────────────
C_SUP   = '#C0392B'
C_DEEP  = '#2471A3'
C_NS    = '#AAAAAA'
C_CON   = '#7D3C98'
C_INT   = '#E67E22'
C_GREY  = '#DDDDDD'   # non-significant square

_SUP_GENES  = {'cux1','cux2','calb1','rasgrf2','pou3f2','sema3c','lhx2',
               'rorb','rspo1','scnn1a','plcxd2','krt73'}
_DEEP_GENES = {'bcl11b','fezf2','foxp2','sox5','tbr1','tle4','tshz3','ldb2',
               'sulf1','cdh13','syt6','nxph4','nfe2l1','etv1','tdo2','chrna6',
               'nxph2','syt17','grik1','trhr','npsr1','chrna4','htr2c'}
_INT_GENES  = {'vip','sst','pvalb','reln','lhx6'}
_CON_GENES  = {'prss12','pou3f3','satb2'}

def _gene_color(g):
    if g in _INT_GENES:  return C_INT
    if g in _DEEP_GENES: return C_DEEP
    if g in _SUP_GENES:  return C_SUP
    return C_CON   # everything else = Ambiguous

METHODS   = ['TIDEST', 't-test', 'SpaGCN', 'DESpace', 'SpatialGEE']
Q_COLS    = {'TIDEST':'qval_tidest', 't-test':'qval_ttest',
             'SpaGCN':'qval_spagcn', 'DESpace':'qval_despace',
             'SpatialGEE':'qval_spatialgee'}
EFF_COLS  = {'TIDEST':'tau_tidest', 't-test':'tau_ttest',
             'SpaGCN':'z_spagcn',  'DESpace':'logfc_despace',
             'SpatialGEE':'coef_spatialgee'}

# ── Load data ─────────────────────────────────────────────────────────────────
plm = pd.read_csv('./results_raw/plm_results0.csv').set_index('gene')
cmp = pd.read_csv('./results_raw/mb_pergene_allmethods.csv').set_index('gene')

genes  = sorted(cmp.index.tolist())
N      = len(genes)
half   = (N + 1) // 2          # ≈ 28
# Reverse so alphabetically-first genes appear at the top (y-axis goes bottom→top)
genes0 = list(reversed(genes[:half]))
genes1 = list(reversed(genes[half:]))

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 5.80, 9.20
fig = plt.figure(figsize=(FIG_W, FIG_H))

gs_main = gridspec.GridSpec(1, 2, figure=fig,
                            wspace=0.28,
                            left=0.02, right=0.99,
                            top=0.96, bottom=0.07)

# ── Helper: draw one column ───────────────────────────────────────────────────
def draw_column(ax_lol, ax_sq, gene_list, show_ylabel=True, col_title=''):
    n = len(gene_list)
    y = np.arange(n)

    # ── Lollipop ──────────────────────────────────────────────────────────────
    for i, g in enumerate(gene_list):
        if g not in plm.index:
            continue
        r   = plm.loc[g]
        tau = r['tau'];  ci = 1.96 * r['se'];  q = r['qval']
        col = _gene_color(g)
        sig = q < 0.05

        ax_lol.plot([tau - ci, tau + ci], [i, i],
                    color=col, lw=0.8, solid_capstyle='round', zorder=2, alpha=0.9)
        ax_lol.scatter([tau], [i], s=18, marker='o', color=col,
                       edgecolors='none', zorder=5,
                       alpha=1.0 if sig else 0.28)

    ax_lol.axvline(0, color='#444444', lw=0.5, ls='--', alpha=0.5)
    ax_lol.xaxis.grid(True, alpha=0.08, lw=0.4, zorder=0)
    ax_lol.set_axisbelow(True)
    ax_lol.set_yticks(y)
    ax_lol.set_yticklabels([g.capitalize() for g in gene_list],
                           fontsize=6, style='italic')
    for tick, g in zip(ax_lol.get_yticklabels(), gene_list):
        tick.set_color(_gene_color(g))
    ax_lol.set_xlim(-0.29, 0.29)
    ax_lol.set_ylim(-0.5, n - 0.5)
    ax_lol.set_xlabel('$\\hat{\\tau}$ ± 95% CI', fontsize=6)
    ax_lol.spines[['top', 'right', 'left']].set_visible(False)
    ax_lol.tick_params(left=False, labelsize=5.5)
    if col_title:
        ax_lol.set_title(col_title, fontsize=6.5, loc='left', pad=3)

    # ── q-value squares (single gradient 0→0.05) ─────────────────────────────
    sq_size  = 0.50
    _cmap    = plt.cm.Greens      # dark=significant, light=near threshold
    _grey_rgb = np.array(mcolors.to_rgb(C_GREY))

    for i, g in enumerate(gene_list):
        for j, m in enumerate(METHODS):
            q = cmp.loc[g, Q_COLS[m]]
            if q >= 0.05:
                col   = tuple(_grey_rgb)
                alpha = 0.40
            else:
                t   = 1.0 - q / 0.05        # 0 at q=0.05 → 1 at q=0
                col = _cmap(0.20 + 0.75 * t) # avoid pure white end
                alpha = 1.0
            rect = Rectangle((j - sq_size/2, i - sq_size/2),
                              sq_size, sq_size,
                              facecolor=col, edgecolor='none', alpha=alpha)
            ax_sq.add_patch(rect)

    ax_sq.set_xlim(-0.52, len(METHODS) - 0.48)
    ax_sq.set_ylim(-0.5, n - 0.5)
    ax_sq.set_xticks(range(len(METHODS)))
    ax_sq.set_xticklabels(METHODS, fontsize=5.5, rotation=40, ha='right')
    ax_sq.set_yticks([])
    ax_sq.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax_sq.tick_params(bottom=False)


# ── Column 0 ──────────────────────────────────────────────────────────────────
gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0],
                                       width_ratios=[2.8, 1.2], wspace=0.02)
ax_lol0 = fig.add_subplot(gs0[0])
ax_sq0  = fig.add_subplot(gs0[1])
draw_column(ax_lol0, ax_sq0, genes0,
            col_title='TIDEST estimated effects — all 55 genes')

# ── Column 1 ──────────────────────────────────────────────────────────────────
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1],
                                       width_ratios=[2.8, 1.2], wspace=0.02)
ax_lol1 = fig.add_subplot(gs1[0])
ax_sq1  = fig.add_subplot(gs1[1])
draw_column(ax_lol1, ax_sq1, genes1,
            col_title='(continued)')

# ── Shared legend ─────────────────────────────────────────────────────────────
leg_items = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_SUP,  ms=5, label='Superficial markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DEEP, ms=5, label='Deep markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_INT,  ms=5, label='Interneuron markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_CON,  ms=5, label='Ambiguous'),
]
fig.legend(handles=leg_items, loc='lower left',
           bbox_to_anchor=(0.02, -0.05), fontsize=5.5, ncol=3,
           frameon=True, framealpha=0.9, edgecolor='#cccccc',
           handletextpad=0.3, columnspacing=0.8, handlelength=1.2)

# ── q-value colorbar ──────────────────────────────────────────────────────────
import matplotlib.cm as cm
import matplotlib.colorbar as mcolorbar
ax_cb = fig.add_axes([0.72, -0.04, 0.22, 0.018])   # [left, bottom, w, h] in fig coords
norm  = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
cb    = mcolorbar.ColorbarBase(ax_cb, cmap=plt.cm.Greens_r,
                               norm=norm, orientation='horizontal')
cb.set_label('BH-adjusted $q$-value', fontsize=5.5, labelpad=2)
cb.ax.tick_params(labelsize=5)
cb.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])

# ── Save ──────────────────────────────────────────────────────────────────────
out = './results_raw/mb_supplementary_figure.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
