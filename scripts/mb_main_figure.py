"""
Nature Methods publication-quality figure — MB mouse brain application.

Layout (3 rows, full-width 183 mm):
  Row 1: [A  tissue section] [B  effect-size lollipop — 21 selected genes]
  Row 2: [C1 Sox5 map] [C2 Tshz3 map] [C3 Vip map] [C4 Satb2 map]
  Row 3: [D1 Sox5 vln] [D2 Tshz3 vln] [D3 Vip multi-method] [D4 Satb2 vln]

Stories:
  D1 — TIDEST detects Sox5 (deep marker); all 4 competitors miss (q >= 0.965)
  D2 — TIDEST null for Tshz3; all 4 competitors FP in wrong direction
  D3 — TIDEST correctly null for Vip (interneuron); all 4 competitors FP
  D4 — TIDEST correctly null for Satb2 (expressed in both layers)

Data sources:
  results_raw/plm_results0.csv        — canonical PLM results (55-gene panel)
  results_raw/mb_pergene_allmethods.csv — per-gene 5-method comparison
  results_raw/tangram_pseudo0.pkl      — pseudo-outcome expression

Output:
  results_raw/mb_main_figure.pdf   — vectorized, submission-ready
  results_raw/mb_main_figure.png   — 300 dpi raster preview
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.sparse import issparse
from scipy.spatial import cKDTree
import squidpy as sq
import scanpy as sc

# ── Global typography (Nature Methods style) ──────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    7.5,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'legend.fontsize':   6.5,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'lines.linewidth':   0.9,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── Color palette ─────────────────────────────────────────────────────────────
C_SUP     = '#C0392B'   # superficial
C_DEEP    = '#2471A3'   # deep
C_NS      = '#AAAAAA'   # not significant
C_CON     = '#7D3C98'   # controversy (Satb2)
C_PLM     = '#1E8449'   # PLM-unique story (Sox5, Tshz3)
C_INT     = '#E67E22'   # interneuron false positive

# Method colors (consistent with simulation figure)
MC = {
    'tidest':     '#EE6677',
    'ttest':      '#4477AA',
    'spagcn':     '#228833',
    'despace':    '#CCBB44',
    'spatialgee': '#AA3377',
}
METHOD_LABELS = {
    'tidest': 'TIDEST', 'ttest': 't-test', 'spagcn': 'SpaGCN',
    'despace': 'DESpace', 'spatialgee': 'SpatialGEE',
}
METHODS = ['tidest', 'ttest', 'spagcn', 'despace', 'spatialgee']

# ── Gene sets ──────────────────────────────────────────────────────────────────
LOLLIPOP_GENES = [
    # Deep markers
    'tshz3', 'sox5', 'fezf2', 'foxp2', 'crym', 'bcl11b', 'etv1', 'syt6', 'tle4', 'tbr1',
    # Interneuron markers (negative controls)
    'lhx6', 'reln', 'vip', 'sst', 'pvalb',
    # Controversies
    'prss12', 'pou3f3', 'satb2',
    # Superficial markers
    'rorb', 'rasgrf2', 'calb1', 'cux1', 'cux2',
]

FOCUS_GENES  = ['sox5', 'tshz3', 'vip', 'satb2']          # C + D panels
BOLD_GENES   = {'sox5', 'tshz3', 'vip', 'satb2', 'pou3f3', 'prss12'}  # bold in lollipop

GENE_DISPLAY = {
    'tshz3':'Tshz3','sox5':'Sox5','fezf2':'Fezf2','foxp2':'Foxp2',
    'crym':'Crym','bcl11b':'Bcl11b','etv1':'Etv1','syt6':'Syt6',
    'tle4':'Tle4','tbr1':'Tbr1','lhx6':'Lhx6','reln':'Reln',
    'vip':'Vip','sst':'Sst','pvalb':'Pvalb','satb2':'Satb2',
    'prss12':'Prss12','pou3f3':'Pou3f3',
    'rorb':'Rorb','rasgrf2':'Rasgrf2','calb1':'Calb1',
    'cux1':'Cux1','cux2':'Cux2',
}

LOLLIPOP_GROUPS = [
    ((0,  9),  'Deep\nmarkers',             C_DEEP),
    ((10, 14), 'Interneuron\nmarkers',      C_INT),
    ((15, 17), 'Controversy',               C_CON),
    ((18, 22), 'Superficial\nmarkers',      C_SUP),
]

_DEEP_GENES = {'tshz3','sox5','fezf2','foxp2','crym','bcl11b','etv1','syt6','tle4','tbr1'}
_INT_GENES  = {'vip','sst','pvalb','reln','lhx6'}
_CON_GENES  = {'prss12','pou3f3','satb2'}

def _dot_color(gene, qval, tau):
    if gene in _CON_GENES:  return C_CON
    if gene in _INT_GENES:  return C_INT
    if gene in _DEEP_GENES: return C_DEEP
    return C_SUP


def _fmt_q(q):
    if q >= 0.05:  return 'ns'
    if q < 0.001:
        exp = int(np.floor(np.log10(q)))
        mant = q / 10**exp
        return f'$q$={mant:.1f}×10$^{{{exp}}}$'
    return f'$q$={q:.3f}'


# ── Hex-patch spatial plot ─────────────────────────────────────────────────────
def _hex_scatter(ax, coords, values, cmap='RdBu_r', vmin=None, vmax=None):
    """
    Draw Visium spots as flat-top hexagonal patches.
    Radius is estimated from the median nearest-neighbor distance.
    """
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn_dist = np.median(dists[:, 1])
    radius = nn_dist / np.sqrt(3) * 0.92   # slight gap between hexes

    if vmin is None or vmax is None:
        vmin = np.percentile(values, 2)
        vmax = np.percentile(values, 98)

    patches = [
        RegularPolygon((x, y), numVertices=6, radius=radius,
                       orientation=np.pi / 6)   # flat-top
        for x, y in coords
    ]
    col = PatchCollection(patches, array=values, cmap=cmap,
                          clim=(vmin, vmax), linewidths=0)
    ax.add_collection(col)
    ax.set_xlim(coords[:, 0].min() - radius, coords[:, 0].max() + radius)
    ax.set_ylim(coords[:, 1].min() - radius, coords[:, 1].max() + radius)
    return col


# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...')
plm = pd.read_csv('./results_raw/plm_results0.csv').set_index('gene')
cmp = pd.read_csv('./results_raw/mb_pergene_allmethods.csv').set_index('gene')

with open('./results_raw/tangram_pseudo0.pkl', 'rb') as f:
    pseudo = pickle.load(f)

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[adata_st.obs.cluster.isin(['Cortex_1', 'Cortex_3'])].copy()
adata_obs = adata_st.copy()
adata_obs.var_names = adata_obs.var_names.str.lower()
sc.pp.normalize_total(adata_obs, target_sum=1e4)
sc.pp.log1p(adata_obs)

spot_names  = list(pseudo.obs_names)
pseudo_g    = list(pseudo.var_names)
g2col       = {g: i for i, g in enumerate(pseudo_g)}
st_obs_n    = list(adata_st.obs_names)
aligned     = [s for s in spot_names if s in set(st_obs_n)]
pseudo_row  = [spot_names.index(s) for s in aligned]
st_row      = [st_obs_n.index(s) for s in aligned]
coords      = adata_st.obsm['spatial'][st_row].copy()
coords[:, 1] = -coords[:, 1]   # flip y
layer_lab   = adata_st.obs['cluster'].values[st_row]
X_pseudo    = np.asarray(pseudo.X).astype(np.float32)[pseudo_row, :]
A_aligned   = (layer_lab == 'Cortex_3').astype(int)   # 1 = superficial

print('Data loaded.')

# ── Image overlay data (Panel A) ───────────────────────────────────────────────
_lib         = list(adata_st.uns['spatial'].keys())[0]
_sf          = adata_st.uns['spatial'][_lib]['scalefactors']['tissue_hires_scalef']
_spd         = adata_st.uns['spatial'][_lib]['scalefactors']['spot_diameter_fullres']
img_hires    = adata_st.uns['spatial'][_lib]['images']['hires']
coords_hires = adata_st.obsm['spatial'][st_row] * _sf   # (N,2) col,row in hires px
spot_r_img   = _spd / 2 * _sf * 0.92

# ── Figure scaffold ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 7.20, 9.00
fig = plt.figure(figsize=(FIG_W, FIG_H))

gs_outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[2.2, 0.09, 3.2],
    hspace=0.0,
    left=0.09, right=0.97, top=0.97, bottom=0.07,
)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: tissue section (A) + summary barplot (B) + lollipop (C)
# ══════════════════════════════════════════════════════════════════════════════
gs_row1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0],
    width_ratios=[1, 3.2], wspace=0.28,
)
gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_row1[0],
    height_ratios=[1.9, 1.1], hspace=0.35,
)

# ── Panel A: tissue image + treatment overlay ──────────────────────────────────
ax_A = fig.add_subplot(gs_left[0])
col_spot = np.where(layer_lab == 'Cortex_3', C_SUP, C_DEEP)

ax_A.imshow(img_hires, origin='upper', aspect='equal')

patches_A = [RegularPolygon((x, y), numVertices=6, radius=spot_r_img,
                             orientation=np.pi / 6)
             for x, y in coords_hires]
col_A = PatchCollection(patches_A, facecolor=col_spot, linewidths=0, alpha=0.60)
ax_A.add_collection(col_A)

# Zoom to the annotated spots with a small margin
_pad = spot_r_img * 3
ax_A.set_xlim(coords_hires[:, 0].min() - _pad, coords_hires[:, 0].max() + _pad)
ax_A.set_ylim(coords_hires[:, 1].max() + _pad, coords_hires[:, 1].min() - _pad)  # y inverted
ax_A.set_aspect('equal', adjustable='box')
ax_A.set_anchor('N')   # anchor image to top of allocated cell
ax_A.axis('off')

leg_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_SUP,  markersize=6, label='Superficial ($A=1$)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DEEP, markersize=6, label='Deep ($A=0$)'),
]
ax_A.legend(handles=leg_handles, loc='upper center',
            bbox_to_anchor=(0.5, -0.04), fontsize=6,
            frameon=False, handletextpad=0.3, borderpad=0, ncol=1)
ax_A.set_title('Cortical zone annotation', loc='left', fontsize=7,
               fontweight='normal', pad=4)
ax_A.text(-0.23, 1.06, 'A', transform=ax_A.transAxes,
          fontsize=9, fontweight='bold', va='bottom', clip_on=False)


# ── Panel B: effect-size lollipop ─────────────────────────────────────────────
# ── Panel B: method comparison barplot ────────────────────────────────────────
ax_B = fig.add_subplot(gs_left[1])

_summ = pd.read_csv('./results_raw/mb_method_comparison.csv')
_morder  = ['tidest', 'SpatialGEE', 't-test', 'DESpace', 'SpaGCN']
_mlabels = ['TIDEST', 'SpatialGEE', 't-test', 'DESpace', 'SpaGCN']
_summ = _summ.set_index('method').loc[_morder]

_y   = np.arange(len(_morder))
_h   = 0.32
_col1 = '#BBBBBB'   # n_sig bar
_col2 = [MC['tidest'], MC['spatialgee'], MC['ttest'], MC['despace'], MC['spagcn']]

ax_B.barh(_y + _h/2, _summ['n_sig'],        height=_h, color=_col1,     alpha=0.7, label='Sig. / 55')
ax_B.barh(_y - _h/2, _summ['n_sig_correct'], height=_h, color=_col2,     alpha=0.9, label='Correct / 38')

ax_B.set_yticks(_y)
ax_B.set_yticklabels(_mlabels, fontsize=6)
ax_B.set_xlabel('Count', fontsize=6)
ax_B.set_xlim(0, 55)
ax_B.axvline(38, color='#999999', lw=0.5, ls=':', alpha=0.8)
ax_B.spines[['top', 'right']].set_visible(False)
ax_B.tick_params(axis='both', labelsize=5.5)
ax_B.legend(fontsize=5, loc='upper right', frameon=False,
            handlelength=1, handletextpad=0.3, labelspacing=0.2)
ax_B.text(-0.23, 1.08, 'B', transform=ax_B.transAxes,
          fontsize=9, fontweight='bold', va='bottom', clip_on=False)

# ── Panel C: effect-size lollipop ─────────────────────────────────────────────
ax_C = fig.add_subplot(gs_row1[1])

N = len(LOLLIPOP_GENES)
y_pos = np.arange(N)

for i, g in enumerate(LOLLIPOP_GENES):
    if g not in plm.index:
        continue
    r    = plm.loc[g]
    tau  = r['tau'];  se = r['se'];  ci = 1.96 * se;  qval = r['qval']
    col  = _dot_color(g, qval, tau)
    sig  = qval < 0.05
    ms   = 35

    ax_C.plot([tau - ci, tau + ci], [i, i],
              color=col, lw=1.0, solid_capstyle='round', zorder=2, alpha=0.9)
    ax_C.scatter([tau], [i], color=col, zorder=5, s=ms, marker='o',
                 edgecolors='none', linewidths=0)

    ax_C.text(0.285, i, _fmt_q(qval), va='center', ha='left', fontsize=5.5,
              color=col)


# Subtle vertical grid
ax_C.xaxis.grid(True, alpha=0.12, lw=0.5, zorder=0)
ax_C.axvline(0, color='#333333', lw=0.5, ls='--', alpha=0.5, zorder=1)
ax_C.set_axisbelow(True)

# y-axis gene labels
ax_C.set_yticks(y_pos)
ylabels = [GENE_DISPLAY.get(g, g) for g in LOLLIPOP_GENES]
ax_C.set_yticklabels(ylabels, fontsize=6.5, style='italic')
for tick, g in zip(ax_C.get_yticklabels(), LOLLIPOP_GENES):
    if g not in plm.index:
        continue
    r = plm.loc[g]
    tick.set_color(_dot_color(g, r['qval'], r['tau']))
    if g in BOLD_GENES:
        tick.set_fontweight('bold')


leg_C_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_SUP,  markersize=6, label='Superficial markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DEEP, markersize=6, label='Deep markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_INT,  markersize=6, label='Interneuron markers'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_CON,  markersize=6, label='Ambiguous'),
]
ax_C.legend(handles=leg_C_handles, loc='lower right',
            bbox_to_anchor=(0.855, 0.0), fontsize=6,
            frameon=True, framealpha=0.9, edgecolor='#cccccc',
            handletextpad=0.3, labelspacing=0.3, borderpad=0.5, ncol=1)

ax_C.set_xlabel('Estimated effect  $\\hat{\\tau}$ ± 95% CI\n(positive = higher in superficial)',
                fontsize=7)
ax_C.set_xlim(-0.315, 0.38)
ax_C.set_ylim(-0.7, N - 0.3)
ax_C.spines['top'].set_visible(False)
ax_C.spines['right'].set_visible(False)
ax_C.tick_params(left=False)
ax_C.spines['left'].set_visible(False)

ax_C.text(-0.10, 1.02, 'C', transform=ax_C.transAxes,
          fontsize=9, fontweight='bold', va='top')
ax_C.text(0.5, 1.01, 'TIDEST estimated effects — selected genes',
          transform=ax_C.transAxes, fontsize=7, va='bottom', ha='center')


# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: spatial expression maps (hex patches)
# ══════════════════════════════════════════════════════════════════════════════
gs_CD = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs_outer[2],
    height_ratios=[1.3, 1.8], hspace=-0.18, wspace=0.40,
)

MAP_META = {
    'sox5':  dict(title='Sox5',  col=C_DEEP),
    'tshz3': dict(title='Tshz3', col=C_DEEP),
    'vip':   dict(title='Vip',   col=C_INT),
    'satb2': dict(title='Satb2', col=C_CON),
}


def _map_subtitle(g):
    r     = plm.loc[g]
    tau, q = r['tau'], r['qval']
    tau_s  = f'$\\hat{{\\tau}}$={tau:+.3f}'
    q_s    = f'$q$={q:.1e}' if q < 0.01 else (f'$q$={q:.2f} ns' if q >= 0.05 else f'$q$={q:.3f}')
    return f'TIDEST: {tau_s}, {q_s}'


for idx, g in enumerate(FOCUS_GENES):
    ax = fig.add_subplot(gs_CD[0, idx])
    expr = X_pseudo[:, g2col[g]]

    col_map = _hex_scatter(ax, coords, expr, cmap='viridis', vmin=None, vmax=None)

    # Compact colorbar below
    cbar = plt.colorbar(col_map, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.02, aspect=18)
    cbar.set_label('Augmented outcome', fontsize=5.5, labelpad=1)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    meta = MAP_META[g]
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 1.08, f'$\\it{{{meta["title"]}}}$',
            transform=ax.transAxes, fontsize=7.5, fontweight='bold',
            ha='center', va='bottom', color=meta['col'])
    ax.text(0.5, 1.01, _map_subtitle(g),
            transform=ax.transAxes, fontsize=5.0,
            ha='center', va='bottom', color='#444444', linespacing=1.35)

    if idx == 0:
        ax.text(-0.23, 1.34, 'D', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', clip_on=False)
        ax.text(0.0, 1.34, 'Spatial augmented outcomes — selected genes',
                transform=ax.transAxes, fontsize=7, va='top')


# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: violin + method comparison
# ══════════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(0)


def _violin_panel(ax, g, row3_idx, col, title_extra=''):
    """Draw violin + strip plot for gene g. Returns axis."""
    expr   = X_pseudo[:, g2col[g]]
    deep_e = expr[A_aligned == 0]
    sup_e  = expr[A_aligned == 1]

    parts = ax.violinplot([deep_e, sup_e], positions=[0, 1],
                          showmedians=False, showextrema=False, widths=0.62)
    for pc, c in zip(parts['bodies'], [C_DEEP, C_SUP]):
        pc.set_facecolor(c);  pc.set_alpha(0.65)
        pc.set_edgecolor('white');  pc.set_linewidth(0.4)

    # Manual median + IQR
    for xi, grp, c in zip([0, 1], [deep_e, sup_e], [C_DEEP, C_SUP]):
        med = np.median(grp)
        q25, q75 = np.percentile(grp, [25, 75])
        ax.plot([xi, xi], [q25, q75], color='white', lw=1.6, zorder=6)
        ax.scatter([xi], [med], color='white', s=20, zorder=7, linewidths=0)

        jit = rng.uniform(-0.14, 0.14, len(grp))
        ax.scatter(xi + jit, grp, s=2, alpha=0.28, color=c, zorder=4, linewidths=0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Deep', 'Superficial'], fontsize=6.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if row3_idx == 0:
        ax.set_ylabel('Augmented outcome', fontsize=6.5)
    return ax


def _method_note(ax, g, note_keys=('ttest', 'spagcn', 'despace', 'spatialgee')):
    """Add multi-method q-value annotation below violin for gene g."""
    lines = []
    for m in note_keys:
        qcol = f'qval_{m}'
        if m == 'spagcn':
            qcol = 'qval_spagcn'
        q = cmp.loc[g, qcol]
        label = METHOD_LABELS[m]
        flag = ' ←' if q < 0.05 else ''
        lines.append((label, q, MC[m], flag))

    note_y = -0.26
    for i_l, (label, q, color, flag) in enumerate(lines):
        q_s = f'{q:.3f}' if q >= 0.001 else f'{q:.2e}'.replace('e-0', '×10⁻').replace('e-', '×10⁻')
        ax.text(0.5, note_y - i_l * 0.075,
                f'{label}: $q$={q_s}{flag}',
                transform=ax.transAxes, fontsize=5.2, ha='center', va='top',
                color=color, style='italic', linespacing=1.2)


# ── D1: Sox5 ──────────────────────────────────────────────────────────────────
ax_D1 = fig.add_subplot(gs_CD[1, 0])
_violin_panel(ax_D1, 'sox5', 0, C_DEEP)
_method_note(ax_D1, 'sox5')

# ── D2: Tshz3 ─────────────────────────────────────────────────────────────────
ax_D2 = fig.add_subplot(gs_CD[1, 1])
_violin_panel(ax_D2, 'tshz3', 1, C_DEEP)
_method_note(ax_D2, 'tshz3')

# ── D3: Vip ───────────────────────────────────────────────────────────────────
ax_D3 = fig.add_subplot(gs_CD[1, 2])
_violin_panel(ax_D3, 'vip', 2, C_INT)
_method_note(ax_D3, 'vip')

# ── D4: Satb2 ─────────────────────────────────────────────────────────────────
ax_D4 = fig.add_subplot(gs_CD[1, 3])
_violin_panel(ax_D4, 'satb2', 3, C_CON)
_method_note(ax_D4, 'satb2')

# ── Save ──────────────────────────────────────────────────────────────────────
for ext, dpi in [('pdf', None), ('png', 300)]:
    path = f'./results_raw/mb_main_figure.{ext}'
    kw = {'dpi': dpi} if dpi else {}
    fig.savefig(path, bbox_inches='tight', **kw)
    print(f'Saved {path}')

plt.close()
