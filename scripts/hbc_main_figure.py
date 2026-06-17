"""
Main figure for HBC (DCIS vs IBC) application — Nature Methods style.

Layout:
  Row 1 (tall):  [A] tissue map (IBC/DCIS cells)  |  [B] Lollipop of all 68 genes
  Row 2 (short): [C] method comparison barplot  |  [D] 2×2 spatial maps
                 [E] S100A4 PLM vs naive  |  [F] EMT initiator vs maintainers

Six stories:
  A  — tissue section showing IBC vs DCIS spatial layout
  B  — validation + overview of full extended panel
  C  — TIDEST beats competitors on in-panel genes (20/25 vs 19/25 vs 18/25 vs …)
  D  — spatial expression maps for key genes
  E  — S100A4: t-test inverted, PLM correct (spatial CAF confounding)
  F  — SNAI1 null, all downstream EMT effectors significant
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import spatialdata as sd

warnings.filterwarnings('ignore')

# ── Global typography (Nature Methods style, matching MB figure) ──────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         10,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10.5,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'lines.linewidth':   0.9,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── Paths ──────────────────────────────────────────────────────────────────
ZARR_PATH  = './Xenium.zarr'
ANNOT_PATH = './Xenium'
PLM_CSV    = './results_raw/hbc_plm_results0.csv'
COMP_CSV   = './results_raw/hbc_method_comparison.csv'
PSEUDO_PKL  = './results_raw/hbc_pseudo0.pkl'
CELLPLM_PKL = './results_raw/hbc_cellplm_pred0.pkl'
SPCA_SPOTS  = './results_raw/hbc_spca_spots0.txt'
OUT_PATH   = './results_raw/hbc_main_figure.pdf'

IBC_CLUSTERS  = ['Invasive_Tumor', 'Prolif_Invasive_Tumor']
DCIS_CLUSTERS = ['DCIS_1', 'DCIS_2']
ALL_CLUSTERS  = IBC_CLUSTERS + DCIS_CLUSTERS

# ── Palette ────────────────────────────────────────────────────────────────
C_IBC    = '#c0392b'
C_DCIS   = '#2980b9'
C_NS     = '#cccccc'

# Biological group colors (lollipop)
GROUP_COLORS = {
    'Luminal / DCIS':    '#2166ac',
    'Myoepithelial':     '#762a83',
    'Proliferation':     '#d73027',
    'EMT':               '#f4a582',
    'IBC stroma':        '#8c510a',
    'Immune':            '#1b7837',
    'Other':             '#888888',
}

def gene_group(g):
    g = g.lower()
    luminal  = {'tff3','tff1','agr2','esr1','areg','pgr','mlph','spdef',
                'krt8','krt18','krt19','cdh1','cldn3','cldn4','cldn7','runx1','erbb4','kit'}
    mepith   = {'krt5','krt14','egfr','mylk','tp63','cnn1','cdh3','itgb6',
                'lamb3','lamc2','col4a1'}
    prolif   = {'mki67','top2a','cdk1','ccnb1','mcm6','pcna','mcm2','ezh2','cdkn2a','aldh1a1'}
    emt      = {'cdh2','vim','fn1','twist1','snai1','snai2','zeb1','zeb2',
                'mmp2','mmp9','mmp11','cxcr4','s100a4','cd44'}
    stroma   = {'postn','thbs2','grem1','col11a1','erbb2','s100a8','s100a9'}
    immune   = {'nkg7','tigit','ctla4','gnly','tyrobp','cxcl10','ca9'}
    if g in luminal: return 'Luminal / DCIS'
    if g in mepith:  return 'Myoepithelial'
    if g in prolif:  return 'Proliferation'
    if g in emt:     return 'EMT'
    if g in stroma:  return 'IBC stroma'
    if g in immune:  return 'Immune'
    return 'Other'

# Genes to label in lollipop
LABEL_GENES = {
    'tff3','tff1','areg','esr1','agr2','pgr',
    'itgb6','erbb4','krt14','krt5','mylk','tp63','lamb3','lamc2','cdh3','cnn1',
    'erbb2','top2a','mki67','ezh2','cdkn2a',
    'cdh2','snai1','vim','fn1',
    'postn','thbs2','col11a1',
    'nkg7','s100a4','cxcl10',
}

# EMT story genes (panel D)
EMT_GENES = ['snai1', 'cdh2', 'vim', 'fn1', 'twist1', 'snai2', 'zeb1', 'zeb2']
EMT_LABELS = {
    'snai1': 'SNAI1',
    'cdh2':  'CDH2', 'vim': 'VIM', 'fn1': 'FN1',
    'twist1':'TWIST1','snai2':'SNAI2','zeb1':'ZEB1','zeb2':'ZEB2',
}


# ═══════════════════════════════════════════════════════════════════════════
print('Loading PLM results...')
plm = pd.read_csv(PLM_CSV).set_index('gene')
plm['group'] = [gene_group(g) for g in plm.index]
plm['color'] = [GROUP_COLORS[gene_group(g)] for g in plm.index]
plm['sig']   = plm['qval'] < 0.05
plm_sorted   = plm.sort_values('tau')   # most negative (DCIS) → most positive (IBC)

# ═══════════════════════════════════════════════════════════════════════════
print('Loading Xenium data for spatial maps and violin...')
sdata   = sd.read_zarr(ZARR_PATH)
st_data = sdata.tables['table']

annotations_st = pd.read_excel(
    pathlib.Path(ANNOT_PATH) / 'Cell_Barcode_Type_Matrices.xlsx',
    sheet_name='Xenium R1 Fig1-5 (supervised)',
)
st_data.obs = st_data.obs.merge(
    annotations_st.set_index('Barcode').loc[
        [int(x) + 1 for x in st_data.obs_names], 'Cluster'
    ],
    left_on='cell_id', right_index=True, how='left',
)

st_plot = st_data[st_data.obs['Cluster'].isin(ALL_CLUSTERS)].copy()
st_plot.obs['group'] = st_plot.obs['Cluster'].replace(
    {'Prolif_Invasive_Tumor': 'Invasive_Tumor', 'DCIS_2': 'DCIS_1'}
)
sc.pp.normalize_total(st_plot, target_sum=1e4)
sc.pp.log1p(st_plot)

spatial  = st_plot.obsm['spatial']
x_coord  = spatial[:, 0]
y_coord  = spatial[:, 1]
is_ibc   = (st_plot.obs['group'].values == 'Invasive_Tumor')
is_dcis  = (st_plot.obs['group'].values == 'DCIS_1')

panel_lower   = {g.lower(): g for g in st_plot.var_names}

def get_expr(gene_sym):
    var = panel_lower.get(gene_sym.lower())
    if var is None:
        return None
    col = st_plot[:, var].X
    return col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()

# ═══════════════════════════════════════════════════════════════════════════
print('Loading pseudo-outcome for S100A4 panel...')
with open(PSEUDO_PKL, 'rb') as f:
    pseudo = pickle.load(f)
spca_spots = pd.read_csv(SPCA_SPOTS, header=None)[0].astype(str).tolist()
pseudo.obs.index = pseudo.obs.index.astype(str)
spot_order = [s for s in spca_spots if s in pseudo.obs.index]
pseudo_sub = pseudo[spot_order]
pseudo_sub.obs['group'] = pseudo_sub.obs['Cluster'].replace(
    {'Prolif_Invasive_Tumor': 'Invasive_Tumor', 'DCIS_2': 'DCIS_1'}
)
pseudo_group  = pseudo_sub.obs['group'].values

# Align pseudo_sub to st_plot cells (some pseudo_sub cells may be absent from st_plot)
_ps_ids    = pseudo_sub.obs.index.astype(str)
_sp_cellidx = {str(v): i for i, v in enumerate(st_plot.obs['cell_id'].values)}
_keep_mask  = np.array([cid in _sp_cellidx for cid in _ps_ids])
_ps_aligned = pseudo_sub[_keep_mask]
_sp_rows    = [_sp_cellidx[cid] for cid in _ps_ids[_keep_mask]]
pseudo_group = _ps_aligned.obs['group'].values  # re-define aligned group

# Augmented outcome (CellPLM prediction + Pearson correction)
s100a4_aug = np.asarray(_ps_aligned[:, 's100a4'].X).ravel()

# Observed (raw Xenium log-CPM): same cells from st_plot
_col_s100a4 = st_plot.var_names.get_loc('S100A4')
_X_st       = st_plot.X.toarray() if issparse(st_plot.X) else np.asarray(st_plot.X)
s100a4_obs  = _X_st[_sp_rows, _col_s100a4]

# CellPLM raw prediction (before Pearson correction)
import io as _io, torch as _torch
class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: _torch.load(_io.BytesIO(b), map_location='cpu')
        return super().find_class(mod, name)
with open(CELLPLM_PKL, 'rb') as f:
    _cellplm = _CpuUnpickler(f).load()
_cellplm.obs.index = _cellplm.obs.index.astype(str)
_c_mask  = _cellplm.var['gene_symbol'].str.lower() == 's100a4'
_c_idx   = int(np.where(_c_mask.values)[0][0])
_Xc      = _cellplm.X
_all_pred = (_Xc[:, _c_idx].toarray().ravel() if issparse(_Xc)
             else np.asarray(_Xc[:, _c_idx]).ravel())
_pred_map = dict(zip(_cellplm.obs.index, _all_pred))
s100a4_pred = np.array([_pred_map.get(s, np.nan) for s in _ps_ids[_keep_mask]])

# ═══════════════════════════════════════════════════════════════════════════
print('Loading method comparison...')
# Load per-method summary; fallback to None if not yet generated
try:
    comp_df = pd.read_csv(COMP_CSV, index_col=0)
    has_comp = True
except FileNotFoundError:
    has_comp = False
    comp_df  = None

print('Building figure...')
fig = plt.figure(figsize=(22, 18), dpi=150)
fig.patch.set_facecolor('white')

gs_main = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.7, 1],
    hspace=0.20,
)

# ── Row 1: left column (A tissue + C method comparison) | right (B lollipop) ─
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_main[0],
    wspace=0.08,
    width_ratios=[1.5, 3.2],
)
gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_top[0],
    height_ratios=[1.9, 1.1],
    hspace=0.35,
)
ax_tissue   = fig.add_subplot(gs_left[0])   # A: tissue map
ax_comp     = fig.add_subplot(gs_left[1])   # C: method comparison barplot
ax_lollipop = fig.add_subplot(gs_top[1])    # B: lollipop

# ── Row 2: three panels (D spatial maps | E S100A4 | F EMT) ──────────────
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_main[1],
    wspace=0.30,
    width_ratios=[2.0, 0.7, 1.1],
)

gs_spatial = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_bot[0],
    hspace=0.18, wspace=0.08,
)
ax_sp = [[fig.add_subplot(gs_spatial[r, c]) for c in range(2)] for r in range(2)]
# shift all 4 spatial-map axes down ~0.2 cm (0.2/2.54/18 ≈ 0.0044 figure fraction)

ax_s100a4 = fig.add_subplot(gs_bot[1])
ax_emt    = fig.add_subplot(gs_bot[2])


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Tissue map: IBC vs DCIS spatial distribution
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_tissue

# Plot DCIS first (background), IBC on top
ax.scatter(x_coord[is_dcis], y_coord[is_dcis],
           c=C_DCIS, s=0.25, alpha=0.6, linewidths=0, rasterized=True, zorder=1,
           label=f'DCIS (A=0, n={is_dcis.sum():,})')
ax.scatter(x_coord[is_ibc],  y_coord[is_ibc],
           c=C_IBC,  s=0.25, alpha=0.6, linewidths=0, rasterized=True, zorder=2,
           label=f'IBC (A=1, n={is_ibc.sum():,})')

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A   Xenium breast tissue section (IBC vs DCIS)', loc='left',
             fontsize=13, fontweight='bold', pad=4)

legend_handles = [
    plt.scatter([], [], c=C_IBC,  s=18, label='IBC  (A=1)'),
    plt.scatter([], [], c=C_DCIS, s=18, label='DCIS (A=0)'),
]
ax.legend(handles=legend_handles, fontsize=10.5, loc='upper center',
          bbox_to_anchor=(0.5, 0.02), framealpha=0.0, markerscale=1.5,
          ncol=1, handletextpad=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Lollipop
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_lollipop
# Only plot genes that have a label
plm_labeled  = plm_sorted[plm_sorted.index.isin(LABEL_GENES)]
genes  = list(plm_labeled.index)
y_pos  = np.arange(len(genes))
taus   = plm_labeled['tau'].values
colors = plm_labeled['color'].values
sigs   = plm_labeled['sig'].values

for i, (tau, col, sig, g) in enumerate(zip(taus, colors, sigs, genes)):
    alpha = 1.0 if sig else 0.35
    lw    = 0.8 if sig else 0.4
    ax.plot([0, tau], [i, i], color=col, alpha=alpha, lw=lw, solid_capstyle='round')
    mfc = col if sig else 'white'
    ax.plot(tau, i, 'o', color=col, mfc=mfc, ms=5 if sig else 3.5,
            alpha=alpha, mew=1.2, zorder=5)
    offset = 0.008 if tau >= 0 else -0.008
    ha     = 'left'  if tau >= 0 else 'right'
    ax.text(tau + offset, i, g.upper(), va='center', ha=ha,
            fontsize=12, color='#222222',
            fontweight='normal')

ax.set_yticks([])
ax.set_xlabel(r'Estimated effect ($\hat{\tau}$)', fontsize=13)
ax.tick_params(axis='x', labelsize=12)
ax.set_title(r'C   TIDEST estimated effects — selected genes', loc='left',
             fontsize=12, fontweight='bold', pad=6)

ax.axvline(0, color='#333333', lw=0.5, ls='--', alpha=0.5, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=12)

tau_abs_max = plm_labeled['tau'].abs().max()

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', mfc=c, ms=7, label=grp)
    for grp, c in GROUP_COLORS.items()
]
ax.legend(handles=legend_elements, loc='center left', fontsize=12.5,
          framealpha=0.9, ncol=2, columnspacing=1)

# Axis direction labels stacked on the left
ax.text(0.02, 0.98, '← DCIS-enriched', va='top', ha='left',
        fontsize=10, color=C_DCIS, fontweight='bold', transform=ax.transAxes)
ax.text(0.02, 0.92, 'IBC-enriched →', va='top', ha='left',
        fontsize=10, color=C_IBC,  fontweight='bold', transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Method comparison barplot
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_comp
METHOD_ORDER = ['TIDEST', 't-test', 'SpaGCN', 'DESpace', 'SpatialGEE']
METHOD_LABELS = ['TIDEST', 't-test', 'SpaGCN\n(Wilcoxon)', 'DESpace', 'SpatialGEE']
M_COLORS = {
    'TIDEST':     '#EE6677',
    't-test':     '#4477AA',
    'SpaGCN':     '#228833',
    'DESpace':    '#CCBB44',
    'SpatialGEE': '#AA3377',
}

if has_comp and all(m in comp_df.index for m in METHOD_ORDER):
    n_known = int(comp_df.loc['TIDEST', 'n_known'])
    n_genes = int(comp_df.loc['TIDEST', 'n_genes'])
    sig_vals  = [int(comp_df.loc[m, 'n_sig'])         for m in METHOD_ORDER]
    corr_vals = [int(comp_df.loc[m, 'n_sig_correct'])  for m in METHOD_ORDER]
    y = np.arange(len(METHOD_ORDER))
    h = 0.33
    # Horizontal bars: grey = sig / n_genes, colored = correct / n_known
    ax.barh(y + h/2, sig_vals,  h, color='#cccccc',
            label=f'Sig (q<0.05) / {n_genes}')
    for m, ypos, cv in zip(METHOD_ORDER, y, corr_vals):
        ax.barh(ypos - h/2, cv, h, color=M_COLORS[m], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_LABELS, fontsize=12)
    ax.set_xlabel('Count', fontsize=12)
    ax.axvline(0, color='#444444', lw=0.6)
    ax.axvline(n_known, color='#999999', lw=0.5, ls=':', alpha=0.8)
    ax.set_xlim(0, max(sig_vals) * 1.35)
    legend_patches = [
        Patch(facecolor='#cccccc',           label=f'Sig (q<0.05) / {n_genes}'),
        Patch(facecolor=M_COLORS['TIDEST'],  alpha=0.9, label=f'Correct dir. / {n_known}'),
    ]
    ax.legend(handles=legend_patches, fontsize=11, loc='upper right', frameon=False)
else:
    ax.text(0.5, 0.5, 'Run hbc_method_comparison.py\nto generate comparison data',
            ha='center', va='center', transform=ax.transAxes, fontsize=9.5, color='#888888')
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=12)

ax.set_title('B   Method comparison\n(30 in-panel genes)', loc='left',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=11)


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — Spatial maps (2×2): ESR1, ERBB2, KRT14, S100A4
# ═══════════════════════════════════════════════════════════════════════════
SPATIAL_GENES = [
    ('ESR1',  C_DCIS),
    ('ERBB2', C_IBC),
    ('KRT14', C_DCIS),
    ('S100A4', C_IBC),
]

def _map_subtitle(gene):
    g = gene.lower()
    if g not in plm.index:
        return ''
    r   = plm.loc[g]
    tau, q = r['tau'], r['qval']
    tau_s = f'$\\hat{{\\tau}}$={tau:+.3f}'
    q_s   = '$q\\approx0$' if q == 0 else (f'$q$={q:.1e}' if q < 0.01 else f'$q$={q:.3f}')
    return f'TIDEST: {tau_s}, {q_s}'

vmin_pct, vmax_pct = 20, 98

for idx, (gene, tcol) in enumerate(SPATIAL_GENES):
    r, c = divmod(idx, 2)
    axs = ax_sp[r][c]

    expr = get_expr(gene)
    if expr is None:
        axs.text(0.5, 0.5, f'{gene}\nnot in panel', ha='center', va='center',
                 transform=axs.transAxes, fontsize=11)
        axs.axis('off')
        continue

    vmin = np.percentile(expr, vmin_pct)
    vmax = np.percentile(expr, vmax_pct)

    sc_plot = None
    for mask, zorder in [(is_dcis, 1), (is_ibc, 2)]:
        sc_plot = axs.scatter(
            x_coord[mask], y_coord[mask],
            c=expr[mask], cmap='viridis',
            vmin=vmin, vmax=vmax,
            s=0.3, alpha=0.7, linewidths=0, rasterized=True,
            zorder=zorder,
        )

    axs.set_aspect('equal')
    axs.axis('off')

    # Colorbar on right side (vertical)
    cbar = fig.colorbar(sc_plot, ax=axs, orientation='vertical',
                        fraction=0.06, pad=0.02, aspect=12)
    cbar.set_label('Augmented outcome', fontsize=10, labelpad=2, rotation=270, va='bottom')
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # Gene name: italic, bold, colored
    axs.text(0.5, 1.08, f'$\\it{{{gene}}}$',
             transform=axs.transAxes, fontsize=12, fontweight='bold',
             ha='center', va='bottom', color=tcol)
    # Subtitle: TIDEST τ̂ and q
    axs.text(0.5, 1.01, _map_subtitle(gene),
             transform=axs.transAxes, fontsize=10,
             ha='center', va='bottom', color='#444444', linespacing=1.35)

# Panel label above top-left spatial map
ax_sp[0][0].annotate('D   Spatial augmented outcomes — selected genes',
                      xy=(0, 1.18), xycoords='axes fraction',
                      fontsize=12, fontweight='bold', color='#222222')


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — S100A4: PLM vs naive
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_s100a4

# Split each quantity into IBC / DCIS
obs_ibc  = s100a4_obs[pseudo_group == 'Invasive_Tumor']
obs_dcis = s100a4_obs[pseudo_group == 'DCIS_1']
pred_ibc  = s100a4_pred[pseudo_group == 'Invasive_Tumor']
pred_dcis = s100a4_pred[pseudo_group == 'DCIS_1']
aug_ibc  = s100a4_aug[pseudo_group == 'Invasive_Tumor']
aug_dcis = s100a4_aug[pseudo_group == 'DCIS_1']

def violin_half(ax, data_ibc, data_dcis, x_center, color_ibc, color_dcis, width=0.28):
    for data, col, xoff in [(data_dcis, color_dcis, -width/2), (data_ibc, color_ibc, width/2)]:
        valid = data[~np.isnan(data)]
        if len(valid) < 4:
            continue
        parts = ax.violinplot([valid], positions=[x_center + xoff],
                              widths=width * 0.95, showmedians=True,
                              showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
            pc.set_edgecolor('#444444')
            pc.set_linewidth(0.5)
        parts['cmedians'].set_color('#222222')
        parts['cmedians'].set_linewidth(1.5)

violin_half(ax, obs_ibc,  obs_dcis,  x_center=0.5, color_ibc=C_IBC, color_dcis=C_DCIS)
violin_half(ax, pred_ibc, pred_dcis, x_center=2.0, color_ibc=C_IBC, color_dcis=C_DCIS)
violin_half(ax, aug_ibc,  aug_dcis,  x_center=3.5, color_ibc=C_IBC, color_dcis=C_DCIS)

# Direction annotations
_obs_dir  = 'IBC > DCIS' if obs_ibc.mean()  > obs_dcis.mean()  else 'DCIS > IBC'
_pred_dir = 'IBC > DCIS' if pred_ibc[~np.isnan(pred_ibc)].mean() > pred_dcis[~np.isnan(pred_dcis)].mean() else 'DCIS > IBC'

_obs_col  = '#27ae60' if _obs_dir == 'IBC > DCIS' else '#c0392b'
_pred_col = '#27ae60' if _pred_dir == 'IBC > DCIS' else '#c0392b'

y_top = max(np.nanpercentile(obs_ibc, 98), np.nanpercentile(obs_dcis, 98),
            np.nanpercentile(pred_ibc, 98), np.nanpercentile(pred_dcis, 98),
            np.percentile(aug_ibc, 98), np.percentile(aug_dcis, 98)) * 1.05

ax.text(3.1, y_top * 1.12, f't-test: {_obs_dir}\n(q=1×10⁻¹⁹)',
        ha='center', va='bottom', fontsize=10, color=_obs_col, fontweight='bold')
ax.text(3.1, y_top, 'TIDEST: IBC > DCIS\n(τ̂=+0.07, $q\\approx0$, ✓)',
        ha='center', va='bottom', fontsize=10, color='#27ae60', fontweight='bold')
ax.set_ylim(top=y_top * 1.5)

ax.set_xticks([0.5, 2.0, 3.5])
ax.set_xticklabels(['Observed\n(raw Xenium)', 'Predicted\n(CellPLM)', 'Augmented\noutcome'], fontsize=11)
ax.set_xlim(0, 4.0)
ax.set_ylabel('log-CPM expression', fontsize=12)
ax.set_title('E   S100A4: spatial CAF\nconfounding corrected by TIDEST',
             loc='left', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [
    Patch(facecolor=C_IBC,  alpha=0.65, label='Invasive Tumor (IBC)'),
    Patch(facecolor=C_DCIS, alpha=0.65, label='DCIS'),
]
ax.legend(handles=legend_handles, fontsize=10.5, loc='upper right', framealpha=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — EMT initiator vs maintainers
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_emt

emt_df = plm.loc[[g for g in EMT_GENES if g in plm.index]]
emt_taus = emt_df['tau'].values
emt_sigs = emt_df['sig'].values
emt_genes = list(emt_df.index)
y_emt = np.arange(len(emt_genes))

for i, (g, tau, sig) in enumerate(zip(emt_genes, emt_taus, emt_sigs)):
    is_initiator = (g == 'snai1')
    col   = '#888888' if is_initiator else (C_IBC if tau > 0 else C_DCIS)
    alpha = 0.9 if sig else 0.35

    ax.barh(i, tau, color=col, alpha=alpha, height=0.55,
            edgecolor='#444444', linewidth=0.4)

    # q-value annotation
    qval = emt_df.loc[g, 'qval']
    if sig:
        q_str = '$q\\approx0$' if qval == 0 else (f'q={qval:.0e}' if qval < 0.001 else f'q={qval:.3f}')
    else:
        q_str = 'n.s.'
    xpos   = tau + 0.004 if tau >= 0 else tau - 0.004
    ha_txt = 'left'     if tau >= 0 else 'right'
    ax.text(xpos, i, q_str, va='center', ha=ha_txt, fontsize=10,
            color='#444444')

ax.set_yticks(y_emt)
ax.set_yticklabels([EMT_LABELS.get(g, g.upper()) for g in emt_genes], fontsize=14)
ax.axvline(0, color='#444444', lw=0.8)

# Divider between initiator and maintainers
ax.axhline(0.5, color='#aaaaaa', lw=1, ls='--')
emt_xlim = max(abs(emt_taus)) * 1.6
ax.text(emt_xlim * 0.80, 0, 'Initiator', va='center', ha='left', fontsize=10.5,
        color='#666666', style='italic')
ax.text(emt_xlim * 0.80, 3.5, 'Effectors /\nmaintainers', va='center', ha='left',
        fontsize=10.5, color='#666666', style='italic')

ax.set_xlabel(r'Estimated effect ($\hat{\tau}$)', fontsize=12)
ax.set_title('F   EMT dynamics: SNAI1 initiator\nis not a sustained IBC marker',
             loc='left', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlim(-emt_xlim * 0.3, emt_xlim)


# ═══════════════════════════════════════════════════════════════════════════
# Fix lollipop x-limits after all other panels set
# ═══════════════════════════════════════════════════════════════════════════
tau_range = plm_labeled['tau'].abs().max()
ax_lollipop.set_xlim(-tau_range * 1.35, 0.7)
ax_lollipop.set_ylim(-1, len(genes))

# Shift bottom row (D, E, F) leftward by 1 cm
fig.canvas.draw()   # force layout computation
_dx   = -(1.0 / 2.54 / 22)
_dy_sp = -(0.2 / 2.54 / 18)   # shift spatial maps + per-plot titles down 0.2 cm
_bot_axes = [ax_sp[r][c] for r in range(2) for c in range(2)] + [ax_s100a4, ax_emt]

# Shift spatial-map axes down (per-plot titles follow automatically)
for _r in range(2):
    for _c in range(2):
        _p = ax_sp[_r][_c].get_position()
        ax_sp[_r][_c].set_position([_p.x0 + _dx, _p.y0 + _dy_sp, _p.width, _p.height])

# Shift E and F only leftward (no vertical shift)
for _ax in [ax_s100a4, ax_emt]:
    _p = _ax.get_position()
    _ax.set_position([_p.x0 + _dx, _p.y0, _p.width, _p.height])

# Compensate panel-D title upward so it stays in place
_ax0_h = ax_sp[0][0].get_position().height   # axes height in figure fraction
_title_compensate = -_dy_sp / _ax0_h         # in axes-fraction units (positive = up)
for _ann in ax_sp[0][0].get_children():
    if hasattr(_ann, 'get_text') and 'Spatial augmented' in str(_ann.get_text()):
        _xy = _ann.get_position()
        _ann.set_position((_xy[0], _xy[1] + _title_compensate))

# Shift colorbars (bottom-half axes not already moved)
_sp_axes_set = set(_bot_axes)
for _ax in fig.axes:
    if _ax not in _sp_axes_set and _ax != ax_lollipop and _ax != ax_tissue and _ax != ax_comp:
        _p = _ax.get_position()
        if _p.y1 < 0.5:
            _ax.set_position([_p.x0 + _dx, _p.y0 + _dy_sp, _p.width, _p.height])

# ═══════════════════════════════════════════════════════════════════════════
print(f'Saving to {OUT_PATH}...')
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Done.')
