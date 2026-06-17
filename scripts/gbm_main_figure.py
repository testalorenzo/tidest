"""
Main figure for GBM application — Nature Methods style.

Layout:
  Row 0 (tall, full width): [A] Strip/boxplot — all 90 genes, per-sample τ dots
                                  + RE pooled estimate, ordered by RE τ
  Rows 1-3: 3 balanced samples × (annotation | SNAP25 spatial | GFAP spatial)

Output: PDF
"""

import warnings, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mpl_colors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
from scipy.sparse import issparse
import scanpy as sc

warnings.filterwarnings('ignore')

META_CSV = './Inputs/general/visium_metadata.csv'
PLM_CSV  = './results_raw/gbm_plm_results_all.csv'
RE_CSV   = './results_raw/gbm_meta_results.csv'
CMP_SUMMARY_CSV = './results_raw/gbm_method_comparison_full_summary.csv'
CMP_PERGENE_CSV = './results_raw/gbm_method_comparison_full.csv'
OUT_PATH = './results_raw/gbm_main_figure.pdf'

# Method colors (consistent with MB/HBC/simulation comparison figures)
MC = {
    'tidest':     '#EE6677',
    'ttest':      '#4477AA',
    'spagcn':     '#228833',
    'despace':    '#CCBB44',
    'spatialgee': '#AA3377',
}
METHOD_LABELS = {
    'tidest': 'tidest', 'ttest': 't-test', 'spagcn': 'SpaGCN',
    'despace': 'DESpace', 'spatialgee': 'SpatialGEE',
}

SAMPLES = ['MGH258', 'UKF304', 'ZH881inf', 'UKF269', 'ZH881T1']
SHOW_GENES = ['snap25', 'gfap']   # spatial maps

# Genes mentioned in the results text — strip plot selection
TEXT_GENES = {
    'snap25', 'syt1', 'rbfox3', 'tubb3', 'l1cam', 'dcx', 'slc1a2',   # LE / neuronal
    'aqp4', 'gfap', 's100b', 'egfr', 'cd44',                          # CT / GBM core
    'ascl1', 'olig1', 'olig2', 'sox10', 'pdgfra',                     # OPC paradox
    'bcan', 'ptprz1', 'mog',                                           # ECM paradox
}

C_CT  = '#d73027'
C_LE  = '#2166ac'

# Cohort colors for per-sample dots
COHORT_COLOR = {'MGH': '#e6a817', 'UKF': '#4393c3', 'ZH': '#01665e'}
def cohort(s):
    if s.startswith('MGH'): return 'MGH'
    if s.startswith('UKF'): return 'UKF'
    return 'ZH'

GROUP_COLORS = {
    'Neuronal (LE)':      C_LE,
    'OPC/NPC (paradox)':  '#762a83',
    'GBM core (AC/MES)':  C_CT,
    'Proliferation':      '#b2182b',
    'Immune/vascular':    '#1b7837',
    'Hypoxia/metabolism': '#8c510a',
    'Invasion/ECM':       '#f46d43',
    'Other':              '#888888',
}

def gene_group(g):
    g = g.lower()
    neuronal = {'syt1','snap25','rbfox3','tubb3','dcx','nrxn1','dlg4','map2',
                'mbp','plp1','mag','mog','ckb','ldhb','ldha','l1cam','sparcl1','slc1a2'}
    paradox  = {'ascl1','bcan','ptprz1','olig1','olig2','sox4','ccnd2','pdgfra',
                'sox10','sox11','egr1','htra1','nrcam','sall1','olfml3','cx3cr1',
                'hexb','p2ry12','tmem119','itgb1'}
    gbm_core = {'gfap','aqp4','s100b','apoe','aldoc','slc1a2','glul','gja1',
                'chi3l1','vim','fn1','cd44','vcan','anxa2','tnc','tgfbi',
                'col1a1','serpine1'}
    prolif   = {'top2a','mki67','pcna','mcm2','mcm6','egfr','met'}
    immune   = {'aif1','cd68','pecam1','vwf','cldn5','spp1','lgals1','lgals3',
                'cd163','msr1','mrc1','postn'}
    hypoxia  = {'hif1a','vegfa','ca9','hk2','epas1','ldha','ldhb'}
    if g in neuronal: return 'Neuronal (LE)'
    if g in paradox:  return 'OPC/NPC (paradox)'
    if g in gbm_core: return 'GBM core (AC/MES)'
    if g in prolif:   return 'Proliferation'
    if g in immune:   return 'Immune/vascular'
    if g in hypoxia:  return 'Hypoxia/metabolism'
    return 'Other'

# ── Load data ─────────────────────────────────────────────────────────────
print('Loading data...')
meta_re = pd.read_csv(RE_CSV)
meta_full = meta_re.copy()   # unfiltered RE results — needed for the paradox-gene panel (F)
meta_re['group'] = meta_re['gene'].map(gene_group)
meta_re['sig']   = meta_re['qval_pool'] < 0.05
meta_re = meta_re[meta_re['gene'].isin(TEXT_GENES)]   # keep only text-cited genes
meta_re = meta_re.sort_values('tau_pool').reset_index(drop=True)   # order by RE τ

plm_all  = pd.read_csv(PLM_CSV)
meta_csv = pd.read_csv(META_CSV)
cmp_summary = pd.read_csv(CMP_SUMMARY_CSV).set_index('method')
cmp_pergene = pd.read_csv(CMP_PERGENE_CSV).set_index('gene')

# ── Load Visium ───────────────────────────────────────────────────────────
print('Loading Visium data...')
visium = {}
for sample in SAMPLES:
    adata = sc.read_visium(f'./Inputs/general/GBM_data/{sample}/outs/')
    adata.var_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.str.lower())
    meta_s = meta_csv[meta_csv['sample'] == sample].set_index('spot_id')
    ct_le  = meta_s[meta_s['ivygap'].isin(['CT', 'LE'])].index
    adata  = adata[adata.obs_names.intersection(ct_le)].copy()
    adata.obs['region'] = meta_s.loc[adata.obs_names, 'ivygap'].values
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    visium[sample] = adata

# ── Load augmented pseudo-outcomes (Y^pseudo, what the PLM is actually fit on) ──
print('Loading augmented pseudo-outcomes...')
pseudo = {}
for sample in SAMPLES:
    with open(f'./results_raw/gbm_pseudo_{sample}_0.pkl', 'rb') as fh:
        pseudo[sample] = pickle.load(fh)

# ── Figure layout ─────────────────────────────────────────────────────────
print('Building figure...')
n_genes = len(meta_re)   # 90

fig = plt.figure(figsize=(21.5, 18.306))
fig.patch.set_facecolor('white')

# Sample-column rows (now at the top of the figure, panels A/B/C): shifted
# ~2cm left and with ~0.5cm narrower gaps between columns than the strip-plot
# block below it.
gs_samples = gridspec.GridSpec(
    3, len(SAMPLES),
    figure=fig,
    height_ratios=[1.6, 1.6, 1.6],
    hspace=0.323, wspace=0.077,
    left=0.0734, right=0.9334, top=0.967, bottom=0.4096,
)

# Bottom block: strip plot (Panel D) + competitor comparison (Panel E). Made
# shorter than the original (less tall, tighter spacing between gene rows)
# while preserving the same gap below the sample-column grid above it.
gs_top = gridspec.GridSpec(
    1, 1, figure=fig,
    left=0.11, right=0.97, top=0.3346, bottom=0.03278,
)
gs_row0 = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_top[0, 0], width_ratios=[3.1, 0.05, 0.85, 0.0], wspace=0.32,
)
ax_strip = fig.add_subplot(gs_row0[0, 0])   # strip plot (left, wide)
ax_cmp   = fig.add_subplot(gs_row0[0, 2])   # method-comparison barplot (right, narrow)


# ══════════════════════════════════════════════════════════════════════════
# Panel D — strip plot: all 90 genes, per-sample τ + RE pooled estimate
# ══════════════════════════════════════════════════════════════════════════
ax = ax_strip

genes_ordered = list(meta_re['gene'])          # sorted by RE τ (ascending)
y_pos         = np.arange(n_genes)
plm_pivot     = plm_all.pivot(index='sample', columns='gene', values='tau')

DOT_COLOR = '#999999'

for i, gene in enumerate(genes_ordered):
    re_row  = meta_re[meta_re['gene'] == gene].iloc[0]
    sig     = re_row['sig']
    dir_col = C_CT if re_row['tau_pool'] < 0 else C_LE   # CT- vs LE-enriched

    # Per-sample dots
    for sname in plm_pivot.index:
        if gene not in plm_pivot.columns: continue
        tau_s = plm_pivot.loc[sname, gene]
        if np.isnan(tau_s): continue
        ax.scatter(tau_s, i, color=DOT_COLOR, s=18, alpha=0.8,
                   linewidths=0, zorder=3)

    # IQR box
    vals = plm_pivot[gene].dropna().values if gene in plm_pivot.columns else np.array([])
    if len(vals) >= 4:
        q25, q75 = np.percentile(vals, [25, 75])
        box_col = dir_col if sig else '#cccccc'
        ax.barh(i, q75 - q25, left=q25, height=0.45,
                color=box_col, alpha=0.25, zorder=2, linewidth=0)

    # RE pooled estimate
    tau_re = re_row['tau_pool']
    se_re  = re_row['se_pool']
    ci_re  = 1.96 * se_re
    marker_col = dir_col if sig else '#aaaaaa'
    ax.errorbar(tau_re, i, xerr=ci_re,
                fmt='D', color=marker_col,
                ms=3.5 if sig else 2.5,
                elinewidth=0.8, capsize=1.5,
                zorder=5, markeredgewidth=0)

# y-axis: gene labels colored by enrichment direction (when significant)
ax.set_yticks(y_pos)
ylabels = []
for gene in genes_ordered:
    re_row = meta_re[meta_re['gene'] == gene].iloc[0]
    star   = '*' if re_row['sig'] else ''
    ylabels.append(f"{gene.upper()}{star}")
ax.set_yticklabels(ylabels, fontsize=7.5)

# color each tick label
for tick, gene in zip(ax.get_yticklabels(), genes_ordered):
    re_row = meta_re[meta_re['gene'] == gene].iloc[0]
    if re_row['sig']:
        tick.set_color(C_CT if re_row['tau_pool'] < 0 else C_LE)
        tick.set_fontweight('bold')
    else:
        tick.set_color('#888888')

ax.axvline(0, color='#444', lw=0.8, ls='--', zorder=1)
ax.set_xlabel(r'Estimated effect $\hat{\tau}$', fontsize=11)
ax.set_title(r'D   Selected marker genes — per-sample $\hat{\tau}$ distribution and RE pooled estimate',
             loc='left', fontsize=13, fontweight='bold', pad=6)
ax.set_ylim(-0.8, n_genes - 0.2)
ax.set_xlim(-0.5, 0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=10.5)

# Move panel D down by 0.5cm while keeping its title fixed in place: capture
# the title's absolute figure position first, redraw it as a figure-level
# text after repositioning the axes.
_title_obj  = ax.title
_tx, _ty    = _title_obj.get_position()
_fig_xy     = fig.transFigure.inverted().transform(ax.transAxes.transform((_tx, _ty)))
_title_text = _title_obj.get_text()
_title_kw   = dict(fontsize=_title_obj.get_fontsize(), fontweight=_title_obj.get_fontweight(),
                   ha=_title_obj.get_ha(), va=_title_obj.get_va(), color=_title_obj.get_color())
ax.set_title('')
_pos = ax.get_position()
_dy_panelD = ((1.0 - 0.5 + 0.3) / 2.54) / fig.get_figheight()   # net: 1cm up requested, minus the 0.5cm-down title-alignment offset, plus 0.3cm more to align with panel E's height
_new_y0_D  = _pos.y0 + _dy_panelD
ax.set_position([_pos.x0, _new_y0_D, _pos.width, _pos.height])
_panelD_top = _new_y0_D + _pos.height   # used to align panel E's top edge below
fig.text(_fig_xy[0], _fig_xy[1], _title_text, transform=fig.transFigure, **_title_kw)


# ══════════════════════════════════════════════════════════════════════════
# Panel E — competitor comparison: significant detections & correct direction
#           (each method fitted per-sample on all 26 sections, pooled with the
#           same DerSimonian-Laird random-effects meta-analysis as tidest)
# ══════════════════════════════════════════════════════════════════════════
ax = ax_cmp
_morder  = ['tidest', 'DESpace', 'SpatialGEE', 't-test', 'SpaGCN']
_mkeys   = ['tidest', 'despace', 'spatialgee', 'ttest', 'spagcn']
_mlabels = ['TIDEST' if k == 'tidest' else METHOD_LABELS[k] for k in _mkeys]
_summ    = cmp_summary.loc[_morder]

_y    = np.arange(len(_morder))
_h    = 0.34
_col1 = '#BBBBBB'
_col2 = [MC[k] for k in _mkeys]

ax.barh(_y + _h/2, _summ['n_sig'],         height=_h, color=_col1, alpha=0.7, label='Significant / 90')
ax.barh(_y - _h/2, _summ['n_sig_correct'], height=_h, color=_col2, alpha=0.95, label='Correct direction / 81')

for yi, (sv, cv) in enumerate(zip(_summ['n_sig'], _summ['n_sig_correct'])):
    ax.text(sv + 1, yi + _h/2, f'{int(sv)}', va='center', ha='left', fontsize=10, color='#777777')
    ax.text(cv + 1, yi - _h/2, f'{int(cv)}', va='center', ha='left', fontsize=10, color=_col2[yi], fontweight='bold')

ax.set_yticks(_y)
ax.set_yticklabels(_mlabels, fontsize=11)
for tick, k in zip(ax.get_yticklabels(), _mkeys):
    tick.set_color(MC[k])
    if k == 'tidest':
        tick.set_fontweight('bold')
ax.set_ylim(len(_morder) - 0.5, -0.5)
ax.set_xlabel('Number of marker genes', fontsize=11)
ax.set_xlim(0, 58)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='both', labelsize=10.5)
_leg_E = ax.legend(fontsize=10, loc='lower right', frameon=False,
                   handlelength=1.1, handletextpad=0.4, labelspacing=0.25)
ax.set_title('E   Competitor comparison\n(per-sample fit, pooled by RE meta-analysis)',
             loc='left', fontsize=12, fontweight='bold', pad=20)

# Shrink panel E to roughly half its allotted height, shift it 2cm left to
# reduce the gap from panel A, and align its top edge with panel D's top
# (i.e. raise it so it starts where panel D starts).
_pos = ax.get_position()
_new_h = _pos.height * 0.5
_dx_left = (3.0 / 2.54) / fig.get_figwidth()
ax.set_position([_pos.x0 - _dx_left, _panelD_top - _new_h, _pos.width, _new_h])

# Shift the panel-E legend right by 0.3cm (converted to axes-fraction units
# using panel E's final width)
_axes_w_in = ax.get_position().width * fig.get_figwidth()
_dx_leg    = (0.3 / 2.54) / _axes_w_in
_leg_E.set_bbox_to_anchor((1.0 + _dx_leg, 0.0), transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════
# Panel F — OPC-like / ECM "paradox" genes: pooled tau (all CT-enriched,
#           opposite the LE-enriched direction expected from single-cell data),
#           placed in the empty space below panel E.
# ══════════════════════════════════════════════════════════════════════════
_pos_E   = ax_cmp.get_position()
_gap_F   = 0.052   # leaves room below panel E's x-axis label before F's title
_ax_F    = fig.add_axes([_pos_E.x0, 0.03278, _pos_E.width, _pos_E.y0 - 0.03278 - _gap_F])
ax       = _ax_F

PARADOX_GENES = ['bcan', 'olig1', 'ptprz1', 'olig2', 'ascl1', 'sox10', 'pdgfra']
_par = meta_full.set_index('gene').loc[PARADOX_GENES]

_yp = np.arange(len(PARADOX_GENES))
ax.barh(_yp, _par['tau_pool'], height=0.62, color=C_CT, alpha=0.85, zorder=3)
ax.errorbar(_par['tau_pool'], _yp, xerr=1.96 * _par['se_pool'], fmt='none',
            ecolor='#444444', elinewidth=0.8, capsize=2, zorder=4)
ax.axvline(0, color='#444', lw=0.8, ls='--', zorder=1)

ax.set_yticks(_yp)
ax.set_yticklabels([g.upper() for g in PARADOX_GENES], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel(r'RE pooled $\hat{\tau}$', fontsize=11)
ax.set_xlim(-0.26, 0.08)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='both', labelsize=10.5)

# Annotate the expected (single-cell-predicted) direction, opposite to what is observed
ax.annotate('expected: LE-enriched',
            xy=(0.022, len(PARADOX_GENES) - 1.6), xytext=(0.06, len(PARADOX_GENES) - 1.6),
            fontsize=10, color=C_LE, ha='left', va='center', style='italic',
            arrowprops=dict(arrowstyle='-|>', color=C_LE, lw=1.2))

ax.set_title('F   OPC-like / ECM paradox genes — 5 methods agree',
             loc='left', fontsize=12, fontweight='bold', pad=6.8)   # +0.1cm (~2.8pt) above the previous pad=4


# ══════════════════════════════════════════════════════════════════════════
# Rows 1-3 — 3 samples × (annotation | GFAP spatial | SNAP25 spatial)
# ══════════════════════════════════════════════════════════════════════════
re_snap = meta_re.loc[meta_re['gene'] == 'snap25', 'tau_pool'].values[0]
re_gfap = meta_re.loc[meta_re['gene'] == 'gfap',  'tau_pool'].values[0]

# Panel labels (A/B/C), placed horizontally in black above the first plot
# of each row. Rows 1-2 carry GFAP then SNAP25 (D/E swapped from the original
# SNAP25-then-GFAP order).
ROW_LABELS = {
    0: 'A   IvyGAP annotation',
    1: f'B   GFAP  (RE $\\hat{{\\tau}}$={re_gfap:+.2f})',
    2: f'C   SNAP25  (RE $\\hat{{\\tau}}$={re_snap:+.2f})',
}

for s_i, sample in enumerate(SAMPLES):
    adata  = visium[sample]
    xy     = adata.obsm['spatial']
    xc     = xy[:, 0]
    yc     = -xy[:, 1]
    reg    = adata.obs['region'].values
    is_le  = (reg == 'LE');  is_ct = (reg == 'CT')
    n_ct   = is_ct.sum();    n_le  = is_le.sum()
    pct_le = 100 * n_le / (n_ct + n_le)

    def get_expr(gene):
        col = adata[:, gene].X
        return col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()

    # Augmented pseudo-outcome (Y^pseudo) — what tidest's PLM is fit on —
    # used for the per-gene spatial expression maps (panels B, C).
    padata = pseudo[sample]
    pxy    = padata.obsm['spatial']
    pxc    = pxy[:, 0]
    pyc    = -pxy[:, 1]

    def get_pseudo(gene):
        col = padata[:, gene].X
        return col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()

    # ── Row 1: annotation (hexbin majority vote, CT/LE colors preserved) ──
    ax = fig.add_subplot(gs_samples[0, s_i])
    region_bin = is_le.astype(float)   # 0 = CT, 1 = LE
    majority_cmap = mpl_colors.ListedColormap([C_CT, C_LE])
    ax.hexbin(xc, yc, C=region_bin,
              reduce_C_function=lambda v: 1.0 if np.mean(v) >= 0.5 else 0.0,
              gridsize=24, cmap=majority_cmap, mincnt=1,
              vmin=0, vmax=1, linewidths=0.1, edgecolors='none',
              zorder=2)
    ax.set_aspect('equal');  ax.axis('off')
    ax.set_title(f'{sample}\n{n_ct} CT / {n_le} LE  ({pct_le:.0f}% LE)',
                 fontsize=10.5, fontweight='bold', pad=3)
    if s_i == 0:
        ax.text(0.0, 1.16, ROW_LABELS[0], transform=ax.transAxes,
                ha='left', va='bottom', fontsize=11.5, fontweight='bold',
                color='black')
    ax.legend(handles=[
        mpatches.Patch(color=C_CT, label='CT'),
        mpatches.Patch(color=C_LE, label='LE'),
    ], fontsize=9, loc='lower right', framealpha=0.85,
       markerscale=1.2, handlelength=1.0)

    # ── Rows 2-3: spatial expression maps ───────────────────────────────
    gene_info = [
        ('gfap',   C_CT, f"GFAP  τ={plm_all.loc[(plm_all['sample']==sample)&(plm_all['gene']=='gfap'),'tau'].values[0]:+.3f}"),
        ('snap25', C_LE, f"SNAP25  τ={plm_all.loc[(plm_all['sample']==sample)&(plm_all['gene']=='snap25'),'tau'].values[0]:+.3f}"),
    ]
    for g_i, (gene, gcol, gtitle) in enumerate(gene_info):
        ax = fig.add_subplot(gs_samples[1 + g_i, s_i])
        expr = get_pseudo(gene)

        vmin = np.percentile(expr, 5)
        vmax = np.percentile(expr, 97)

        sc_map = ax.hexbin(pxc, pyc, C=expr, reduce_C_function=np.mean,
                           gridsize=24, cmap='viridis', mincnt=1,
                           vmin=vmin, vmax=vmax,
                           linewidths=0.1, edgecolors='none',
                           zorder=2)
        ax.set_aspect('equal');  ax.axis('off')
        ax.set_title(gtitle, fontsize=10, fontweight='bold', color=gcol, pad=3)
        if s_i == 0:
            ax.text(0.0, 1.16, ROW_LABELS[1 + g_i], transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=11.5, fontweight='bold',
                    color='black')

        # Colorbar
        cbar = plt.colorbar(sc_map, ax=ax, shrink=0.55, pad=0.02,
                            orientation='vertical')
        cbar.set_label('Augmented outcome', fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)

        # Pooled (random-effects, all 26 samples) competitor q-values — shown once,
        # under the middle sample's panel, to avoid repeating per-sample.
        if s_i == 1 and gene in cmp_pergene.index:
            row = cmp_pergene.loc[gene]
            _note_fontsize = 11.6
            children = []
            for j, mk in enumerate(['tidest', 'despace', 'spatialgee', 'ttest', 'spagcn']):
                q = row.get(f'qval_{mk}', np.nan)
                if pd.isna(q):
                    continue
                if j > 0 and children:
                    children.append(TextArea('   |   ', textprops=dict(
                        fontsize=_note_fontsize, color='#555555', style='italic')))
                q_s  = f'{q:.1e}'.replace('e-0', '×10⁻').replace('e-', '×10⁻')
                flag = ' ✓' if q < 0.05 else ''
                children.append(TextArea(f'{METHOD_LABELS[mk]}: $q$={q_s}{flag}',
                                          textprops=dict(fontsize=_note_fontsize,
                                                         color=MC[mk], style='italic')))
            packer = HPacker(children=children, align='center', pad=0, sep=0)
            # Center on the full figure width (not the sample axes), shifted
            # right by an extra 6cm per the requested offset.
            _ax_pos  = ax.get_position()
            _dx_fig  = (6.0 / 2.54) / fig.get_figwidth()
            _dy_fig  = (0.5 / 2.54) / fig.get_figheight()   # net 1cm down - 0.5cm up
            _y_fig   = _ax_pos.y0 + (-0.07) * _ax_pos.height - _dy_fig
            ab = AnnotationBbox(packer, (0.5 + _dx_fig, _y_fig),
                                xycoords=fig.transFigure, box_alignment=(0.5, 1.0),
                                frameon=False, annotation_clip=False)
            ax.add_artist(ab)

print(f'Saving to {OUT_PATH}...')
plt.savefig(OUT_PATH, bbox_inches='tight', facecolor='white')
plt.close()
print('Done.')
