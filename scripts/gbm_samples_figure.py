"""
GBM sample-level figure: 3 balanced samples × (annotation | SNAP25 dist | GFAP dist).

Rows: UKF304 (48% LE), ZH881inf (45% LE), UKF269 (44% LE)
Cols: IvyGAP spatial annotation | SNAP25 (LE-enriched, τ>0) | GFAP (CT-enriched, τ<0)
"""

import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc

warnings.filterwarnings('ignore')

META_CSV   = './Inputs/general/visium_metadata.csv'
PLM_CSV    = './results_raw/gbm_plm_results_all.csv'
RE_CSV     = './results_raw/gbm_meta_results.csv'
OUT_PATH   = './results_raw/gbm_samples_figure.png'

SAMPLES = ['UKF304', 'ZH881inf', 'UKF269']
GENES   = ['snap25', 'gfap']          # col 2 = positive τ, col 3 = negative τ

C_CT  = '#d73027'    # red
C_LE  = '#2166ac'    # blue
GENE_COLORS = {'snap25': C_LE, 'gfap': C_CT}

# ── Load metadata and per-sample PLM results ──────────────────────────────
meta_csv = pd.read_csv(META_CSV)
plm_all  = pd.read_csv(PLM_CSV)
re_meta  = pd.read_csv(RE_CSV).set_index('gene')

# ── Figure layout: 3 rows × 3 cols ────────────────────────────────────────
fig = plt.figure(figsize=(13, 11), dpi=180)
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.45, wspace=0.32,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

COL_TITLES = [
    'IvyGAP annotation\n(CT vs LE)',
    'SNAP25 expression\n(LE-enriched, $\\hat{\\tau}>0$)',
    'GFAP expression\n(CT-enriched, $\\hat{\\tau}<0$)',
]

# Column title row
for col, title in enumerate(COL_TITLES):
    ax_dummy = fig.add_subplot(gs[0, col])
    ax_dummy.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax_dummy.remove()

# ── RE pooled τ for annotation ─────────────────────────────────────────────
re_snap = re_meta.loc['snap25', 'tau_pool']
re_gfap = re_meta.loc['gfap',  'tau_pool']

# ── Per-sample data loading and plotting ──────────────────────────────────
for row, sample in enumerate(SAMPLES):
    print(f'Processing {sample}...')

    # Load Visium
    adata = sc.read_visium(f'./Inputs/general/GBM_data/{sample}/outs/')
    adata.var_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.str.lower())

    # Filter to CT + LE spots
    meta_s  = meta_csv[meta_csv['sample'] == sample].set_index('spot_id')
    ct_le   = meta_s[meta_s['ivygap'].isin(['CT', 'LE'])].index
    adata   = adata[adata.obs_names.intersection(ct_le)].copy()
    adata.obs['region'] = meta_s.loc[adata.obs_names, 'ivygap'].values

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    xy   = adata.obsm['spatial']
    xc   = xy[:, 0];  yc = xy[:, 1]
    reg  = adata.obs['region'].values
    is_le = (reg == 'LE');  is_ct = (reg == 'CT')
    n_ct = is_ct.sum();  n_le = is_le.sum()
    pct_le = 100 * n_le / (n_ct + n_le)

    def get_expr(gene):
        col = adata[:, gene].X
        return col.toarray().ravel() if issparse(col) else np.asarray(col).ravel()

    # Per-sample PLM τ for these genes
    def plm_tau(gene):
        row_df = plm_all[(plm_all['sample'] == sample) & (plm_all['gene'] == gene)]
        if len(row_df):
            r = row_df.iloc[0]
            return r['tau'], r['se'], r['qval']
        return np.nan, np.nan, np.nan

    # ── Col 0: Spatial annotation ──────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 0])

    # Flip y for standard Visium orientation
    yc_flip = -yc

    ax.scatter(xc[is_ct], yc_flip[is_ct], c=C_CT, s=2.0, alpha=0.7,
               linewidths=0, rasterized=True, label='CT', zorder=2)
    ax.scatter(xc[is_le], yc_flip[is_le], c=C_LE, s=2.0, alpha=0.85,
               linewidths=0, rasterized=True, label='LE', zorder=3)

    ax.set_aspect('equal');  ax.axis('off')
    ax.set_title(f'{sample}\n{n_ct} CT  /  {n_le} LE  ({pct_le:.0f}% LE)',
                 fontsize=8.5, fontweight='bold', pad=3)

    leg = ax.legend(handles=[
        mpatches.Patch(color=C_CT, label='CT'),
        mpatches.Patch(color=C_LE, label='LE'),
    ], fontsize=7, loc='lower right', framealpha=0.85,
       markerscale=1.5, handlelength=1)

    # ── Cols 1-2: Violin distributions ────────────────────────────────────
    for col, gene in enumerate(GENES, start=1):
        ax = fig.add_subplot(gs[row, col])
        col_g = GENE_COLORS[gene]

        expr_ct = get_expr(gene)[is_ct]
        expr_le = get_expr(gene)[is_le]

        # Violin
        vp = ax.violinplot([expr_ct, expr_le],
                           positions=[0, 1],
                           widths=0.65,
                           showmedians=True,
                           showextrema=False)
        for i, (body, color) in enumerate(zip(vp['bodies'], [C_CT, C_LE])):
            body.set_facecolor(color)
            body.set_alpha(0.65)
            body.set_edgecolor('none')
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(2.0)
        vp['cmedians'].set_zorder(5)

        # Overlay jittered dots (subsample for readability)
        rng = np.random.default_rng(42 + col)
        for xi, expr, color in [(0, expr_ct, C_CT), (1, expr_le, C_LE)]:
            n_show = min(len(expr), 120)
            idx = rng.choice(len(expr), n_show, replace=False)
            jit = rng.uniform(-0.15, 0.15, n_show)
            ax.scatter(xi + jit, expr[idx], color=color, s=3, alpha=0.35,
                       linewidths=0, zorder=4)

        # Per-sample PLM τ annotation
        tau_s, se_s, q_s = plm_tau(gene)
        if not np.isnan(tau_s):
            sig = '***' if q_s < 0.001 else '**' if q_s < 0.01 else '*' if q_s < 0.05 else 'ns'
            sign = '+' if tau_s >= 0 else ''
            ann  = f'$\\hat{{\\tau}}$={sign}{tau_s:.3f}  {sig}'
            ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                    fontsize=7.5, ha='right', va='top', color=col_g,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec=col_g, lw=0.8, alpha=0.9))

        # Median line connecting the two
        ax.plot([0, 1], [np.median(expr_ct), np.median(expr_le)],
                color='#555555', lw=0.9, ls='--', alpha=0.7, zorder=6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CT', 'LE'], fontsize=9)
        ax.set_ylabel('log-CPM', fontsize=8)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(f'{gene.upper()}', fontsize=9, fontweight='bold',
                     color=col_g, pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=7.5)

# ── Column headers (shared) ────────────────────────────────────────────────
col_header_titles = [
    'IvyGAP annotation',
    f'SNAP25  (RE: $\\hat{{\\tau}}$={re_snap:+.2f})',
    f'GFAP  (RE: $\\hat{{\\tau}}$={re_gfap:+.2f})',
]
col_header_colors = ['#333333', C_LE, C_CT]
for col, (title, col_c) in enumerate(zip(col_header_titles, col_header_colors)):
    fig.text((0.07 + col * 0.30 + 0.15), 0.965,
             title, ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=col_c,
             transform=fig.transFigure)

# ── Figure title ───────────────────────────────────────────────────────────
fig.suptitle(
    'GBM IvyGAP spatial transcriptomics — three balanced samples\n'
    '($\\hat{\\tau}$ per sample from PLM; annotation = CT (red) / LE (blue))',
    fontsize=11, y=1.00, va='bottom'
)

print(f'Saving to {OUT_PATH}...')
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Done.')
