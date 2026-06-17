"""
Generate the simulation study figure for Nature Methods.

Two-row figure:
  Row 1 — DGP illustration (one fixed replicate):
    A: SC gene-gene Pearson correlation matrix (block module structure)
    B: Spatial treatment layout (A on 2D grid)
    C: Spatial confounder field (Z on 2D grid)
    D: True expression of one DE gene (treatment + confounder combined)

  Row 2 — Simulation results (100 reps, tau=medium, sigma_imp=medium):
    E: Augmentation quality — RMSE(Y_pred) vs RMSE(Y_pseudo) across sigma_imp
    F: Power (TPR) vs confounder strength alpha
    G: FPR (type-I error) vs confounder strength alpha
    H: |Bias| of tau-hat vs confounder strength alpha

Reads:
  results_raw/sim_results.csv
  results_raw/sim_augmentation.csv

Saves:
  results_raw/sim_figure.png  (300 dpi)
  results_raw/sim_figure.pdf
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import copy
from sklearn.preprocessing import normalize

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_SIM_DIR, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'tidest'))
sys.path.insert(0, _SIM_DIR)

from dgp import (
    generate_module_loadings, generate_sc_adata,
    generate_grid, assign_treatment,
    generate_st_expression, _sample_grf,
    precompute_spatial_pcs,
)

# ── Style ──────────────────────────────────────────────────────────────────────
COL = {
    'tidest':     '#EE6677',   # red
    'ttest':      '#4477AA',   # blue
    'spagcn':     '#228833',   # green   (SpaGCN uses Wilcoxon rank-sum)
    'despace':    '#CCBB44',   # yellow-gold
    'spatialgee': '#AA3377',   # purple
    'pred':       '#BBBBBB',   # grey
    'pseudo':     '#EE6677',   # red (same as tidest)
}
LABELS = {
    'tidest':     'TIDEST',
    'ttest':      't-test',
    'spagcn':     'SpaGCN (Wilcoxon)',
    'despace':    'DESpace',
    'spatialgee': 'SpatialGEE',
    'pred':       'Raw imputation',
    'pseudo':     'Augmented (TIDEST)',
}

METHODS = ['tidest', 'ttest', 'spagcn', 'despace', 'spatialgee']

plt.rcParams.update({
    'font.size':        12,
    'axes.labelsize':   12,
    'axes.titlesize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       150,
})


# ── Row 1: DGP illustration ────────────────────────────────────────────────────

def _pearson_from_sc(sc_adata):
    """Compute gene×gene Pearson from SC (same logic as tidest compute_pearson)."""
    from scanpy.preprocessing import normalize_total
    sc = sc_adata.copy()
    normalize_total(sc, target_sum=1e4)
    X = sc.X.astype(np.float32)
    X = np.log1p(X)
    X -= X.mean(axis=0)
    X = normalize(X, norm='l2', axis=0)
    return X.T @ X   # (G, G)


def panel_corr_matrix(ax, loadings, sc_adata, G=200, M=10):
    """
    Heatmap of the SC gene-gene Pearson correlation matrix.
    Genes are ordered by module membership, so block structure is visible.
    """
    P = _pearson_from_sc(sc_adata)   # (G, G)

    # Clip diagonal for display (auto-correlation = 1 swamps colour scale)
    P_display = P.copy()
    np.fill_diagonal(P_display, np.nan)
    im = ax.imshow(P_display, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')

    # Draw module boundary lines
    genes_per_module = G // M
    for k in range(1, M):
        b = k * genes_per_module - 0.5
        ax.axhline(b, color='k', lw=0.5, alpha=0.5)
        ax.axvline(b, color='k', lw=0.5, alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Genes (ordered by module)')
    ax.set_ylabel('Genes (ordered by module)')
    ax.set_title('A   scRNA-seq gene–gene\nPearson correlation (block modules)',
                 loc='left', fontweight='bold')
    ax.spines[:].set_visible(True)


# Simple blue→red colormap for binary treatment (no white mid-point)
_CMAP_BINARY = LinearSegmentedColormap.from_list('br', ['#3A7DC9', '#CC3333'])
# Sequential light→teal colormap for expression (matches seaborn hexbin example)
_CMAP_EXPR = LinearSegmentedColormap.from_list('teal_hex', ['#eaf6f1', '#4CB391'])


def _scatter_spatial(ax, coords, values, cmap, vmin, vmax, title, cbar_label,
                     norm=None):
    # Proper flat-top hex lattice: col spacing = √3·R, row spacing = 3R/2,
    # odd rows offset by √3·R/2. Sized so n_side-1 columns span x=[0,1].
    n_side = int(np.ceil(np.sqrt(len(coords))))
    R  = 1.0 / ((n_side - 1) * np.sqrt(3))
    dx = np.sqrt(3) * R   # column centre-to-centre
    dy = 1.5 * R          # row centre-to-centre

    # Map square-grid coords → hex-lattice positions
    col_idx = np.round(coords[:, 0] * (n_side - 1)).astype(int)
    row_idx = np.round(coords[:, 1] * (n_side - 1)).astype(int)
    hx = col_idx * dx + np.where(row_idx % 2 == 1, dx / 2, 0.0)
    hy = row_idx * dy

    patches = [RegularPolygon((xi, yi), numVertices=6, radius=R, orientation=0)
               for xi, yi in zip(hx, hy)]
    col = PatchCollection(patches,
                          cmap=(plt.get_cmap(cmap) if isinstance(cmap, str) else cmap),
                          norm=norm, linewidth=0.4, edgecolor='white')
    col.set_array(values.astype(float))
    if norm is None:
        col.set_clim(vmin, vmax)
    ax.add_collection(col)
    plt.colorbar(col, ax=ax, orientation='horizontal', pad=0.04,
                 fraction=0.046, label=cbar_label)
    margin = R * 1.2
    ax.set_xlim(hx.min() - margin, hx.max() + margin)
    ax.set_ylim(hy.min() - margin, hy.max() + margin)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines[:].set_visible(False)


def panel_treatment(ax, coords, A):
    # Binary: one fixed colour per group, legend instead of colorbar.
    from matplotlib.patches import Patch as _Patch
    n_side = int(np.ceil(np.sqrt(len(coords))))
    R  = 1.0 / ((n_side - 1) * np.sqrt(3))
    dx = np.sqrt(3) * R
    dy = 1.5 * R
    col_idx = np.round(coords[:, 0] * (n_side - 1)).astype(int)
    row_idx = np.round(coords[:, 1] * (n_side - 1)).astype(int)
    hx = col_idx * dx + np.where(row_idx % 2 == 1, dx / 2, 0.0)
    hy = row_idx * dy

    _TCOL = {0: '#3A7DC9', 1: '#CC3333'}
    _TLAB = {0: 'A = 0', 1: 'A = 1'}
    for val in [0, 1]:
        mask = (A == val)
        patches = [RegularPolygon((xi, yi), numVertices=6, radius=R, orientation=0)
                   for xi, yi in zip(hx[mask], hy[mask])]
        col = PatchCollection(patches, facecolor=_TCOL[val],
                              linewidth=0.4, edgecolor='white', match_original=False)
        ax.add_collection(col)

    margin = R * 1.2
    ax.set_xlim(hx.min() - margin, hx.max() + margin)
    ax.set_ylim(hy.min() - margin, hy.max() + margin)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('B   Treatment A', loc='left', fontweight='bold')
    ax.spines[:].set_visible(False)

    # Invisible dummy colorbar: steals the same axes space as C and D
    # so the hex area in B matches exactly.
    import matplotlib.colors as _mc
    _dummy = plt.cm.ScalarMappable(cmap=_mc.ListedColormap(['white']),
                                    norm=_mc.Normalize(0, 1))
    _dummy.set_array([])
    _cb = plt.colorbar(_dummy, ax=ax, orientation='horizontal',
                       pad=0.04, fraction=0.046)
    _cb.ax.set_visible(False)

    ax.legend(handles=[_Patch(facecolor=_TCOL[v], edgecolor='white', label=_TLAB[v])
                        for v in [0, 1]],
              loc='upper center', bbox_to_anchor=(0.5, -0.04),
              fontsize=10, framealpha=0.8, ncol=2)


def panel_confounder(ax, coords, Z):
    norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
    _scatter_spatial(
        ax, coords, Z,
        cmap='PuOr', vmin=None, vmax=None, norm=norm,
        title='C   Spatial confounder Z',
        cbar_label='Z',
    )


def panel_true_expression(ax, coords, Y_true, tau_true, gene_idx=0):
    vals = Y_true[:, gene_idx]
    sign = '+' if tau_true[gene_idx] > 0 else '−'
    tau_abs = abs(tau_true[gene_idx])
    _scatter_spatial(
        ax, coords, vals,
        cmap='viridis', vmin=vals.min(), vmax=vals.max(), norm=None,
        title='D   True expression (gene 0)',
        cbar_label='log-expression',
    )


# ── Row 2: results ─────────────────────────────────────────────────────────────

def _agg(df, group_col, value_col, band='se'):
    g = df.groupby(group_col)[value_col].agg(['mean', 'sem', 'std']).reset_index()
    g = g.sort_values(group_col)
    ci = 1.96 * g['sem'] if band == 'se' else g['std']
    return g[group_col].values, g['mean'].values, ci.values


def _line(ax, x, y, err, color, label, ls='-', marker='o'):
    ax.plot(x, y, color=color, label=label, ls=ls,
            marker=marker, ms=4.5, lw=1.8, zorder=3)
    ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, zorder=2)
    ax.spines[['top', 'right']].set_visible(False)


def panel_confounder_reconstruction(ax, coords, Z, n_pcs=20, ell=0.3):
    """
    R²(Z, U[:, :k]) vs k: how well do the first k spatial PCs reconstruct
    the unobserved confounder Z.  Uses the same fixed illustration replicate
    as the spatial maps in row 1.  Only tidest appears here — it is the only
    method that uses the spatial PCA confounder representation.
    """
    U_sp = precompute_spatial_pcs(coords, n_pcs=n_pcs, ell=ell)
    Z_c  = Z - Z.mean()   # centre Z for clean R² computation
    ss_tot = float(Z_c @ Z_c)

    ks, r2_vals = [], []
    for k in range(1, n_pcs + 1):
        U_k  = U_sp[:, :k]
        beta = np.linalg.lstsq(U_k, Z_c, rcond=None)[0]
        Z_hat = U_k @ beta
        r2 = 1.0 - float((Z_c - Z_hat) @ (Z_c - Z_hat)) / ss_tot
        ks.append(k); r2_vals.append(r2)

    ax.plot(ks, r2_vals, color=COL['tidest'], lw=2, marker='o', ms=4.5, zorder=3)
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlim(0.5, n_pcs + 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Number of spatial PCs (k)')
    ax.set_ylabel('Reconstruction power R²')
    ax.set_title('F   Spatial PCA reconstructs\nunobserved confounder Z',
                 loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)


def panel_augmentation(ax, df_aug):
    for key, col, color in [
        ('pred',   'rmse_pred',   COL['pred']),
        ('pseudo', 'rmse_pseudo', COL['pseudo']),
    ]:
        x, y, err = _agg(df_aug, 'sigma_imp', col, band='sd')
        _line(ax, x, y, err, color, LABELS[key])
    ax.set_xlabel('Imputation noise σ  (→ lower quality)')
    ax.set_ylabel('RMSE')
    ax.set_title('E   Augmentation reduces\nreconstruction error', loc='left', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.axvline(0.3, color='gray', ls=':', lw=1, alpha=0.6)


def panel_power(ax, sub):
    for key in METHODS:
        col = f'tpr_{key}'
        if col not in sub.columns or sub[col].isna().all():
            continue
        x, y, err = _agg(sub, 'alpha_conf', col)
        _line(ax, x, y, err, COL[key], LABELS[key])
    ax.set_xlabel('Confounder strength α')
    ax.set_ylabel('Power (TPR at q < 0.05)')
    ax.set_ylim(0.8, 1.01)
    ax.set_title('H   Power vs confounder\nstrength', loc='left', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)


def panel_fpr(ax, sub):
    for key in METHODS:
        col = f'fpr_{key}'
        if col not in sub.columns or sub[col].isna().all():
            continue
        x, y, err = _agg(sub, 'alpha_conf', col)
        _line(ax, x, y, err, COL[key], LABELS[key])
    ax.axhline(0.05, color='gray', ls='--', lw=1, zorder=1, label='Nominal 5%')
    ax.set_ylim(0, 0.7)
    ax.set_xlabel('Confounder strength α')
    ax.set_ylabel('FPR (type-I error rate)')
    ax.set_title('G   Type-I error vs\nconfounder strength', loc='left', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)


def panel_bias(ax, sub):
    # Relative bias = raw_bias / SD(τ_true[DE]), shown only for methods whose
    # native effect is on the same log-expression scale as τ_true.
    # SpaGCN (z-stat) and DESpace (log2FC) are excluded — different units.
    BIAS_METHODS = ['tidest', 'ttest', 'spatialgee']
    for key in BIAS_METHODS:
        col = f'bias_{key}'
        if col not in sub.columns or sub[col].isna().all():
            continue
        x, y, err = _agg(sub, 'alpha_conf', col)
        _line(ax, x, np.abs(y), err, COL[key], LABELS[key])
    ax.axhline(0, color='gray', ls='--', lw=1, zorder=1)
    ax.set_xlabel('Confounder strength α')
    ax.set_ylabel('|Relative bias| (÷ SD[τ_true])')
    ax.set_title('H   Relative bias vs\nconfounder strength', loc='left', fontweight='bold')
    ax.legend(framealpha=0.9)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dgp-variant', default='two-region',
                        choices=['two-region', 'grf', 'latent-factor'])
    args = parser.parse_args()

    # ── Load results ──────────────────────────────────────────────────────────
    csv_main = os.path.join(_ROOT, 'results_raw', 'sim_results.csv')
    csv_aug  = os.path.join(_ROOT, 'results_raw', 'sim_augmentation.csv')
    df     = pd.read_csv(csv_main)
    df_aug = pd.read_csv(csv_aug)

    sub = df[
        (df['n_spots'] == 300) &
        (df['dgp_variant'] == args.dgp_variant) &
        (df['tau_level'] == 1.0) &
        (df['sigma_imp'] == 0.5)
    ].copy()
    n_reps = len(sub) // sub['alpha_conf'].nunique()
    print(f'Results rows: {len(sub)}, {n_reps} reps per alpha level')

    # ── Generate one fixed DGP replicate for illustration ─────────────────────
    G, G_DE, G_NULL, M, C, ELL, N_PCS = 200, 50, 150, 10, 2000, 0.3, 20
    loadings  = generate_module_loadings(G, M, seed=0)
    sc_adata  = generate_sc_adata(G, M, loadings, C=C, seed=100)
    coords    = generate_grid(400, seed=42)
    A         = assign_treatment(coords, variant=args.dgp_variant, ell=ELL, seed=42)
    Z         = _sample_grf(coords, ell=ELL, seed=1042)   # confounder field
    Y_true, _, tau_true = generate_st_expression(
        coords, A, loadings, G_DE=G_DE, G_null=G_NULL,
        tau_level=1.0, alpha=0.5, sigma=1.0, ell=ELL, seed=42,
    )

    # ── Layout: 2 rows × 4 columns ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.35,
                            top=0.97, bottom=0.09, left=0.06, right=0.97)

    ax_corr  = fig.add_subplot(gs[0, 0])
    ax_treat = fig.add_subplot(gs[0, 1])
    ax_conf  = fig.add_subplot(gs[0, 2])
    ax_expr  = fig.add_subplot(gs[0, 3])

    ax_aug    = fig.add_subplot(gs[1, 0])
    ax_recon  = fig.add_subplot(gs[1, 1])
    ax_power  = fig.add_subplot(gs[1, 2])
    ax_fpr    = fig.add_subplot(gs[1, 3])

    # Row 1
    panel_corr_matrix(ax_corr, loadings, sc_adata, G=G, M=M)
    panel_treatment(ax_treat, coords, A)
    panel_confounder(ax_conf, coords, Z)
    panel_true_expression(ax_expr, coords, Y_true, tau_true, gene_idx=0)

    # Row 2
    panel_augmentation(ax_aug, df_aug)
    panel_confounder_reconstruction(ax_recon, coords, Z, n_pcs=N_PCS, ell=ELL)
    panel_fpr(ax_power, sub)
    panel_power(ax_fpr, sub)

    # no figure-level title

    out = os.path.join(_ROOT, 'results_raw', 'sim_figure')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'Saved {out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
