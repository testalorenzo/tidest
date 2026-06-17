"""
Main simulation orchestration for the tidest simulation study.

Usage:
  python scripts/simulation/run_sim.py                    # main figure grid
  python scripts/simulation/run_sim.py --quick            # 10 reps, one setting (sanity)
  python scripts/simulation/run_sim.py --full-grid        # full supplementary grid
  python scripts/simulation/run_sim.py --skip-r-methods   # skip DESpace / SpatialGEE

Results saved to: results_raw/sim_results.csv
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import anndata as ad

# ── Path setup ─────────────────────────────────────────────────────────────────
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS  = os.path.dirname(_SIM_DIR)
_ROOT     = os.path.dirname(_SCRIPTS)
sys.path.insert(0, os.path.join(_ROOT, 'tidest'))

from tidest._pseudo_outcome import compute_pearson, build_pseudo_outcome
from dgp import (
    generate_module_loadings, generate_sc_adata,
    generate_grid, assign_treatment,
    generate_st_expression, build_st_adata,
    simulate_imputation, precompute_spatial_pcs,
)
from methods import (
    run_tidest, run_ttest, run_wilcoxon,
    run_despace, run_spatialgee,
)
from sim_utils import compute_metrics, compute_reconstruction_metrics


# ── Constants ──────────────────────────────────────────────────────────────────
G      = 200    # total genes
G_DE   = 50     # DE genes (first G_DE)
G_NULL = 150    # null genes
M      = 10     # gene modules
C      = 2000   # SC cells for Pearson reference
N_PCS  = 20     # spatial kernel PCs
ELL    = 0.3    # GRF / kernel length scale

TAU_LEVELS = {
    'small':  0.5,
    'medium': 1.0,
    'large':  2.0,
}
SIGMA_IMP_LEVELS = {
    'high':   0.1,   # near-perfect imputation
    'medium': 0.5,
    'low':    1.0,   # noisy imputation
}
ALPHA_GRID     = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SPOTS_GRID   = [100, 300, 500]
DGP_VARIANTS   = ['two-region', 'grf', 'latent-factor']

IS_NULL = np.array([False] * G_DE + [True] * G_NULL)

_NAN_METRICS = {'tpr': np.nan, 'fpr': np.nan, 'bias': np.nan,
                'rmse': np.nan, 'auc': np.nan}


# ── Per-replicate simulation ────────────────────────────────────────────────────

def simulate_one(
    seed,
    n_spots=300,
    tau_level=1.0,
    alpha_conf=0.5,
    sigma_imp=0.5,
    dgp_variant='two-region',
    loadings=None,
    sc_pearson=None,
    sc_pearson_genes=None,
    U_spatial=None,
    skip_r_methods=False,
):
    """
    Run one simulation replicate. Returns dict of metrics.

    loadings, sc_pearson, sc_pearson_genes, U_spatial are precomputed and
    passed in to avoid redundant work across replicates.
    """
    rng_seed = seed

    # 1. Spatial grid (fixed for given n_spots/variant; coords vary by variant seed)
    coords = generate_grid(n_spots, seed=rng_seed)

    # 2. Treatment
    A = assign_treatment(coords, variant=dgp_variant, ell=ELL,
                         noise_std=0.1, seed=rng_seed + 1)

    # 3. True expression + observed counts
    Y_true, C_obs, tau_true = generate_st_expression(
        coords, A, loadings,
        G_DE=G_DE, G_null=G_NULL,
        tau_level=tau_level,
        alpha=alpha_conf,
        sigma=1.0,
        ell=ELL,
        seed=rng_seed + 2,
    )

    # 4. Noisy imputed predictions (module-structured noise)
    Y_pred, pred_adata = simulate_imputation(
        Y_true, loadings, sigma_imp=sigma_imp, seed=rng_seed + 3
    )

    # 5. ST AnnData (pred_is_log=True convention: .X = log1p(C_obs))
    st_adata = build_st_adata(C_obs, coords, A, G_DE, G_NULL)

    # Align pred_adata spots to st_adata
    pred_adata.obs_names = st_adata.obs_names

    # 6. Pearson correction (actual tidest code)
    pseudo, _, _ = build_pseudo_outcome(
        sc_adata=None,              # precomputed Pearson passed directly
        st_adata=st_adata,
        pred_adata=pred_adata,
        pearson=sc_pearson,
        pearson_genes=sc_pearson_genes,
        top_k=5,
        pred_is_log=True,
    )

    # Extract pseudo-outcome matrix aligned to st_adata gene order
    gene_names = [f"gene{g}" for g in range(G)]
    shared_genes = [g for g in gene_names if g in pseudo.var_names]
    Y_pseudo = np.asarray(pseudo[:, shared_genes].X).astype(np.float64)

    # 7. Reconstruction metrics (augmentation panel)
    gene_idx = [int(g.replace('gene', '')) for g in shared_genes]
    Y_true_shared  = Y_true[:, gene_idx]
    Y_pred_shared  = Y_pred[:, gene_idx]
    rmse_pred, rmse_pseudo = compute_reconstruction_metrics(
        Y_pred_shared, Y_pseudo, Y_true_shared
    )

    # 8. Confounder matrix (recomputed per seed since coords vary)
    U_sp = precompute_spatial_pcs(coords, n_pcs=N_PCS, ell=ELL)
    log_lib = np.log1p(np.expm1(np.asarray(st_adata.X).astype(np.float64)).sum(axis=1))

    tau_true_shared = tau_true[gene_idx]
    is_null_shared  = IS_NULL[gene_idx]
    C_obs_shared    = C_obs[:, gene_idx]
    coords_shared   = coords  # full coords, not gene-indexed

    # ── M1: tidest (PLM on pseudo-outcome + spatial PCA) ──────────────────────
    try:
        tau1, se1 = run_tidest(Y_pseudo, A, U_sp, log_lib,
                               n_folds=2, n_estimators=100, seed=rng_seed)
        m1 = compute_metrics(tau1, se1, tau_true_shared, is_null_shared)
    except Exception as e:
        print(f'[seed={seed}] tidest failed: {e}', flush=True)
        m1 = _NAN_METRICS.copy()

    # ── M2: t-test on observed counts ─────────────────────────────────────────
    tau2, se2 = run_ttest(C_obs_shared, A)
    m2 = compute_metrics(tau2, se2, tau_true_shared, is_null_shared)

    # ── M3: SpaGCN proxy = Wilcoxon rank-sum on observed counts ───────────────
    tau3, se3 = run_wilcoxon(C_obs_shared, A)
    m3 = compute_metrics(tau3, se3, tau_true_shared, is_null_shared)

    # ── M4: DESpace (edgeR NB via subprocess Rscript) ─────────────────────────
    if skip_r_methods:
        m4 = _NAN_METRICS.copy()
    else:
        try:
            tau4, se4 = run_despace(C_obs_shared, A, coords_shared, seed=rng_seed)
            m4 = compute_metrics(tau4, se4, tau_true_shared, is_null_shared)
        except Exception as e:
            print(f'[seed={seed}] DESpace failed: {e}', flush=True)
            m4 = _NAN_METRICS.copy()

    # ── M5: SpatialGEE (GEE via subprocess Rscript) ───────────────────────────
    if skip_r_methods:
        m5 = _NAN_METRICS.copy()
    else:
        try:
            tau5, se5 = run_spatialgee(C_obs_shared, A, coords_shared, seed=rng_seed)
            m5 = compute_metrics(tau5, se5, tau_true_shared, is_null_shared)
        except Exception as e:
            print(f'[seed={seed}] SpatialGEE failed: {e}', flush=True)
            m5 = _NAN_METRICS.copy()

    return {
        'seed':        seed,
        'n_spots':     n_spots,
        'tau_level':   tau_level,
        'alpha_conf':  alpha_conf,
        'sigma_imp':   sigma_imp,
        'dgp_variant': dgp_variant,
        # Reconstruction
        'rmse_pred':   rmse_pred,
        'rmse_pseudo': rmse_pseudo,
        # tidest
        'tpr_tidest':      m1['tpr'],  'fpr_tidest':      m1['fpr'],
        'bias_tidest':     m1['bias'], 'rmse_tau_tidest': m1['rmse'],
        'auc_tidest':      m1['auc'],
        # t-test
        'tpr_ttest':       m2['tpr'],  'fpr_ttest':       m2['fpr'],
        'bias_ttest':      m2['bias'], 'rmse_tau_ttest':  m2['rmse'],
        'auc_ttest':       m2['auc'],
        # SpaGCN (Wilcoxon rank-sum)
        'tpr_spagcn':      m3['tpr'],  'fpr_spagcn':      m3['fpr'],
        'bias_spagcn':     m3['bias'], 'rmse_tau_spagcn': m3['rmse'],
        'auc_spagcn':      m3['auc'],
        # DESpace
        'tpr_despace':     m4['tpr'],  'fpr_despace':     m4['fpr'],
        'bias_despace':    m4['bias'], 'rmse_tau_despace':m4['rmse'],
        'auc_despace':     m4['auc'],
        # SpatialGEE
        'tpr_spatialgee':  m5['tpr'],  'fpr_spatialgee':  m5['fpr'],
        'bias_spatialgee': m5['bias'], 'rmse_tau_spatialgee': m5['rmse'],
        'auc_spatialgee':  m5['auc'],
    }


# ── Precompute shared objects ──────────────────────────────────────────────────

def precompute_shared(n_spots, dgp_variant, sc_seed=0):
    """
    Precompute objects that are constant across replicates for a given setting:
    - Gene module loadings
    - SC Pearson matrix
    - (spatial PCs not precomputed here: coords vary by seed)
    """
    loadings = generate_module_loadings(G, M, seed=sc_seed)
    sc_adata = generate_sc_adata(G, M, loadings, C=C, seed=sc_seed + 100)
    print('  Computing Pearson matrix from synthetic SC reference...')
    pearson, pearson_genes = compute_pearson(sc_adata)
    return loadings, pearson, pearson_genes


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run 10 reps at one setting (sanity check)')
    parser.add_argument('--full-grid', action='store_true',
                        help='Run full supplementary parameter grid')
    parser.add_argument('--n-reps', type=int, default=None,
                        help='Override number of replicates')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs (-1 = all cores)')
    parser.add_argument('--n-spots', type=int, default=None,
                        help='Override n_spots (e.g. 300 to fix spots across all grid runs)')
    parser.add_argument('--skip-r-methods', action='store_true',
                        help='Skip DESpace and SpatialGEE (fill with NaN) for fast iteration')
    args = parser.parse_args()

    out_path = os.path.join(_ROOT, 'results_raw', 'sim_results.csv')
    os.makedirs(os.path.join(_ROOT, 'results_raw'), exist_ok=True)

    spots_grid = [args.n_spots] if args.n_spots else N_SPOTS_GRID

    if args.quick:
        n_s = args.n_spots or 300
        settings = [{'n_spots': n_s, 'tau_level': 1.0, 'alpha_conf': 0.5,
                     'sigma_imp': 0.5, 'dgp_variant': 'two-region'}]
        n_reps = args.n_reps or 10
    elif args.full_grid:
        settings = [
            {'n_spots': n, 'tau_level': TAU_LEVELS[t], 'alpha_conf': a,
             'sigma_imp': SIGMA_IMP_LEVELS[q], 'dgp_variant': v}
            for n in spots_grid
            for t in TAU_LEVELS
            for a in ALPHA_GRID
            for q in SIGMA_IMP_LEVELS
            for v in DGP_VARIANTS
        ]
        n_reps = args.n_reps or 500
    else:
        # Main figure: fix N=300, tau=medium, sigma_imp=medium; vary alpha and DGP variant
        settings = [
            {'n_spots': 300, 'tau_level': 1.0, 'alpha_conf': a,
             'sigma_imp': 0.5, 'dgp_variant': v}
            for a in ALPHA_GRID
            for v in DGP_VARIANTS
        ]
        n_reps = args.n_reps or 500

    if args.skip_r_methods:
        print('NOTE: --skip-r-methods active; DESpace and SpatialGEE columns will be NaN.')

    print(f'Settings: {len(settings)}, reps each: {n_reps}')
    print(f'Total replicates: {len(settings) * n_reps}')

    all_rows = []
    partial_path = out_path.replace('.csv', '_partial.csv')

    for s_idx, setting in enumerate(settings):
        n = setting['n_spots']
        v = setting['dgp_variant']
        a = setting['alpha_conf']
        t = setting['tau_level']
        q = setting['sigma_imp']
        print(f'\n[{s_idx+1}/{len(settings)}] n={n}, variant={v}, alpha={a:.2f},'
              f' tau={t:.1f}, sigma_imp={q:.2f}')

        loadings, pearson, pearson_genes = precompute_shared(n, v)

        rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(simulate_one)(
                seed=rep,
                n_spots=n, tau_level=t, alpha_conf=a, sigma_imp=q,
                dgp_variant=v,
                loadings=loadings,
                sc_pearson=pearson,
                sc_pearson_genes=pearson_genes,
                skip_r_methods=args.skip_r_methods,
            )
            for rep in range(n_reps)
        )
        all_rows.extend(rows)
        # Save partial results after each setting for crash recovery
        pd.DataFrame(all_rows).to_csv(partial_path, index=False)

        def _fpr(key):
            vals = [r[f'fpr_{key}'] for r in rows]
            return np.nanmean(vals)

        print(f'  Done {n_reps} reps. '
              f'TPR tidest={np.nanmean([r["tpr_tidest"] for r in rows]):.3f}  '
              f'FPR: tidest={_fpr("tidest"):.3f} ttest={_fpr("ttest"):.3f} '
              f'spagcn={_fpr("spagcn"):.3f} despace={_fpr("despace"):.3f} '
              f'spatialgee={_fpr("spatialgee"):.3f}')

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
