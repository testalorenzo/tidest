"""
Simulation stress tests A-D (referee #3).

Fixed setting: two-region DGP, alpha_conf=1.0, sigma_imp=0.5, n_spots=300,
tau_level=1.0 (the high-confounding setting underlying the FPR=13.7% headline
number and the calibration analysis, run_calibration.py). For each stress
test below, only the stressed aspect of the DGP/reference is varied; TIDEST
and a naive t-test are run as in run_calibration.py (no spatial-confounder
adjustment for the t-test, full pipeline for TIDEST).

Stress tests:
  A. Weak gene-gene correlation: corr_strength in {1.0, 0.5, 0.15} scales the
     module loadings that drive both the SC-reference Pearson correlations
     and the module-structured imputation noise (dgp.generate_module_loadings).
  B. Reference mismatch: mismatch_frac in {0.0, 0.3, 0.6} -- fraction of genes
     whose SC-reference module loadings are permuted relative to the ST tissue
     (dgp.generate_sc_adata_mismatched).
  C. Non-smooth confounding: confounder_type in {smooth, discontinuous,
     hotspot, multiscale} (dgp.generate_st_expression / _sample_confounder).
  D. Small reference: SC reference size C in {2000, 500, 100}
     (dgp.generate_sc_adata, no new code).

Usage:
  python scripts/simulation/run_sim_stress_tests.py [--quick] [--n-reps N] [--n-jobs N]

Results saved to: results_raw/sim_stress_tests_raw.csv
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_SIM_DIR)
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, os.path.join(_ROOT, 'tidest'))

from tidest._pseudo_outcome import compute_pearson, build_pseudo_outcome
from dgp import (
    generate_module_loadings, generate_sc_adata, generate_sc_adata_mismatched,
    generate_grid, assign_treatment, generate_st_expression,
    build_st_adata, simulate_imputation, precompute_spatial_pcs,
)
from methods import run_tidest, run_ttest
from sim_utils import compute_metrics, compute_reconstruction_metrics
from run_sim import G, G_DE, G_NULL, M, ELL, N_PCS, IS_NULL


# ── Fixed setting (high-confounding point used in the calibration analysis) ────
N_SPOTS     = 300
TAU_LEVEL   = 1.0
ALPHA_CONF  = 1.0
SIGMA_IMP   = 0.5
DGP_VARIANT = 'two-region'
C_DEFAULT   = 2000  # default SC reference size

STRESS_TESTS = {
    'A_corr_strength':   [1.0, 0.5, 0.15],
    'B_mismatch_frac':   [0.0, 0.3, 0.6],
    'C_confounder_type': ['smooth', 'discontinuous', 'hotspot', 'multiscale'],
    'D_sc_size':         [2000, 500, 100],
}

_NAN_M = {'tpr': np.nan, 'fpr': np.nan, 'bias': np.nan, 'rmse': np.nan, 'auc': np.nan}


def _precompute(corr_strength=1.0, mismatch_frac=0.0, c_sc=C_DEFAULT, sc_seed=0):
    """Precompute module loadings + SC-reference Pearson matrix for one stress setting."""
    loadings = generate_module_loadings(G, M, seed=sc_seed, corr_strength=corr_strength)
    if mismatch_frac > 0:
        sc_adata = generate_sc_adata_mismatched(G, M, loadings, C=c_sc,
                                                 seed=sc_seed + 100, mismatch_frac=mismatch_frac)
    else:
        sc_adata = generate_sc_adata(G, M, loadings, C=c_sc, seed=sc_seed + 100)
    pearson, pearson_genes = compute_pearson(sc_adata)
    return loadings, pearson, pearson_genes


def simulate_one_stress(seed, loadings, pearson, pearson_genes, confounder_type='smooth'):
    """One replicate: TIDEST + t-test, under the given stress condition."""
    rng_seed = seed

    coords = generate_grid(N_SPOTS, seed=rng_seed)
    A = assign_treatment(coords, variant=DGP_VARIANT, ell=ELL,
                          noise_std=0.1, seed=rng_seed + 1)
    Y_true, C_obs, tau_true = generate_st_expression(
        coords, A, loadings,
        G_DE=G_DE, G_null=G_NULL,
        tau_level=TAU_LEVEL, alpha=ALPHA_CONF, sigma=1.0, ell=ELL,
        confounder_type=confounder_type, seed=rng_seed + 2,
    )
    Y_pred, pred_adata = simulate_imputation(
        Y_true, loadings, sigma_imp=SIGMA_IMP, seed=rng_seed + 3
    )
    st_adata = build_st_adata(C_obs, coords, A, G_DE, G_NULL)
    pred_adata.obs_names = st_adata.obs_names

    pseudo, _, _ = build_pseudo_outcome(
        sc_adata=None, st_adata=st_adata, pred_adata=pred_adata,
        pearson=pearson, pearson_genes=pearson_genes,
        top_k=5, pred_is_log=True,
    )
    gene_names = [f"gene{g}" for g in range(G)]
    shared_genes = [g for g in gene_names if g in pseudo.var_names]
    Y_pseudo = np.asarray(pseudo[:, shared_genes].X).astype(np.float64)
    gene_idx = [int(g.replace('gene', '')) for g in shared_genes]

    Y_true_shared   = Y_true[:, gene_idx]
    Y_pred_shared   = Y_pred[:, gene_idx]
    C_obs_shared    = C_obs[:, gene_idx]
    tau_true_shared = tau_true[gene_idx]
    is_null_shared  = IS_NULL[gene_idx]

    rmse_pred, rmse_pseudo = compute_reconstruction_metrics(Y_pred_shared, Y_pseudo, Y_true_shared)

    U_sp = precompute_spatial_pcs(coords, n_pcs=N_PCS, ell=ELL)
    log_lib = np.log1p(np.expm1(np.asarray(st_adata.X).astype(np.float64)).sum(axis=1))

    try:
        tau, se = run_tidest(Y_pseudo, A, U_sp, log_lib,
                              n_folds=2, n_estimators=100, seed=rng_seed)
        m_tidest = compute_metrics(tau, se, tau_true_shared, is_null_shared)
    except Exception as e:
        print(f'[seed={seed}] tidest failed: {e}', flush=True)
        m_tidest = _NAN_M.copy()

    tau_t, se_t = run_ttest(C_obs_shared, A)
    m_ttest = compute_metrics(tau_t, se_t, tau_true_shared, is_null_shared)

    return {
        'seed': seed,
        'rmse_pred': rmse_pred,
        'rmse_pseudo': rmse_pseudo,
        'tpr_tidest': m_tidest['tpr'], 'fpr_tidest': m_tidest['fpr'], 'auc_tidest': m_tidest['auc'],
        'tpr_ttest':  m_ttest['tpr'],  'fpr_ttest':  m_ttest['fpr'],  'auc_ttest':  m_ttest['auc'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='5 reps, sanity check')
    parser.add_argument('--n-reps', type=int, default=100)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    n_reps = 5 if args.quick else args.n_reps

    out_dir = os.path.join(_ROOT, 'results_raw')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sim_stress_tests_raw.csv')

    all_rows = []
    for stress_name, values in STRESS_TESTS.items():
        for val in values:
            print(f'\n{stress_name}={val} ({n_reps} reps)...')

            if stress_name == 'A_corr_strength':
                loadings, pearson, pearson_genes = _precompute(corr_strength=val)
                confounder_type = 'smooth'
            elif stress_name == 'B_mismatch_frac':
                loadings, pearson, pearson_genes = _precompute(mismatch_frac=val)
                confounder_type = 'smooth'
            elif stress_name == 'C_confounder_type':
                loadings, pearson, pearson_genes = _precompute()
                confounder_type = val
            elif stress_name == 'D_sc_size':
                loadings, pearson, pearson_genes = _precompute(c_sc=val)
                confounder_type = 'smooth'
            else:
                raise ValueError(stress_name)

            rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(simulate_one_stress)(
                    seed=rep, loadings=loadings,
                    pearson=pearson, pearson_genes=pearson_genes,
                    confounder_type=confounder_type,
                )
                for rep in range(n_reps)
            )
            for r in rows:
                r['stress'] = stress_name
                r['value'] = val
            all_rows.extend(rows)
            print(f'  TIDEST: TPR={np.nanmean([r["tpr_tidest"] for r in rows]):.3f}  '
                  f'FPR={np.nanmean([r["fpr_tidest"] for r in rows]):.3f}  '
                  f'AUC={np.nanmean([r["auc_tidest"] for r in rows]):.3f}  '
                  f'RMSE_pseudo={np.nanmean([r["rmse_pseudo"] for r in rows]):.4f}  |  '
                  f't-test: FPR={np.nanmean([r["fpr_ttest"] for r in rows]):.3f}')
            # Save partial results after each stress point for crash recovery
            pd.DataFrame(all_rows).to_csv(out_path.replace('.csv', '_partial.csv'), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
