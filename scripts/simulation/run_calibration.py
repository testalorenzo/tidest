"""
Calibration diagnostics for the simulation study (referees #4, #5).

At the main figure's high-confounding setting (two-region DGP, alpha_conf=1.0,
sigma_imp=0.5, n_spots=300, tau_level=1.0 -- the setting underlying the
FPR=13.7% headline number in the main text), this script runs all 6 methods
(TIDEST, t-test, SpaGCN-proxy/Wilcoxon, DESpace, SpatialGEE, and the new PPI
baseline) for n_reps replicates and records:

  - per-null-gene p-values (for p-value histograms and QQ plots)
  - per-replicate empirical CI coverage for null genes (should contain 0)
    and DE genes (should contain tau_true), at the 95% nominal level
  - per-replicate TPR/FPR/AUC (as in run_sim.py)

Usage:
  python scripts/simulation/run_calibration.py [--quick] [--n-reps N] [--n-jobs N] [--skip-r-methods]

Results saved to:
  results_raw/sim_calibration_pvals.csv  (rep, method, pval)
  results_raw/sim_calibration_raw.csv    (rep, method, tpr, fpr, auc, coverage_null, coverage_de)
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_SIM_DIR)
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, os.path.join(_ROOT, 'tidest'))

from tidest._pseudo_outcome import build_pseudo_outcome
from dgp import (
    generate_grid, assign_treatment, generate_st_expression,
    build_st_adata, simulate_imputation, precompute_spatial_pcs,
)
from methods import (
    run_tidest, run_ttest, run_wilcoxon, run_ppi,
    run_despace, run_spatialgee,
)
from sim_utils import (
    compute_metrics, compute_null_pvalues, compute_ci_coverage,
)
from run_sim import G, G_DE, G_NULL, ELL, N_PCS, IS_NULL, precompute_shared


# ── Fixed setting: main figure's high-confounding point ─────────────────────────
N_SPOTS     = 300
TAU_LEVEL   = 1.0
ALPHA_CONF  = 1.0
SIGMA_IMP   = 0.5
DGP_VARIANT = 'two-region'

METHODS = ['tidest', 'ttest', 'spagcn', 'despace', 'spatialgee', 'ppi']


def simulate_one_calibration(seed, loadings, sc_pearson, sc_pearson_genes,
                              skip_r_methods=False):
    """One replicate: run all METHODS, compute calibration diagnostics."""
    rng_seed = seed

    coords = generate_grid(N_SPOTS, seed=rng_seed)
    A = assign_treatment(coords, variant=DGP_VARIANT, ell=ELL,
                          noise_std=0.1, seed=rng_seed + 1)
    Y_true, C_obs, tau_true = generate_st_expression(
        coords, A, loadings,
        G_DE=G_DE, G_null=G_NULL,
        tau_level=TAU_LEVEL, alpha=ALPHA_CONF, sigma=1.0, ell=ELL,
        seed=rng_seed + 2,
    )
    Y_pred, pred_adata = simulate_imputation(
        Y_true, loadings, sigma_imp=SIGMA_IMP, seed=rng_seed + 3
    )
    st_adata = build_st_adata(C_obs, coords, A, G_DE, G_NULL)
    pred_adata.obs_names = st_adata.obs_names

    pseudo, _, _ = build_pseudo_outcome(
        sc_adata=None, st_adata=st_adata, pred_adata=pred_adata,
        pearson=sc_pearson, pearson_genes=sc_pearson_genes,
        top_k=5, pred_is_log=True,
    )
    gene_names = [f"gene{g}" for g in range(G)]
    shared_genes = [g for g in gene_names if g in pseudo.var_names]
    Y_pseudo = np.asarray(pseudo[:, shared_genes].X).astype(np.float64)
    gene_idx = [int(g.replace('gene', '')) for g in shared_genes]

    Y_pred_shared   = Y_pred[:, gene_idx]
    C_obs_shared    = C_obs[:, gene_idx]
    coords_shared   = coords
    tau_true_shared = tau_true[gene_idx]
    is_null_shared  = IS_NULL[gene_idx]

    U_sp = precompute_spatial_pcs(coords, n_pcs=N_PCS, ell=ELL)
    log_lib = np.log1p(np.expm1(np.asarray(st_adata.X).astype(np.float64)).sum(axis=1))

    pval_rows = []
    raw_rows = []

    def _record(method, tau, se):
        m = compute_metrics(tau, se, tau_true_shared, is_null_shared)
        cov = compute_ci_coverage(tau, se, tau_true_shared, is_null_shared)
        raw_rows.append({
            'rep': seed, 'method': method,
            'tpr': m['tpr'], 'fpr': m['fpr'], 'auc': m['auc'],
            'coverage_null': cov['coverage_null'], 'coverage_de': cov['coverage_de'],
        })
        pvals = compute_null_pvalues(tau, se, is_null_shared)
        for p in pvals:
            pval_rows.append({'rep': seed, 'method': method, 'pval': p})

    # TIDEST
    try:
        tau, se = run_tidest(Y_pseudo, A, U_sp, log_lib, n_folds=2,
                              n_estimators=100, seed=rng_seed)
        _record('tidest', tau, se)
    except Exception as e:
        print(f'[seed={seed}] tidest failed: {e}', flush=True)

    # t-test
    tau, se = run_ttest(C_obs_shared, A)
    _record('ttest', tau, se)

    # SpaGCN proxy (Wilcoxon)
    tau, se = run_wilcoxon(C_obs_shared, A)
    _record('spagcn', tau, se)

    # PPI
    tau, se = run_ppi(Y_pred_shared, C_obs_shared, A, seed=rng_seed)
    _record('ppi', tau, se)

    # R-based methods
    if not skip_r_methods:
        try:
            tau, se = run_despace(C_obs_shared, A, coords_shared, seed=rng_seed)
            _record('despace', tau, se)
        except Exception as e:
            print(f'[seed={seed}] DESpace failed: {e}', flush=True)
        try:
            tau, se = run_spatialgee(C_obs_shared, A, coords_shared, seed=rng_seed)
            _record('spatialgee', tau, se)
        except Exception as e:
            print(f'[seed={seed}] SpatialGEE failed: {e}', flush=True)

    return raw_rows, pval_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='10 reps, sanity check')
    parser.add_argument('--n-reps', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--skip-r-methods', action='store_true')
    args = parser.parse_args()

    n_reps = 10 if args.quick else args.n_reps

    out_dir = os.path.join(_ROOT, 'results_raw')
    os.makedirs(out_dir, exist_ok=True)

    print('Precomputing shared SC reference / module loadings...')
    loadings, pearson, pearson_genes = precompute_shared(N_SPOTS, DGP_VARIANT)

    print(f'Running {n_reps} replicates at alpha_conf={ALPHA_CONF}, '
          f'sigma_imp={SIGMA_IMP}, n_spots={N_SPOTS}, dgp={DGP_VARIANT}...')
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(simulate_one_calibration)(
            seed=rep, loadings=loadings,
            sc_pearson=pearson, sc_pearson_genes=pearson_genes,
            skip_r_methods=args.skip_r_methods,
        )
        for rep in range(n_reps)
    )

    raw_rows = [r for rep_raw, _ in results for r in rep_raw]
    pval_rows = [r for _, rep_pval in results for r in rep_pval]

    raw_df = pd.DataFrame(raw_rows)
    pval_df = pd.DataFrame(pval_rows)

    raw_path = os.path.join(out_dir, 'sim_calibration_raw.csv')
    pval_path = os.path.join(out_dir, 'sim_calibration_pvals.csv')
    raw_df.to_csv(raw_path, index=False)
    pval_df.to_csv(pval_path, index=False)
    print(f'Saved {len(raw_df)} rows to {raw_path}')
    print(f'Saved {len(pval_df)} rows to {pval_path}')

    print('\nSummary (mean across replicates):')
    print(raw_df.groupby('method')[['tpr', 'fpr', 'auc', 'coverage_null', 'coverage_de']].mean().round(3))


if __name__ == '__main__':
    main()
