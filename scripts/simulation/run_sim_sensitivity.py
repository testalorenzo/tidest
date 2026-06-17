"""
Sensitivity sweeps for TIDEST hyperparameters (referees #2 and #7).

Fixed setting: two-region DGP, alpha_conf=0.5, sigma_imp=0.5, n_spots=300,
tau_level=1.0 (the main figure's central setting). For each sweep below,
only the swept parameter is varied; all others are held at their pipeline
defaults (top_k=5, min_corr=None, pearson_noise_sd=0, n_pcs=20,
corr_threshold=0.5). Only TIDEST is run -- the swept parameters affect only
the augmentation step (build_pseudo_outcome) and/or the TIDEST PLM stage
(_build_U / robinson_plm), so the other methods are unaffected by
construction.

Sweeps:
  top_k            in {1, 3, 5, 10, 20}          -- augmentation neighbour count
  min_corr         in {0.0, 0.1, 0.2, 0.3}       -- augmentation correlation floor
  pearson_noise_sd in {0.0, 0.05, 0.1, 0.2, 0.3} -- noise on SC reference correlations
  n_pcs            in {5, 10, 20, 50}            -- SpatialPCA components
  corr_threshold   in {0.3, 0.4, 0.5, 0.6, 0.7}  -- PC-vs-treatment screen

Usage:
  python scripts/simulation/run_sim_sensitivity.py [--quick] [--n-reps N] [--n-jobs N]

Results saved to: results_raw/sim_sensitivity_raw.csv
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
from methods import run_tidest
from sim_utils import compute_metrics, compute_reconstruction_metrics
from run_sim import G, G_DE, G_NULL, ELL, IS_NULL, precompute_shared


# ── Fixed setting (main figure's central point) ─────────────────────────────────
N_SPOTS     = 300
TAU_LEVEL   = 1.0
ALPHA_CONF  = 0.5
SIGMA_IMP   = 0.5
DGP_VARIANT = 'two-region'

# Pipeline defaults (baseline for every sweep)
DEFAULTS = dict(top_k=5, min_corr=None, pearson_noise_sd=0.0,
                n_pcs=20, corr_threshold=0.5)

SWEEPS = {
    'top_k':            [1, 3, 5, 10, 20],
    # Top-5 within-module correlations in this DGP have a minimum of ~0.41
    # (median ~0.65), so thresholds below ~0.4 never bind; this grid spans
    # the region where the floor starts to drop neighbours.
    'min_corr':         [0.0, 0.3, 0.45, 0.55, 0.65],
    'pearson_noise_sd': [0.0, 0.05, 0.1, 0.2, 0.3],
    'n_pcs':            [5, 10, 20, 50],
    'corr_threshold':   [0.3, 0.4, 0.5, 0.6, 0.7],
}

_NAN_M = {'tpr': np.nan, 'fpr': np.nan, 'bias': np.nan, 'rmse': np.nan, 'auc': np.nan}


def simulate_one_sensitivity(seed, loadings, sc_pearson, sc_pearson_genes,
                              top_k=5, min_corr=None, pearson_noise_sd=0.0,
                              n_pcs=20, corr_threshold=0.5):
    """One replicate, TIDEST only, with the given hyperparameters."""
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

    # Optionally perturb the SC reference correlation matrix (symmetric noise,
    # clipped to a valid correlation range, diagonal left at 1).
    pearson = sc_pearson
    if pearson_noise_sd > 0:
        rng = np.random.default_rng(rng_seed + 1000)
        noise = rng.normal(0, pearson_noise_sd, size=pearson.shape)
        noise = np.triu(noise, k=1)
        noise = noise + noise.T
        pearson = np.clip(pearson + noise, -1.0, 1.0)
        np.fill_diagonal(pearson, 1.0)

    pseudo, _, _ = build_pseudo_outcome(
        sc_adata=None, st_adata=st_adata, pred_adata=pred_adata,
        pearson=pearson, pearson_genes=sc_pearson_genes,
        top_k=top_k, min_corr=min_corr, pred_is_log=True,
    )

    gene_names = [f"gene{g}" for g in range(G)]
    shared_genes = [g for g in gene_names if g in pseudo.var_names]
    Y_pseudo = np.asarray(pseudo[:, shared_genes].X).astype(np.float64)
    gene_idx = [int(g.replace('gene', '')) for g in shared_genes]

    Y_true_shared = Y_true[:, gene_idx]
    Y_pred_shared = Y_pred[:, gene_idx]
    _, rmse_pseudo = compute_reconstruction_metrics(Y_pred_shared, Y_pseudo, Y_true_shared)

    U_sp = precompute_spatial_pcs(coords, n_pcs=n_pcs, ell=ELL)
    log_lib = np.log1p(np.expm1(np.asarray(st_adata.X).astype(np.float64)).sum(axis=1))

    tau_true_shared = tau_true[gene_idx]
    is_null_shared  = IS_NULL[gene_idx]

    try:
        tau, se = run_tidest(Y_pseudo, A, U_sp, log_lib,
                              n_folds=2, n_estimators=100,
                              corr_threshold=corr_threshold, seed=rng_seed)
        m = compute_metrics(tau, se, tau_true_shared, is_null_shared)
    except Exception as e:
        print(f'[seed={seed}] tidest failed: {e}', flush=True)
        m = _NAN_M.copy()

    return {
        'seed': seed,
        'rmse_pseudo':     rmse_pseudo,
        'tpr_tidest':      m['tpr'],
        'fpr_tidest':      m['fpr'],
        'bias_tidest':     m['bias'],
        'rmse_tau_tidest': m['rmse'],
        'auc_tidest':      m['auc'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='5 reps, sanity check')
    parser.add_argument('--n-reps', type=int, default=100)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    n_reps = 5 if args.quick else args.n_reps

    out_path = os.path.join(_ROOT, 'results_raw', 'sim_sensitivity_raw.csv')
    os.makedirs(os.path.join(_ROOT, 'results_raw'), exist_ok=True)

    print('Precomputing shared SC reference / module loadings...')
    loadings, pearson, pearson_genes = precompute_shared(N_SPOTS, DGP_VARIANT)

    all_rows = []
    for sweep_name, values in SWEEPS.items():
        for val in values:
            kwargs = DEFAULTS.copy()
            kwargs[sweep_name] = val
            print(f'\nSweep {sweep_name}={val} ({n_reps} reps)...')
            rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(simulate_one_sensitivity)(
                    seed=rep, loadings=loadings,
                    sc_pearson=pearson, sc_pearson_genes=pearson_genes,
                    **kwargs,
                )
                for rep in range(n_reps)
            )
            for r in rows:
                r['sweep'] = sweep_name
                r['value'] = val
            all_rows.extend(rows)
            print(f'  TPR={np.nanmean([r["tpr_tidest"] for r in rows]):.3f}  '
                  f'FPR={np.nanmean([r["fpr_tidest"] for r in rows]):.3f}  '
                  f'AUC={np.nanmean([r["auc_tidest"] for r in rows]):.3f}  '
                  f'RMSE_pseudo={np.nanmean([r["rmse_pseudo"] for r in rows]):.4f}')
            # Save partial results after each sweep point for crash recovery
            pd.DataFrame(all_rows).to_csv(out_path.replace('.csv', '_partial.csv'), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
