"""
All comparison methods for the simulation study.
Each function takes the same core inputs and returns (tau_hat, se, gene_names).

Methods:
  run_tidest    — Robinson PLM on Y_pseudo with spatial kernel-PCA confounders
  run_ttest     — Welch t-test on log1p(C_obs)
  run_wilcoxon  — Wilcoxon rank-sum on log1p(C_obs)  [used as SpaGCN proxy]
  run_despace   — DESpace svg_test (edgeR NB) via subprocess Rscript
  run_spatialgee — SpatialGEE run_gee_gst (GEE) via subprocess Rscript
  run_ppi       — Prediction-powered inference baseline (no spatial deconfounding)
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Add tidest to path (package installed in editable mode, but be explicit)
_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'tidest')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))

from tidest._plm import robinson_plm


def _build_U(U_spatial, A, log_lib, corr_threshold=0.5):
    """
    Screen spatial PCs by correlation with treatment, then append log_lib.
    Mirrors the real pipeline's CORR_THRESHOLD logic.
    """
    pc_corrs = np.array([np.corrcoef(U_spatial[:, k], A)[0, 1]
                         for k in range(U_spatial.shape[1])])
    keep = np.abs(pc_corrs) < corr_threshold
    U_kept = U_spatial[:, keep]
    return np.hstack([U_kept, log_lib[:, None]])


def run_tidest(Y_pseudo, A, U_spatial, log_lib,
               n_folds=2, n_estimators=100, corr_threshold=0.5, seed=42):
    """
    Robinson PLM on the Pearson-augmented pseudo-outcome.

    Parameters
    ----------
    Y_pseudo   : (N, G) — log pseudo-outcome from build_pseudo_outcome
    A          : (N,) int — treatment
    U_spatial  : (N, K) — spatial kernel-PCA eigenvectors (precomputed)
    log_lib    : (N,) — log(1 + total_expression) per spot
    """
    U = _build_U(U_spatial, A, log_lib, corr_threshold)
    tau, se = robinson_plm(Y_pseudo, A, U, n_folds=n_folds,
                           n_estimators=n_estimators, seed=seed)
    return tau, se


def run_ttest(C_obs, A):
    """
    Welch t-test on log1p(C_obs). Returns (tau_hat, se) where tau_hat =
    mean(log1p, A=1) - mean(log1p, A=0) and se is the pooled SE of the difference.
    """
    Y = np.log1p(C_obs)
    G = Y.shape[1]
    tau = np.zeros(G, dtype=np.float64)
    se = np.zeros(G, dtype=np.float64)
    mask1 = A == 1
    mask0 = A == 0
    for g in range(G):
        y1 = Y[mask1, g]
        y0 = Y[mask0, g]
        tau[g] = y1.mean() - y0.mean()
        # SE of difference of means
        se[g] = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return tau, se


def run_wilcoxon(C_obs, A):
    """
    Wilcoxon rank-sum test on log1p(C_obs).
    SpaGCN uses Wilcoxon internally; the signed z-statistic IS the native effect.
    z > 0  → group A=1 has higher ranks.
    se = 1.0 because the z-statistic is already standardised.

    Bias is computed after SD-standardisation in compute_metrics so it is
    comparable to methods on other scales (PLM τ, logFC, GEE coefficient).
    """
    Y = np.log1p(C_obs)
    G = Y.shape[1]
    tau = np.zeros(G, dtype=np.float64)
    se  = np.ones(G,  dtype=np.float64)   # z-stat is already standardised
    mask1 = A == 1
    mask0 = A == 0
    for g in range(G):
        y1 = Y[mask1, g]
        y0 = Y[mask0, g]
        z, _ = stats.ranksums(y1, y0)
        tau[g] = z   # signed z-statistic: >0 means y1 > y0 in rank
    return tau, se


def run_ppi(Y_pred, C_obs, A, label_frac=0.5, seed=42):
    """
    Prediction-powered inference (PPI) baseline (Angelopoulos et al. 2023;
    Shang et al. 2025), with no spatial-confounder adjustment.

    The raw imputed predictions Y_pred (available for every spot) are used
    for the main group-difference estimate, then rectified by the prediction
    error (observed - predicted) on a random "labeled" subset L of spots
    where the observed outcome is treated as available:

        tau_PPI = [mean(Ytilde|A=1) - mean(Ytilde|A=0)]                (all spots)
                - [mean(Ytilde|A=1,L) - mean(Ytilde|A=0,L)]
                + [mean(Yobs|A=1,L)   - mean(Yobs|A=0,L)]

    This isolates the effect of prediction-powered rectification alone: it
    uses the same predictions as TIDEST's augmentation step but applies no
    spatial deconfounding, so any FPR inflation from spatial confounding is
    expected to persist.

    Parameters
    ----------
    Y_pred : (N, G) — raw imputed predictions (log scale)
    C_obs  : (N, G) — observed counts (log1p applied internally)
    A      : (N,) int — binary treatment
    label_frac : float — fraction of spots used as the labeled set L
    seed   : int — RNG seed for the labeled/unlabeled split

    Returns
    -------
    tau_hat : (G,) float
    se      : (G,) float — sqrt of the sum of the three group-difference variances
    """
    rng = np.random.default_rng(seed)
    N = Y_pred.shape[0]
    L = rng.random(N) < label_frac

    Y_obs = np.log1p(C_obs)
    mask1, mask0 = A == 1, A == 0
    mask1_L, mask0_L = mask1 & L, mask0 & L

    def _mean_diff_var(Y, m1, m0):
        y1, y0 = Y[m1], Y[m0]
        diff = y1.mean(axis=0) - y0.mean(axis=0)
        var = y1.var(axis=0, ddof=1) / len(y1) + y0.var(axis=0, ddof=1) / len(y0)
        return diff, var

    diff_full,  var_full  = _mean_diff_var(Y_pred, mask1,   mask0)
    diff_predL, var_predL = _mean_diff_var(Y_pred, mask1_L, mask0_L)
    diff_obsL,  var_obsL  = _mean_diff_var(Y_obs,  mask1_L, mask0_L)

    tau = diff_full - diff_predL + diff_obsL
    se = np.sqrt(var_full + var_predL + var_obsL)
    return tau, se


# ── R-based methods ────────────────────────────────────────────────────────────

def _run_r_method(rscript_path, C_obs, A, coords, timeout=300):
    """
    Shared helper: write temp CSVs, call an R script, parse the output.

    The R script is called as:
        Rscript <rscript_path> <counts.csv> <meta.csv> <out.csv>

    Returns
    -------
    pval    : (G,) float — p-values in gene order (NaN on failure)
    tau_hat : (G,) float — log1p group mean difference (computed here in Python)
    se      : (G,) float — SE back-computed from p-value via normal approximation
    """
    G = C_obs.shape[1]
    N = C_obs.shape[0]
    gene_names = [f"gene{g}" for g in range(G)]
    spot_names = [f"spot{i}" for i in range(N)]

    # tau_hat from log1p counts (consistent with t-test / Wilcoxon)
    Y = np.log1p(C_obs)
    tau_hat = Y[A == 1].mean(axis=0) - Y[A == 0].mean(axis=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        counts_file = os.path.join(tmpdir, 'counts.csv')
        meta_file   = os.path.join(tmpdir, 'meta.csv')
        out_file    = os.path.join(tmpdir, 'results.csv')

        # counts.csv: genes × spots (R convention), integer counts
        counts_df = pd.DataFrame(
            np.round(C_obs).astype(np.int64).T,  # (G, N)
            index=gene_names,
            columns=spot_names,
        )
        counts_df.to_csv(counts_file)

        # meta.csv: spots × (spot_id, A, x, y)
        meta_df = pd.DataFrame({
            'spot_id': spot_names,
            'A':       A.astype(np.int32),
            'x':       coords[:, 0].astype(np.float64),
            'y':       coords[:, 1].astype(np.float64),
        })
        meta_df.to_csv(meta_file, index=False)

        result = subprocess.run(
            ['Rscript', '--vanilla', rscript_path,
             counts_file, meta_file, out_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0 or not os.path.exists(out_file):
            stderr_tail = result.stderr[-600:] if result.stderr else '(no stderr)'
            raise RuntimeError(
                f"R script {os.path.basename(rscript_path)} failed "
                f"(rc={result.returncode}):\n{stderr_tail}"
            )

        res_df = pd.read_csv(out_file)

    # Parse p-values and native effect estimates from R output.
    # DESpace    → logfc  (edgeR log2 fold change, from glmLrt$table)
    # SpatialGEE → gee_coef (Poisson GEE log-coefficient, trt vs ctrl)
    gene_to_idx = {f"gene{g}": g for g in range(G)}
    pval   = np.ones(G,  dtype=np.float64)
    effect = np.full(G,  np.nan, dtype=np.float64)

    eff_col = None
    if 'logfc' in res_df.columns:
        eff_col = 'logfc'       # DESpace: edgeR log2 fold change
    elif 'gee_coef' in res_df.columns:
        eff_col = 'gee_coef'    # SpatialGEE: Poisson log-coefficient

    for _, row in res_df.iterrows():
        idx = gene_to_idx.get(str(row['gene']))
        if idx is not None:
            if not np.isnan(float(row['pvalue'])):
                pval[idx] = float(row['pvalue'])
            if eff_col is not None:
                v = float(row[eff_col])
                if not np.isnan(v):
                    effect[idx] = v

    # Use native effect estimate where available; fall back to group mean diff.
    tau_out = np.where(np.isfinite(effect), effect, tau_hat)

    # SE back-computed so that tau/se ≈ the method's z-score (for TPR/FPR/AUC).
    z_abs  = np.abs(stats.norm.isf(np.clip(pval / 2.0, 1e-300, 0.5)))
    safe_z = np.where(z_abs > 1e-10, z_abs, np.nan)
    se     = np.abs(tau_out) / safe_z

    return tau_out, se


def run_despace(C_obs, A, coords, seed=42):
    """
    DESpace svg_test on raw Poisson counts with treatment A as spatial cluster.
    Uses edgeR NB model (LRT); native effect is the log2 fold change from glmLrt.

    Parameters
    ----------
    C_obs  : (N, G) float — observed Poisson counts
    A      : (N,) int    — binary treatment (0/1)
    coords : (N, 2) float — spatial coordinates

    Returns
    -------
    tau_hat : (G,) — edgeR log2 fold change (logFC from glmLrt$table)
    se      : (G,) — back-computed so tau/se ≈ LRT z-score
    """
    rscript = os.path.join(_SIM_DIR, 'run_despace.R')
    return _run_r_method(rscript, C_obs, A, coords)


def run_spatialgee(C_obs, A, coords, seed=42):
    """
    SpatialGEE: GST p-value from run_gee_gst; effect from full GEE Poisson model.
    The GST (score test at H0) gives only a p-value; the Poisson GEE coefficient
    from geeglm(geneexp ~ Pathology.Annotations) is the native effect estimate.

    Parameters
    ----------
    C_obs  : (N, G) float — observed Poisson counts
    A      : (N,) int    — binary treatment (0/1)
    coords : (N, 2) float — spatial coordinates

    Returns
    -------
    tau_hat : (G,) — Poisson GEE log-coefficient (trt vs ctrl, natural log scale)
    se      : (G,) — back-computed so tau/se ≈ GST z-score
    """
    rscript = os.path.join(_SIM_DIR, 'run_spatialgee.R')
    return _run_r_method(rscript, C_obs, A, coords)
