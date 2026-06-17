"""Metric computation for the simulation study."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compute_metrics(tau_hat, se, tau_true, is_null, alpha=0.05):
    """
    Compute power, FPR, bias, RMSE, and AUC for one simulation replicate.

    Parameters
    ----------
    tau_hat  : (G,) float — estimated treatment effects
    se       : (G,) float — standard errors (for z-score and AUC)
    tau_true : (G,) float — ground-truth effects (0 for null genes)
    is_null  : (G,) bool  — True for null genes
    alpha    : float       — significance threshold after BH correction

    Returns
    -------
    dict with keys: tpr, fpr, bias, rmse, auc
    """
    is_de = ~is_null

    # p-values from z-scores
    with np.errstate(divide='ignore', invalid='ignore'):
        z = tau_hat / np.where(se > 0, se, np.nan)
    pval = 2 * stats.norm.sf(np.abs(z))
    pval = np.where(np.isnan(pval), 1.0, pval)

    # BH correction
    _, qval, _, _ = multipletests(pval, method='fdr_bh')
    sig = qval < alpha

    # TPR (power): fraction of DE genes detected
    tpr = sig[is_de].mean() if is_de.any() else np.nan

    # FPR: fraction of null genes falsely called DE
    fpr = sig[is_null].mean() if is_null.any() else np.nan

    # Relative bias: compute raw bias per replicate, then divide by SD(tau_true[DE]).
    # This standardises the *distribution* of biases across replicates by the
    # spread of the true signal, giving a dimensionless relative measure.
    # Meaningful only for methods whose tau_hat is on the same scale as tau_true
    # (PLM, t-test, GEE coefficient); NaN otherwise (handled in figures.py).
    #
    #   bias_raw = mean(tau_hat[DE] - tau_true[DE])
    #   bias_rel = bias_raw / SD(tau_true[DE])
    if is_de.any():
        sd_true = np.nanstd(tau_true[is_de])
        diff    = tau_hat[is_de] - tau_true[is_de]
        bias_raw = diff.mean()
        bias     = bias_raw / sd_true if sd_true > 1e-10 else np.nan
        rmse     = np.sqrt((diff ** 2).mean()) / sd_true if sd_true > 1e-10 else np.nan
    else:
        bias = rmse = np.nan

    # AUC: discriminate DE from null by |z-score|
    if is_de.any() and is_null.any():
        labels = is_de.astype(int)
        score = np.abs(np.where(np.isnan(z), 0.0, z))
        # Wilcoxon-based AUC: proportion of (de, null) pairs where de has higher score
        de_scores = score[is_de]
        null_scores = score[is_null]
        # Vectorised: each de vs each null
        auc = (de_scores[:, None] > null_scores[None, :]).mean()
    else:
        auc = np.nan

    return {'tpr': tpr, 'fpr': fpr, 'bias': bias, 'rmse': rmse, 'auc': auc}


def compute_null_pvalues(tau_hat, se, is_null):
    """
    Two-sided z-test p-values for null genes only (for calibration diagnostics:
    null p-value histograms and QQ plots).

    Parameters
    ----------
    tau_hat : (G,) float
    se      : (G,) float
    is_null : (G,) bool

    Returns
    -------
    pval_null : (n_null,) float
    """
    tau_null = tau_hat[is_null]
    se_null  = se[is_null]
    with np.errstate(divide='ignore', invalid='ignore'):
        z = tau_null / np.where(se_null > 0, se_null, np.nan)
    pval = 2 * stats.norm.sf(np.abs(z))
    return np.where(np.isnan(pval), 1.0, pval)


def compute_ci_coverage(tau_hat, se, tau_true, is_null, level=0.95):
    """
    Empirical Wald CI coverage for null and DE genes.

    For null genes, coverage is the fraction of CIs containing 0 (the true
    value). For DE genes, coverage is the fraction of CIs containing the
    gene-specific true effect tau_true.

    Parameters
    ----------
    tau_hat  : (G,) float
    se       : (G,) float
    tau_true : (G,) float
    is_null  : (G,) bool
    level    : float — nominal CI level (default 0.95)

    Returns
    -------
    dict with keys: coverage_null, coverage_de
    """
    z_crit = stats.norm.isf((1 - level) / 2)
    lo = tau_hat - z_crit * se
    hi = tau_hat + z_crit * se

    is_de = ~is_null
    if is_null.any():
        cov_null = ((lo[is_null] <= 0) & (0 <= hi[is_null])).mean()
    else:
        cov_null = np.nan
    if is_de.any():
        cov_de = ((lo[is_de] <= tau_true[is_de]) & (tau_true[is_de] <= hi[is_de])).mean()
    else:
        cov_de = np.nan
    return {'coverage_null': cov_null, 'coverage_de': cov_de}


def compute_reconstruction_metrics(Y_pred, Y_pseudo, Y_true):
    """
    Compute RMSE of Y_pred and Y_pseudo against Y_true (gene-averaged).

    Used for the augmentation panel: shows whether Pearson correction brings
    the pseudo-outcome closer to the true latent expression.

    Returns
    -------
    rmse_pred  : float — RMSE of raw imputation
    rmse_pseudo: float — RMSE of augmented pseudo-outcome
    """
    rmse_pred  = np.sqrt(np.mean((Y_pred  - Y_true) ** 2))
    rmse_pseudo = np.sqrt(np.mean((Y_pseudo - Y_true) ** 2))
    return rmse_pred, rmse_pseudo
