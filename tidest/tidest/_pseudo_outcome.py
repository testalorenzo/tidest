import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import normalize
from scanpy.preprocessing import normalize_total


def compute_pearson(sc_adata):
    """Compute gene-gene Pearson correlation matrix on 10k-normalized log-CPM of sc_adata.

    Parameters
    ----------
    sc_adata : AnnData
        Single-cell reference. Raw counts expected (normalize_total is applied internally).

    Returns
    -------
    pearson : ndarray (n_sc_genes, n_sc_genes)
    pearson_genes : list[str]
        Gene names (lowercase) corresponding to pearson axes.
    """
    sc = sc_adata.copy()
    normalize_total(sc, target_sum=1e4)

    X = sc.X
    if issparse(X):
        X = X.toarray()
    X = np.log1p(X.astype(np.float32))
    X -= X.mean(axis=0)
    X_norm = normalize(X, norm='l2', axis=0, copy=False)
    pearson = X_norm.T @ X_norm  # (n_sc_genes, n_sc_genes)

    pearson_genes = [g.lower() for g in sc.var_names]
    return pearson, pearson_genes


def build_pseudo_outcome(
    sc_adata,
    st_adata,
    pred_adata,
    pearson=None,
    pearson_genes=None,
    top_k=5,
    min_corr=None,
    pred_is_log=False,
):
    """Build Pearson-corrected pseudo-outcome from imputed predictions.

    Parameters
    ----------
    sc_adata : AnnData
        Single-cell reference. Used only if pearson is None.
    st_adata : AnnData
        Spatial transcriptomics data with raw counts and obsm['spatial'].
    pred_adata : AnnData
        Imputed predictions aligned to st_adata.obs (same spot order).
        If pred_is_log=False (default): count space (per-spot scaled for Tangram).
        If pred_is_log=True: already log-transformed (CellPLM).
    pearson : ndarray or None
        Precomputed (n_sc_genes, n_sc_genes) Pearson matrix. Computed from
        sc_adata if None.
    pearson_genes : list[str] or None
        Gene names (lowercase) for pearson axes. Required if pearson is given.
    top_k : int
        Number of Pearson neighbors used for the residual correction.
    min_corr : float or None
        Minimum |Pearson r| required for a candidate neighbor to be used.
        Neighbors below this threshold are dropped (and the per-gene
        normalization is by the number of neighbors actually retained,
        not by top_k). Default None: no thresholding (all top_k candidates
        are used, matching the original behavior).
    pred_is_log : bool
        True if pred_adata.X is already log-transformed (e.g. CellPLM output).

    Returns
    -------
    pseudo : AnnData
        Pseudo-outcome. .X is in log-count space (float32).
        .obsm['spatial'] is populated from st_adata.
        .var_names are lowercase.
    pearson : ndarray
    pearson_genes : list[str]
    """
    if pearson is None:
        print('Computing Pearson correlation matrix from sc_adata...', flush=True)
        pearson, pearson_genes = compute_pearson(sc_adata)
    elif pearson_genes is None:
        raise ValueError("pearson_genes must be provided when pearson is given.")

    sc_gene_to_idx = {g: i for i, g in enumerate(pearson_genes)}

    # Work on copies with lowercase var_names
    pred = pred_adata.copy()
    pred.var_names = pd.Index(pred.var_names.str.lower())
    if pred.var_names.duplicated().any():
        pred = pred[:, ~pred.var_names.duplicated()].copy()

    st = st_adata.copy()
    st.var_names = pd.Index(st.var_names.str.lower())

    # Restrict ST to genes present in pred
    st = st[:, st.var_names.isin(pred.var_names)].copy()

    # Per-spot scaling (Tangram case only): maps weighted-cell-average to ST count scale
    if not pred_is_log:
        N_s = np.asarray(st.X.sum(axis=1)).ravel()
        pred_obs = pred[:, st.var_names]
        pred_obs_mat = np.asarray(pred_obs.X.toarray() if issparse(pred_obs.X) else pred_obs.X)
        ge_obs_sums = pred_obs_mat.sum(axis=1)
        zero_spots = ge_obs_sums == 0
        if zero_spots.any():
            print(f'  WARNING: {zero_spots.sum()} spot(s) with zero predicted expression; '
                  f'scale set to 1 for those spots.', flush=True)
        scale = N_s / np.where(zero_spots, 1.0, ge_obs_sums)
        pred_X = np.asarray(pred.X.toarray() if issparse(pred.X) else pred.X).astype(np.float32)
        pred.X = pred_X * scale[:, None]

    # Genes observed in both ST and pred
    observed_genes = pred.var_names.intersection(st.var_names)

    # Keep only genes that have a Pearson index
    pred = pred[:, [g in sc_gene_to_idx for g in pred.var_names]].copy()
    observed_genes = observed_genes[[g in sc_gene_to_idx for g in observed_genes]]
    n_all = pred.n_vars
    n_obs = len(observed_genes)

    # Pearson sub-matrix: (n_all, n_obs)
    all_sc_idx = np.array([sc_gene_to_idx[g] for g in pred.var_names])
    obs_sc_idx = np.array([sc_gene_to_idx[g] for g in observed_genes])
    pearson_sub = pearson[np.ix_(all_sc_idx, obs_sc_idx)]

    # Top-k Pearson neighbors per gene, optionally thresholded by min_corr
    top_k_actual = min(top_k, n_obs)
    if min_corr is not None:
        pearson_sub = np.where(pearson_sub >= min_corr, pearson_sub, -np.inf)
    if top_k_actual < n_obs:
        top_k_idx = np.argpartition(-pearson_sub, top_k_actual, axis=1)[:, :top_k_actual]
    else:
        top_k_idx = np.tile(np.arange(n_obs), (n_all, 1))
    weights = pearson_sub[np.arange(n_all)[:, None], top_k_idx]  # (n_all, top_k)

    if min_corr is not None:
        valid = np.isfinite(weights)
        weights = np.where(valid, weights, 0.0)
        n_neighbors = np.maximum(valid.sum(axis=1), 1).astype(np.float32)  # (n_all,)
    else:
        n_neighbors = np.full(n_all, top_k_actual, dtype=np.float32)

    # Residuals in log space
    st_obs_mat = np.asarray(
        st[:, observed_genes].X.toarray() if issparse(st[:, observed_genes].X)
        else st[:, observed_genes].X
    ).astype(np.float32)
    pred_obs_mat = np.asarray(pred[:, observed_genes].X).astype(np.float32)

    if pred_is_log:
        log_obs = st_obs_mat   # already log-CPM
        log_pred_obs = pred_obs_mat
    else:
        log_obs = np.log1p(st_obs_mat)
        log_pred_obs = np.log1p(pred_obs_mat)

    diff = (log_obs - log_pred_obs).astype(np.float32)  # (n_spots, n_obs)

    # Sparse weight matrix (n_all x n_obs)
    row_idx = np.repeat(np.arange(n_all), top_k_actual)
    W_sparse = csr_matrix(
        (weights.ravel(), (row_idx, top_k_idx.ravel())),
        shape=(n_all, n_obs),
    )

    correction = (diff @ W_sparse.T) / n_neighbors[None, :]  # (n_spots, n_all)

    pred_X = np.asarray(pred.X).astype(np.float32)
    log_pred_all = pred_X if pred_is_log else np.log1p(pred_X)
    log_pseudo = log_pred_all + correction  # (n_spots, n_all)

    pseudo = pred.copy()
    pseudo.X = log_pseudo.astype(np.float32)
    pseudo.obsm['spatial'] = st.obsm['spatial']

    print(f'Pseudo-outcome: {pseudo.n_obs} spots x {pseudo.n_vars} genes', flush=True)
    return pseudo, pearson, pearson_genes
