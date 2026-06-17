import numpy as np
from scipy.sparse import issparse


def select_confounders(
    pseudo,
    pearson,
    pearson_genes,
    outcome_genes,
    cv_threshold=0.3,
    marker_r_thresh=0.3,
    n_confounder_genes=2000,
):
    """Select confounder genes for SpatialPCA from the pseudo-outcome.

    Two-stage negative-control filter:
    1. Low spatial CV (treatment-independent): CV < cv_threshold on log-CPM
       across spots, excluding outcome genes.
    2. Low SC co-expression (outcome-independent): max |Pearson r| <
       marker_r_thresh to any outcome gene in the SC reference.

    Then ranks by spatial variance and returns the top n_confounder_genes.

    Parameters
    ----------
    pseudo : AnnData
        Pseudo-outcome with .X in log space and lowercase var_names.
    pearson : ndarray (n_sc_genes, n_sc_genes)
    pearson_genes : list[str]
        Lowercase gene names for pearson axes.
    outcome_genes : list[str]
        Lowercase names of genes being tested (excluded from confounders).
    cv_threshold : float
        Maximum allowed CV of log-expression across spots.
    marker_r_thresh : float
        Maximum allowed |Pearson r| to any outcome gene in SC space.
    n_confounder_genes : int
        How many top-variance genes to pass to SpatialPCA.

    Returns
    -------
    confounder_genes : list[str]
        Gene names (lowercase) to pass to SpatialPCA.
    """
    pearson_gene_to_idx = {g: i for i, g in enumerate(pearson_genes)}
    outcome_set = set(g.lower() for g in outcome_genes)
    marker_sc_idx = [pearson_gene_to_idx[g] for g in outcome_set if g in pearson_gene_to_idx]

    gene_names = np.array(pseudo.var_names.tolist())
    X_log = np.asarray(pseudo.X.toarray() if issparse(pseudo.X) else pseudo.X).astype(np.float64)

    # Filter 1: low spatial CV, excluding outcome genes
    gene_mean = X_log.mean(axis=0)
    gene_cv = np.where(gene_mean > 0.1, X_log.std(axis=0) / gene_mean, np.inf)
    not_outcome = np.array([g not in outcome_set for g in gene_names])
    candidate_idx = np.where((gene_cv < cv_threshold) & not_outcome)[0]
    print(f'  Confounder filter 1: {len(candidate_idx)} low-CV candidates '
          f'(CV<{cv_threshold}, excl. outcomes)', flush=True)

    # Filter 2: low |Pearson r| to any outcome gene in SC space
    candidate_names = gene_names[candidate_idx]
    in_pearson_mask = np.array([g in pearson_gene_to_idx for g in candidate_names])
    cand_in_pearson = np.where(in_pearson_mask)[0]
    candidate_sc_idx = [pearson_gene_to_idx[candidate_names[i]] for i in cand_in_pearson]

    if len(candidate_sc_idx) == 0 or len(marker_sc_idx) == 0:
        final_idx = candidate_idx
    else:
        max_abs_r = np.abs(pearson[np.ix_(candidate_sc_idx, marker_sc_idx)]).max(axis=1)
        final_local = cand_in_pearson[max_abs_r < marker_r_thresh]
        final_idx = candidate_idx[final_local]

    print(f'  Confounder filter 2: {len(final_idx)} after |r|<{marker_r_thresh} SC filter',
          flush=True)

    # Rank by spatial variance, take top N
    top_idx = final_idx[np.argsort(-X_log.var(axis=0)[final_idx])[:n_confounder_genes]]
    confounder_genes = gene_names[top_idx].tolist()
    print(f'  Using top {len(confounder_genes)} by spatial variance for SpatialPCA', flush=True)

    return confounder_genes
