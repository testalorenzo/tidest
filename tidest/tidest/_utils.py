import numpy as np


def gene_index_lower(var_names):
    """Return a dict mapping lowercase gene name -> original name."""
    return {g.lower(): g for g in var_names}


def lookup_genes(genes, var_names):
    """Case-insensitive lookup of genes in var_names.

    Parameters
    ----------
    genes : list[str] or None
        Requested gene names. If None, returns all var_names.
    var_names : Index
        AnnData var_names.

    Returns
    -------
    found : list[str]
        Names as they appear in var_names (original case).
    missing : list[str]
        Requested names not found.
    """
    if genes is None:
        return list(var_names), []

    idx = gene_index_lower(var_names)
    found = []
    missing = []
    for g in genes:
        key = g.lower()
        if key in idx:
            found.append(idx[key])
        else:
            missing.append(g)
    return found, missing


def parse_treatment(treatment, treatment_val, obs):
    """Return a binary (0/1) numpy array for the treatment indicator.

    Parameters
    ----------
    treatment : str or array-like
        Column name in obs, or a pre-computed binary array.
    treatment_val : scalar or None
        The value in obs[treatment] that corresponds to A=1.
        Required when treatment is a string.
    obs : pd.DataFrame
        AnnData.obs of the spatial data.

    Returns
    -------
    A : ndarray (n_spots,) int
    """
    if isinstance(treatment, str):
        if treatment_val is None:
            raise ValueError(
                "treatment_val must be provided when treatment is a column name."
            )
        A = (obs[treatment] == treatment_val).astype(int).values
    else:
        A = np.asarray(treatment, dtype=int)
    return A
