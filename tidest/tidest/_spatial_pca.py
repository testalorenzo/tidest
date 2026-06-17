import os
import subprocess
import importlib.resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def _bundled_rscript_path():
    """Return the path to the bundled spatialPCA_runner.R."""
    try:
        # Python 3.9+
        ref = importlib.resources.files("tidest") / "R" / "spatialPCA_runner.R"
        return str(ref)
    except AttributeError:
        import pkg_resources
        return pkg_resources.resource_filename("tidest", "R/spatialPCA_runner.R")


def estimate_U(
    pseudo,
    confounder_genes,
    n_pcs=10,
    tmp_dir="./tidest_tmp",
    rscript_path=None,
):
    """Estimate spatial confounders via SpatialPCA (R subprocess).

    Writes feather files to tmp_dir, calls Rscript, reads back PCs.

    Parameters
    ----------
    pseudo : AnnData
        Pseudo-outcome with .X in log space and .obsm['spatial'].
    confounder_genes : list[str]
        Confounder gene names (must be a subset of pseudo.var_names).
    n_pcs : int
        Number of spatial PCs to extract.
    tmp_dir : str or Path
        Directory where feather files and the gene list are written.
    rscript_path : str or None
        Path to the R script. None uses the bundled spatialPCA_runner.R.

    Returns
    -------
    U_df : pd.DataFrame
        Index = spot IDs, columns = SpatialPC1 … SpatialPC{n_pcs}.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if rscript_path is None:
        rscript_path = _bundled_rscript_path()

    # Write confounder gene list
    gene_list_path = tmp_dir / "confounder_genes.txt"
    with open(gene_list_path, "w") as f:
        f.write("\n".join(confounder_genes))

    # Write gene × spot counts feather
    X = np.asarray(pseudo.X.toarray() if issparse(pseudo.X) else pseudo.X).astype(np.float32)
    counts_df = pd.DataFrame(X.T, index=pseudo.var_names, columns=pseudo.obs_names)
    counts_path = tmp_dir / "pseudo_counts.feather"
    counts_df.reset_index(names="gene").to_feather(str(counts_path))

    # Write spot × xy location feather
    spatial = pseudo.obsm["spatial"]
    loc_df = pd.DataFrame(spatial, index=pseudo.obs_names, columns=["x", "y"])
    loc_path = tmp_dir / "pseudo_location.feather"
    loc_df.reset_index(names="spot").to_feather(str(loc_path))

    print(f'  Running SpatialPCA (n_pcs={n_pcs}, {len(confounder_genes)} genes)...', flush=True)
    result = subprocess.run(
        ["Rscript", rscript_path, str(tmp_dir), str(n_pcs)],
        check=True,
        capture_output=False,
    )

    pcs_path = tmp_dir / "spatialPCA_pcs.feather"
    U_df = pd.read_feather(str(pcs_path)).set_index("spot")
    print(f'  SpatialPCA done: {U_df.shape[0]} spots x {U_df.shape[1]} PCs', flush=True)
    return U_df
