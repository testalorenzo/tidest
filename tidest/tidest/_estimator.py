import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from ._pseudo_outcome import build_pseudo_outcome
from ._confounders import select_confounders
from ._spatial_pca import estimate_U
from ._plm import robinson_plm
from ._utils import lookup_genes, parse_treatment


class tidest:
    """Testing Imputed Differential Expressions for Spatial Transcriptomics.

    Estimates the causal effect of a binary spatial treatment on gene
    expression using a Robinson (1988) partially-linear model with
    Pearson-corrected imputation as the pseudo-outcome.

    Parameters
    ----------
    n_pcs : int
        Number of SpatialPCA components to use as confounders.
    n_folds : int
        Number of cross-fitting folds for the Robinson PLM.
    n_estimators : int
        Number of trees in each random forest (nuisance models).
    corr_threshold : float
        Drop spatial PCs whose |corr(PC, A)| >= this threshold before
        building the confounder matrix (prevents over-controlling).
    cv_threshold : float
        Maximum CV of log-expression across spots for a gene to be
        considered a confounder candidate (treatment-independence filter).
    marker_r_thresh : float
        Maximum |Pearson r| to any outcome gene in SC space for a gene
        to remain a confounder candidate (outcome-independence filter).
    n_confounder_genes : int
        Number of top-spatial-variance confounder genes passed to SpatialPCA.
    top_k : int
        Number of Pearson neighbors used in the pseudo-outcome correction.
    spatialPCA_rscript : str or None
        Path to a custom R script for SpatialPCA. None uses the bundled
        spatialPCA_runner.R. Ignored when U is supplied to fit().
    tmp_dir : str
        Directory for feather files exchanged with R.
    seed : int
        Random seed for reproducibility.

    Notes
    -----
    R and the SpatialPCA package must be installed when SpatialPCA is used
    for confounder estimation. Pass U= to fit() to bypass R entirely.
    """

    def __init__(
        self,
        n_pcs=10,
        n_folds=2,
        n_estimators=200,
        corr_threshold=0.5,
        cv_threshold=0.3,
        marker_r_thresh=0.3,
        n_confounder_genes=2000,
        top_k=5,
        spatialPCA_rscript=None,
        tmp_dir="./tidest_tmp",
        seed=42,
    ):
        self.n_pcs = n_pcs
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.corr_threshold = corr_threshold
        self.cv_threshold = cv_threshold
        self.marker_r_thresh = marker_r_thresh
        self.n_confounder_genes = n_confounder_genes
        self.top_k = top_k
        self.spatialPCA_rscript = spatialPCA_rscript
        self.tmp_dir = tmp_dir
        self.seed = seed

    def _screen_pcs(self, U_full, A):
        """Drop spatial PCs whose |corr| with treatment >= corr_threshold."""
        corr_mat = np.corrcoef(np.column_stack([U_full, A]).T)
        pc_corrs = corr_mat[:-1, -1]
        keep_mask = np.abs(pc_corrs) < self.corr_threshold
        dropped = np.where(~keep_mask)[0]
        if len(dropped):
            print(f'  Dropped PCs (|corr|>={self.corr_threshold}): '
                  f'{[int(d+1) for d in dropped]}, '
                  f'corr={[round(pc_corrs[d], 3) for d in dropped]}',
                  flush=True)
        print(f'  Kept {keep_mask.sum()} / {len(keep_mask)} PCs', flush=True)
        return U_full[:, keep_mask]

    def fit(
        self,
        sc_adata=None,
        st_adata=None,
        pred_adata=None,
        treatment=None,
        genes=None,
        treatment_val=None,
        pearson=None,
        pearson_genes=None,
        pseudo=None,
        U=None,
        pred_is_log=False,
    ):
        """Fit the TIDEST model.

        Parameters
        ----------
        sc_adata : AnnData or None
            Single-cell reference for Pearson computation. Not required when
            pseudo= is supplied or when pearson= is pre-supplied.
        st_adata : AnnData or None
            Spatial transcriptomics data with raw counts and .obsm['spatial'].
            Not required when pseudo= is supplied.
        pred_adata : AnnData or None
            Imputed predictions (Tangram or CellPLM output). Not required when
            pseudo= is supplied.
        treatment : str or array-like
            If str: column name in the pseudo/st obs DataFrame; use
            treatment_val= for the A=1 level.
            If array-like: pre-computed binary (0/1) array.
        genes : list[str] or None
            Gene names to test. None tests all genes in the pseudo-outcome.
        treatment_val : scalar or None
            Value in obs[treatment] encoding A=1. Required when treatment is str.
        pearson : ndarray (n_sc_genes, n_sc_genes) or None
            Precomputed Pearson matrix. Computed from sc_adata if None and
            pseudo= is not provided.
        pearson_genes : list[str] or None
            Lowercase gene names for pearson axes. Required when pearson= given.
        pseudo : AnnData or None
            Pre-built pseudo-outcome (skips build_pseudo_outcome entirely).
            .X must be in log space; .obs must contain the treatment column.
        U : ndarray (n_spots, n_features) or None
            Precomputed confounder matrix. Skips SpatialPCA when provided.
        pred_is_log : bool
            True if pred_adata.X is already log-transformed (CellPLM).

        Returns
        -------
        self
        """
        # --- Step 1 & 2: Pseudo-outcome ---
        if pseudo is not None:
            print('Using pre-built pseudo-outcome.', flush=True)
            self.pseudo_ = pseudo
        else:
            if pred_adata is None or st_adata is None:
                raise ValueError(
                    "st_adata and pred_adata are required when pseudo= is not supplied."
                )
            print('Building pseudo-outcome...', flush=True)
            self.pseudo_, pearson, pearson_genes = build_pseudo_outcome(
                sc_adata=sc_adata,
                st_adata=st_adata,
                pred_adata=pred_adata,
                pearson=pearson,
                pearson_genes=pearson_genes,
                top_k=self.top_k,
                pred_is_log=pred_is_log,
            )

        pseudo = self.pseudo_
        spot_order = pseudo.obs_names.tolist()

        # --- Treatment vector (aligned to pseudo spot order) ---
        A = parse_treatment(treatment, treatment_val, pseudo.obs)
        print(f'Treatment: {A.sum()} A=1 / {(1-A).sum()} A=0', flush=True)

        # --- Step 3: Confounder matrix U ---
        if U is not None:
            print('Using user-supplied confounder matrix U.', flush=True)
            if hasattr(U, 'values'):
                U_mat = U.loc[spot_order].values.astype(np.float64)
            else:
                U_mat = np.asarray(U, dtype=np.float64)
            U_full = U_mat[:, :self.n_pcs]
            self.confounder_genes_ = None
        else:
            outcome_genes_lower = (
                [g.lower() for g in genes] if genes is not None
                else list(pseudo.var_names)
            )

            print('Selecting confounder genes...', flush=True)
            self.confounder_genes_ = select_confounders(
                pseudo=pseudo,
                pearson=pearson,
                pearson_genes=pearson_genes,
                outcome_genes=outcome_genes_lower,
                cv_threshold=self.cv_threshold,
                marker_r_thresh=self.marker_r_thresh,
                n_confounder_genes=self.n_confounder_genes,
            )
            if not self.confounder_genes_:
                raise ValueError(
                    "No confounder genes passed both filters. "
                    "Consider relaxing cv_threshold or marker_r_thresh."
                )

            U_df = estimate_U(
                pseudo=pseudo,
                confounder_genes=self.confounder_genes_,
                n_pcs=self.n_pcs,
                tmp_dir=self.tmp_dir,
                rscript_path=self.spatialPCA_rscript,
            )
            U_full = U_df.loc[spot_order].values[:, :self.n_pcs].astype(np.float64)

        # PC correlation screening: drop PCs whose |corr(PC, A)| >= corr_threshold
        U_arr = self._screen_pcs(U_full, A)

        # Append log library size to absorb spot-level count differences
        X_pseudo = np.asarray(
            pseudo.X.toarray() if issparse(pseudo.X) else pseudo.X
        ).astype(np.float64)
        log_lib = np.log1p(np.expm1(X_pseudo).sum(axis=1, keepdims=True))
        U_arr = np.hstack([U_arr, log_lib])
        self.U_ = U_arr
        print(f'  U: {U_arr.shape[0]} spots x {U_arr.shape[1]} features '
              f'(PCs + log-libsize)', flush=True)

        # --- Step 4: Outcome matrix Y ---
        found, missing = lookup_genes(genes, pseudo.var_names)
        if missing:
            print(f'  WARNING: {len(missing)} gene(s) not found in pseudo-outcome '
                  f'(first 5): {missing[:5]}', flush=True)
        if not found:
            raise ValueError("No outcome genes found in pseudo-outcome.")

        Y = np.asarray(pseudo[spot_order, found].X).astype(np.float64)
        print(f'  Y: {Y.shape[0]} spots x {Y.shape[1]} genes (log)', flush=True)

        # --- Step 5: Robinson PLM ---
        print('Fitting Robinson PLM...', flush=True)
        tau, se = robinson_plm(
            Y=Y,
            A=A,
            U=U_arr,
            n_folds=self.n_folds,
            n_estimators=self.n_estimators,
            seed=self.seed,
        )

        # --- Step 6: Inference ---
        z = tau / np.where(se > 0, se, np.nan)
        pval = 2 * stats.norm.sf(np.abs(z))

        valid = ~np.isnan(pval)
        qval_full = np.full(len(pval), np.nan)
        if valid.any():
            _, qval, _, _ = multipletests(pval[valid], method='fdr_bh')
            qval_full[valid] = qval

        self.results_ = pd.DataFrame({
            'gene': found,
            'tau':  tau,
            'se':   se,
            'z':    z,
            'pval': pval,
            'qval': qval_full,
        }).sort_values('pval').reset_index(drop=True)

        n_sig = (self.results_['qval'] < 0.05).sum()
        print(f'Done. Significant (q<0.05): {n_sig} / {len(self.results_)}', flush=True)
        return self
