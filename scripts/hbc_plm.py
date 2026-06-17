"""
HBC (IBC vs DCIS) PLM using tidest.

Input (pre-computed by hbc_cellplm_raw.py):
  results_raw/hbc_cellplm_pred{rep}.pkl

Runs internally:
  1. Pearson matrix from SC reference
  2. Pseudo-outcome construction (Pearson correction in log-CPM space)
  3. Confounder gene selection (low-CV, low Pearson-r with markers)
  4. SpatialPCA on N_CELLS_SPCA stratified subsample (via hbc_spatialPCA.R)
  5. Nyström projection of all cells onto SpatialPCA eigenvectors
  6. Robinson PLM on all cells

Output:
  results_raw/hbc_plm_results{rep}.csv
"""

import io, pathlib, subprocess, pickle
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import torch
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

from tidest import tidest

# ── Pipeline constants ─────────────────────────────────────────────────────────
N_REPS          = 1
TOP_K           = 5           # neighbours for Pearson correction
CV_THRESHOLD    = 1.5         # max CV for confounder candidate genes
MARKER_R_THRESH = 0.3         # max |Pearson r| with marker genes
N_NONHVG        = 2000        # top spatial-variance confounder genes for SpatialPCA
N_CELLS_SPCA    = 10000       # SpatialPCA training subsample (10k×10k kernel = 0.8 GB)
N_PCS_USE       = 50
N_FOLDS         = 2
N_ESTIMATORS    = 200
CORR_THRESHOLD  = 0.5

ANNOTATIONS_PATH = './Xenium'
ZARR_PATH        = './Xenium.zarr'
SC_H5_NAME       = ('Chromium_FFPE_Human_Breast_Cancer_'
                    'Chromium_FFPE_Human_Breast_Cancer_'
                    'count_sample_filtered_feature_bc_matrix.h5')
IBC_CLUSTERS  = ['Invasive_Tumor', 'Prolif_Invasive_Tumor']
DCIS_CLUSTERS = ['DCIS_1', 'DCIS_2']
ALL_CLUSTERS  = IBC_CLUSTERS + DCIS_CLUSTERS

MARKER_GENES = [
    # === ORIGINAL PANEL (44 genes) ===
    # IBC-enriched (expect positive tau = higher in Invasive_Tumor)
    'MKI67','TOP2A','PCNA','CDK1','CCNB1','MCM2','MCM6',
    'VIM','FN1','CDH2','TWIST1','SNAI1','SNAI2','ZEB1','ZEB2',
    'MMP2','MMP9','MMP11','CXCR4',
    'EGFR','ERBB2','S100A8','S100A9','S100A4',
    'KRT5','KRT14','CD44',
    # DCIS-enriched (expect negative tau = higher in DCIS)
    'ESR1','PGR','FOXA1','GATA3','TFF1','TFF3','AREG',
    'KRT8','KRT18','KRT19',
    'CDH1','CLDN3','CLDN4','CLDN7',
    'SPDEF','MLPH','AGR2',
    # === EXTENDED PANEL (25 new genes) ===
    # Myoepithelial layer (expect DCIS-enriched)
    'CNN1','TP63','MYLK','ITGB6',
    # Basement membrane (expect DCIS-enriched)
    'LAMB3','LAMC2','COL4A1',
    # Luminal TFs and differentiation (DCIS except EZH2)
    'RUNX1','ERBB4','KIT','EZH2',
    # IBC-enriched stroma/ECM
    'COL11A1','POSTN','CXCL10','THBS2','GREM1',
    # Immune TME (TIGIT DCIS-enriched; rest IBC)
    'TIGIT','CTLA4','NKG7','GNLY','TYROBP',
    # Contested biology
    'CDH3','CA9','CDKN2A','ALDH1A1',
]


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


# ── Step 1: Pearson matrix from SC reference ──────────────────────────────────

def build_pearson():
    print('Building Pearson matrix from SC reference...')
    sc_data = sc.read_10x_h5(pathlib.Path(ANNOTATIONS_PATH) / SC_H5_NAME)

    annotations_sc = pd.read_excel(
        pathlib.Path(ANNOTATIONS_PATH) / 'Cell_Barcode_Type_Matrices.xlsx',
        sheet_name='scFFPE-Seq',
    )
    sc_data = sc_data[sc_data.obs_names.isin(annotations_sc['Barcode'])].copy()

    sc.pp.calculate_qc_metrics(sc_data, inplace=True, log1p=True)
    sc.pp.filter_cells(sc_data, min_genes=100)
    sc.pp.filter_genes(sc_data, min_cells=3)
    sc.pp.normalize_total(sc_data, target_sum=1e4)

    X = sc_data.X
    if issparse(X):
        X = X.toarray()
    X = np.log1p(X.astype(np.float32))
    X -= X.mean(axis=0)
    X_norm = normalize(X, norm='l2', axis=0, copy=False)
    pearson = X_norm.T @ X_norm

    sc_genes = pd.Index(sc_data.var_names.str.lower())
    print(f'  Pearson: {pearson.shape[0]} × {pearson.shape[1]} genes')

    with open('./results_raw/hbc_pearson.pkl', 'wb') as f:
        pickle.dump(pearson, f)
    with open('./results_raw/hbc_pearson_genes.pkl', 'wb') as f:
        pickle.dump(list(sc_genes), f)

    return pearson, sc_genes


# ── Step 2: Pseudo-outcome construction ───────────────────────────────────────

def build_pseudo_outcome(rep, pearson, sc_genes, st_data):
    print(f'  Building pseudo-outcome (rep {rep})...')
    sc_gene_to_idx = {g: i for i, g in enumerate(sc_genes)}

    with open(f'./results_raw/hbc_cellplm_pred{rep}.pkl', 'rb') as f:
        ad_ge = _CpuUnpickler(f).load()

    # ENSG → gene symbols (lowercase); drop duplicates
    symbols = pd.Index(ad_ge.var['gene_symbol'].str.lower())
    dup     = symbols.duplicated(keep='first')
    ad_ge   = ad_ge[:, ~dup].copy()
    ad_ge.var_names = symbols[~dup]

    # Filter to IBC + DCIS; merge subtypes
    ad_ge2 = ad_ge[ad_ge.obs['Cluster'].isin(ALL_CLUSTERS)].copy()
    ad_ge2.obs.loc[ad_ge2.obs['Cluster'] == 'Prolif_Invasive_Tumor', 'Cluster'] = 'Invasive_Tumor'
    ad_ge2.obs.loc[ad_ge2.obs['Cluster'] == 'DCIS_2',                'Cluster'] = 'DCIS_1'

    ad_ge2 = ad_ge2[:, [g for g in ad_ge2.var_names if g in sc_gene_to_idx]].copy()

    common_spots = ad_ge2.obs_names.intersection(st_data.obs_names)
    ad_ge2    = ad_ge2[common_spots].copy()
    adata_st2 = st_data[common_spots].copy()
    adata_st2 = adata_st2[:, adata_st2.var_names.isin(ad_ge2.var_names)].copy()

    print(f'  {ad_ge2.n_obs} cells, {ad_ge2.n_vars} pred genes, '
          f'{adata_st2.n_vars} observed genes')

    X_pred = np.asarray(ad_ge2.X).astype(np.float32)
    X_st   = (adata_st2.X.toarray() if issparse(adata_st2.X)
              else np.array(adata_st2.X)).astype(np.float32)

    observed_genes = ad_ge2.var_names.intersection(adata_st2.var_names)
    n_all = ad_ge2.n_vars

    all_sc_idx = np.array([sc_gene_to_idx[g] for g in ad_ge2.var_names])
    obs_sc_idx = np.array([sc_gene_to_idx[g] for g in observed_genes])
    pearson_sub = pearson[np.ix_(all_sc_idx, obs_sc_idx)]

    top_k   = np.argpartition(-pearson_sub, TOP_K, axis=1)[:, :TOP_K]
    weights = pearson_sub[np.arange(n_all)[:, None], top_k]

    obs_var_idx  = [list(adata_st2.var_names).index(g) for g in observed_genes]
    pred_obs_idx = [list(ad_ge2.var_names).index(g)    for g in observed_genes]

    diff = (X_st[:, obs_var_idx] - X_pred[:, pred_obs_idx]).astype(np.float32)

    row_idx  = np.repeat(np.arange(n_all), TOP_K)
    W_sparse = csr_matrix(
        (weights.ravel(), (row_idx, top_k.ravel())),
        shape=(n_all, len(observed_genes)),
    )
    correction_log = (diff @ W_sparse.T) / TOP_K
    log_pseudo     = X_pred + correction_log

    pseudo           = ad_ge2.copy()
    pseudo.X         = log_pseudo.astype(np.float32)
    pseudo.obsm['spatial'] = adata_st2.obsm['spatial']

    with open(f'./results_raw/hbc_pseudo{rep}.pkl', 'wb') as f:
        pickle.dump(pseudo, f)
    print(f'  Saved hbc_pseudo{rep}.pkl, shape={pseudo.shape}')
    return pseudo


# ── Step 3 + 4 + 5: Confounder selection + SpatialPCA + Nyström ──────────────

def run_confounding_and_spatialPCA(rep, pseudo, pearson, sc_genes):
    print(f'  Confounder selection + SpatialPCA (rep {rep})...')
    marker_set = {m.lower() for m in MARKER_GENES}
    sc_gene_to_idx = {g: i for i, g in enumerate(sc_genes)}

    gene_names = np.array(pseudo.var_names.tolist())
    X_log      = np.asarray(pseudo.X).astype(np.float64)

    # Filter 1: low CV
    gene_mean = X_log.mean(axis=0)
    gene_cv   = np.where(gene_mean > 0.01, X_log.std(axis=0) / gene_mean, np.inf)
    not_marker    = np.array([g not in marker_set for g in gene_names])
    candidate_idx = np.where((gene_cv < CV_THRESHOLD) & not_marker)[0]
    print(f'  {len(candidate_idx)} low-CV candidates (CV<{CV_THRESHOLD}, excl. markers)')

    # Filter 2: low Pearson r with markers
    marker_pearson_idx = [sc_gene_to_idx[m] for m in marker_set if m in sc_gene_to_idx]
    candidate_names  = gene_names[candidate_idx]
    in_pearson_mask  = np.array([g in sc_gene_to_idx for g in candidate_names])
    cand_in_pearson  = np.where(in_pearson_mask)[0]
    candidate_sc_idx = [sc_gene_to_idx[candidate_names[i]] for i in cand_in_pearson]
    max_abs_r = np.abs(pearson[np.ix_(candidate_sc_idx, marker_pearson_idx)]).max(axis=1)
    final_idx = candidate_idx[cand_in_pearson[max_abs_r < MARKER_R_THRESH]]
    print(f'  {len(final_idx)} after |r|<{MARKER_R_THRESH} SC filter')

    # Rank by spatial variance, take top N
    top_idx = final_idx[np.argsort(-X_log.var(axis=0)[final_idx])[:N_NONHVG]]
    non_hvg = gene_names[top_idx].tolist()
    print(f'  Using top {len(non_hvg)} by spatial variance for SpatialPCA')

    with open(f'./results_raw/hbc_non_hvg_genes{rep}.txt', 'w') as f:
        f.write('\n'.join(non_hvg))

    # Stratified subsample for SpatialPCA kernel fit
    clusters   = pseudo.obs['Cluster'].values
    unique_cls = np.unique(clusters)
    n_per_cls  = N_CELLS_SPCA // len(unique_cls)
    rng        = np.random.default_rng(seed=rep)
    spca_idx   = []
    for cls in unique_cls:
        cls_idx = np.where(clusters == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(n_per_cls, len(cls_idx)), replace=False)
        spca_idx.extend(chosen.tolist())
    spca_idx = np.array(sorted(spca_idx))

    pseudo_spca = pseudo[spca_idx]
    print(f'  SpatialPCA training subsample = {len(spca_idx)} cells '
          f'({dict(zip(*np.unique(clusters[spca_idx], return_counts=True)))})')

    # Export feather files for R — training subsample
    spot_ids_train = pseudo_spca.obs_names
    spatial_train  = pseudo_spca.obsm['spatial']
    X_spca         = np.asarray(pseudo_spca.X).astype('float32')
    gene_names_full = gene_names  # all genes (rows of counts matrix)

    counts_df = pd.DataFrame(X_spca.T, index=gene_names_full, columns=spot_ids_train)
    counts_df.reset_index(names='gene').to_feather(
        f'./results_raw/hbc_pseudo_counts{rep}.feather'
    )
    loc_train_df = pd.DataFrame(spatial_train, index=spot_ids_train, columns=['x', 'y'])
    loc_train_df.reset_index(names='spot').to_feather(
        f'./results_raw/hbc_pseudo_location{rep}.feather'
    )

    # Export locations of ALL cells for Nyström projection
    spatial_all  = pseudo.obsm['spatial']
    spot_ids_all = pseudo.obs_names
    loc_all_df = pd.DataFrame(spatial_all, index=spot_ids_all, columns=['x', 'y'])
    loc_all_df.reset_index(names='spot').to_feather(
        f'./results_raw/hbc_all_location{rep}.feather'
    )
    print(f'  Exported all-cell locations ({len(spot_ids_all)} cells) for Nyström projection')

    subprocess.run(['Rscript', 'scripts/hbc_spatialPCA.R'], check=True)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load ST data once (shared across reps) ────────────────────────────────
    print('Loading ST data...')
    sdata   = sd.read_zarr(ZARR_PATH)
    st_data = sdata.tables['table']

    annotations_st = pd.read_excel(
        pathlib.Path(ANNOTATIONS_PATH) / 'Cell_Barcode_Type_Matrices.xlsx',
        sheet_name='Xenium R1 Fig1-5 (supervised)',
    )
    st_data.obs = st_data.obs.merge(
        annotations_st.set_index('Barcode').loc[
            [int(x) + 1 for x in st_data.obs_names], 'Cluster'
        ],
        left_on='cell_id', right_index=True, how='left',
    )
    st_data = st_data[st_data.obs['Cluster'].isin(ALL_CLUSTERS)].copy()
    st_data.obs.loc[st_data.obs['Cluster'] == 'Prolif_Invasive_Tumor', 'Cluster'] = 'Invasive_Tumor'
    st_data.obs.loc[st_data.obs['Cluster'] == 'DCIS_2',                'Cluster'] = 'DCIS_1'
    sc.pp.normalize_total(st_data, target_sum=1e4)
    sc.pp.log1p(st_data)
    st_data.var_names = pd.Index(st_data.var_names.str.lower())
    print(f'ST (IBC+DCIS): {st_data.n_obs} cells × {st_data.n_vars} genes (log-CPM)')

    # ── Build Pearson matrix once ─────────────────────────────────────────────
    pearson, sc_genes = build_pearson()

    # ── Per-rep loop ──────────────────────────────────────────────────────────
    for rep in range(N_REPS):
        print(f'\n=== Rep {rep} ===')

        pseudo = build_pseudo_outcome(rep, pearson, sc_genes, st_data)
        pseudo.obs.index = pseudo.obs.index.astype(str)

        run_confounding_and_spatialPCA(rep, pseudo, pearson, sc_genes)

        U_df = pd.read_feather(f'./results_raw/hbc_spatialPCA_pcs{rep}.feather').set_index('spot')
        U_df.index = U_df.index.astype(str)

        common = pseudo.obs_names.intersection(U_df.index)
        if len(common) < len(pseudo.obs_names):
            print(f'  Warning: {len(pseudo.obs_names) - len(common)} cells missing from U_df; restricting.')
        pseudo = pseudo[common].copy()

        model = tidest(
            n_pcs=N_PCS_USE,
            n_folds=N_FOLDS,
            n_estimators=N_ESTIMATORS,
            corr_threshold=CORR_THRESHOLD,
        )
        model.fit(
            pseudo=pseudo,
            U=U_df,
            treatment='Cluster',
            treatment_val='Invasive_Tumor',
            genes=MARKER_GENES,
        )

        out_path = f'./results_raw/hbc_plm_results{rep}.csv'
        model.results_.to_csv(out_path, index=False)
        print(f'Saved {len(model.results_)} results to {out_path}')
        print(f'Significant (q<0.05): {(model.results_["qval"] < 0.05).sum()} / {len(model.results_)}')
        print(model.results_.head(10).to_string(index=False))
