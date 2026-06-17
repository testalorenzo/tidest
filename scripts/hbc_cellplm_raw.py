#
# HBC: CellPLM imputation (step 1 of causal pipeline)
#
# Normalization philosophy — same as MB/GBM Tangram:
#   SC: normalize_total(1e4) only (CPM), no log1p
#   ST: raw counts, no normalization
# CellPLM predictions are therefore in CPM scale; per-spot scaling
# in hbc_pseudo_outcome.py converts them to raw-count scale.
#

import scanpy as sc
import numpy as np
import pandas as pd
import pathlib
import pickle
import hdf5plugin
import spatialdata as sd
import torch

# Patch torch.load BEFORE importing CellPLM so checkpoint loading works even
# when the conda env has CPU-only PyTorch (torch.cuda.is_available() == False).
_orig_torch_load = torch.load
def _safe_torch_load(f, map_location=None, **kw):
    if map_location is None and not torch.cuda.is_available():
        map_location = torch.device('cpu')
    return _orig_torch_load(f, map_location=map_location, **kw)
torch.load = _safe_torch_load

from anndata import AnnData
from tqdm import tqdm

from CellPLM.utils import set_seed
from CellPLM.pipeline.imputation import (
    ImputationPipeline,
    ImputationDefaultPipelineConfig,
    ImputationDefaultModelConfig,
)

N_REPS           = 1
PRETRAIN_VERSION = '20231027_85M'

MARKER_GENES = [
    'MKI67','TOP2A','PCNA','CDK1','CCNB1','MCM2','MCM6',
    'VIM','FN1','CDH2','TWIST1','SNAI1','SNAI2','ZEB1','ZEB2',
    'MMP2','MMP9','MMP11','CXCR4',
    'EGFR','ERBB2','S100A8','S100A9','S100A4',
    'KRT5','KRT14','CD44',
    'ESR1','PGR','FOXA1','GATA3','TFF1','TFF3','AREG',
    'KRT8','KRT18','KRT19',
    'CDH1','CLDN3','CLDN4','CLDN7',
    'SPDEF','MLPH','AGR2',
    'CNN1','TP63','MYLK','ITGB6',
    'LAMB3','LAMC2','COL4A1',
    'RUNX1','ERBB4','KIT','EZH2',
    'COL11A1','POSTN','CXCL10','THBS2','GREM1',
    'TIGIT','CTLA4','NKG7','GNLY','TYROBP',
    'CDH3','CA9','CDKN2A','ALDH1A1',
]
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
ANNOTATIONS_PATH = './Xenium'
ZARR_PATH        = './Xenium.zarr'
SC_H5_NAME       = ('Chromium_FFPE_Human_Breast_Cancer_'
                    'Chromium_FFPE_Human_Breast_Cancer_'
                    'count_sample_filtered_feature_bc_matrix.h5')

if __name__ == '__main__':

    print(f'CUDA available: {torch.cuda.is_available()}, running on: {DEVICE}')
    pipeline_config = ImputationDefaultPipelineConfig.copy()
    model_config    = ImputationDefaultModelConfig.copy()

    # ------------------------------------------------------------------ #
    # SC data — CPM normalized, no log1p
    # ------------------------------------------------------------------ #
    sc_data = sc.read_10x_h5(pathlib.Path(ANNOTATIONS_PATH) / SC_H5_NAME)

    annotations_sc = pd.read_excel(
        pathlib.Path(ANNOTATIONS_PATH) / 'Cell_Barcode_Type_Matrices.xlsx',
        sheet_name='scFFPE-Seq',
    )
    sc_data = sc_data[sc_data.obs_names.isin(annotations_sc['Barcode'])].copy()
    sc_data.obs['annotation'] = (
        annotations_sc.set_index('Barcode').loc[list(sc_data.obs_names), 'Annotation']
    )

    # SC: log-CPM (normalize_total(1e4) + log1p).
    # CellPLM pretrained model (HLCA) was trained on log-CPM; common_preprocess does no
    # normalization internally, so normalization is the caller's responsibility.
    sc.pp.normalize_total(sc_data, target_sum=1e4)
    sc.pp.log1p(sc_data)
    sc.pp.highly_variable_genes(sc_data, n_top_genes=10000)
    # Force-include all panel marker genes regardless of HVG rank
    marker_lower = {g.lower() for g in MARKER_GENES}
    sc_data.var.loc[
        [v for v in sc_data.var_names if v.lower() in marker_lower],
        'highly_variable'
    ] = True
    sc_data = sc_data[:, sc_data.var['highly_variable']].copy()

    print(f'SC: {sc_data.n_obs} cells × {sc_data.n_vars} HVG genes (log-CPM, marker genes force-included)')

    # ------------------------------------------------------------------ #
    # ST data — log-CPM (same space as SC and CellPLM output).
    # CellPLM outputs log-CPM; ST observed genes must also be log-CPM so the
    # Pearson correction in hbc_pseudo_outcome.py operates in a consistent space.
    # ------------------------------------------------------------------ #
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

    sc.pp.normalize_total(st_data, target_sum=1e4)
    sc.pp.log1p(st_data)

    # Restrict ST to SC genes (by gene symbol, before ENSG renaming)
    st_data = st_data[:, st_data.var_names.isin(sc_data.var_names)].copy()
    print(f'ST: {st_data.n_obs} cells × {st_data.n_vars} genes (log-CPM)')

    # ------------------------------------------------------------------ #
    # Rename var_names to ENSG IDs — required by CellPLM model gene_set
    # ------------------------------------------------------------------ #
    sc_data.var['gene_symbol'] = sc_data.var_names
    st_data.var['gene_symbol'] = st_data.var_names
    sc_data.var_names = sc_data.var['gene_ids']
    st_data.var_names = st_data.var['gene_ids']

    # ------------------------------------------------------------------ #
    # Build new_st_data: all ST cells × SC gene space
    # Observed genes filled with log-CPM ST values; unobserved stay zero.
    # ------------------------------------------------------------------ #
    new_st_data = AnnData(
        X   = np.zeros((st_data.n_obs, sc_data.n_vars), dtype='float32'),
        obs = st_data.obs.copy(),
        var = sc_data.var.copy(),
    )
    obs_mask = new_st_data.var_names.isin(st_data.var_names)
    new_st_data.X[:, obs_mask] = (
        st_data[:, st_data.var_names.isin(new_st_data.var_names)].X.toarray()
        if hasattr(st_data.X, 'toarray') else
        np.array(st_data[:, st_data.var_names.isin(new_st_data.var_names)].X)
    )

    new_st_data.obsm['truth'] = (
        st_data.X.toarray() if hasattr(st_data.X, 'toarray') else np.array(st_data.X)
    )

    # FOV tiling from spatial coordinates (same as xenium.py)
    pixel_size  = 0.2125
    fov_size_um = 500.0
    coords      = st_data.obsm['spatial']
    x_um, y_um  = coords[:, 0], coords[:, 1]
    x_tile      = (x_um // fov_size_um).astype(int)
    y_tile      = (y_um // fov_size_um).astype(int)
    new_st_data.obs['fov_id']   = x_tile.astype(str) + '_' + y_tile.astype(str)
    new_st_data.obs['x_FOV_px'] = ((x_um % fov_size_um) / pixel_size).astype('float32')
    new_st_data.obs['y_FOV_px'] = ((y_um % fov_size_um) / pixel_size).astype('float32')
    new_st_data.obs['batch']    = new_st_data.obs['fov_id']
    new_st_data.obs['platform'] = '10x_xenium'

    sc_data.obs['batch']    = 'Flex_Reference_Batch_1'
    sc_data.obs['platform'] = '10x_chromium'

    # ------------------------------------------------------------------ #
    # Filter to CellPLM model gene_set (load pipeline once to get gene_set)
    # ------------------------------------------------------------------ #
    _probe = ImputationPipeline(
        pretrain_prefix=PRETRAIN_VERSION,
        overwrite_config=model_config,
        pretrain_directory=pathlib.Path(ANNOTATIONS_PATH) / 'ckpt',
    )
    model_genes = set(_probe.model.gene_set)

    new_st_data = new_st_data[:, new_st_data.var_names.isin(model_genes)].copy()
    sc_data     = sc_data[:, sc_data.var_names.isin(model_genes)].copy()
    print(f'After model gene_set filter: ST {new_st_data.n_vars} genes, SC {sc_data.n_vars} genes')

    query_genes   = list(new_st_data.var_names)
    query_batches = [str(b) for b in new_st_data.obs['batch'].unique()]
    ref_batches   = [str(b) for b in sc_data.obs['batch'].unique()]
    batch_gene_list = dict(zip(
        query_batches + ref_batches,
        [query_genes] * len(query_batches) + [sc_data.var_names.tolist()] * len(ref_batches),
    ))

    train_data = new_st_data.concatenate(sc_data, join='outer', batch_key=None, index_unique=None)
    train_data.obs['split'] = 'train'
    last_st_batch = new_st_data.obs['batch'].iloc[-1]
    last_sc_batch = sc_data.obs['batch'].iloc[-1]
    train_data.obs.loc[train_data.obs['batch'] == last_st_batch, 'split'] = 'valid'
    train_data.obs.loc[train_data.obs['batch'] == last_sc_batch, 'split'] = 'valid'

    print(f'Train data: {train_data.n_obs} cells × {train_data.n_vars} genes')

    # ------------------------------------------------------------------ #
    # CellPLM loop
    # ------------------------------------------------------------------ #
    for rep in tqdm(range(N_REPS)):

        pipeline = ImputationPipeline(
            pretrain_prefix=PRETRAIN_VERSION,
            overwrite_config=model_config,
            pretrain_directory=pathlib.Path(ANNOTATIONS_PATH) / 'ckpt',
        )
        set_seed(rep)
        pipeline.fit(
            train_data, pipeline_config,
            split_field='split', train_split='train', valid_split='valid',
            batch_gene_list=batch_gene_list, device=DEVICE,
        )
        pred = pipeline.predict(new_st_data, pipeline_config, device=DEVICE)

        ad_ge   = new_st_data.copy()
        ad_ge.X = np.array(pred.cpu())

        # Move any remaining CUDA tensors to numpy before pickling
        for k in list(ad_ge.obsm.keys()):
            v = ad_ge.obsm[k]
            if torch.is_tensor(v):
                ad_ge.obsm[k] = v.cpu().numpy()
        for k in list(ad_ge.varm.keys()):
            v = ad_ge.varm[k]
            if torch.is_tensor(v):
                ad_ge.varm[k] = v.cpu().numpy()

        out_path = f'./results_raw/hbc_cellplm_pred{rep}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(ad_ge, f)
        print(f'Rep {rep}: saved {out_path}, shape={ad_ge.shape}')
