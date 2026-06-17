#
# Xenium data
#

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

import importlib
import pickle
import hdf5plugin
import anndata as ad

from tqdm import tqdm

import spatialdata as sd
import DDSPI
from scipy.sparse import csr_matrix
from anndata import AnnData

import torch

from CellPLM.utils import set_seed
from CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig

# Based on 
# https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_xenium.html
# https://giottosuite.com/articles/xenium_breast_cancer.html
# https://github.com/OmicsML/CellPLM/blob/main/tutorials/spatial_imputation.ipynb

if __name__ == "__main__":

    torch.device('cuda')
  
    xenium_path = "./Xenium/outs"
    annotations_path = "./Xenium"
    zarr_path = "./Xenium.zarr"

    # sdata = xenium(xenium_path)
    # sdata.write(zarr_path)

    sc_data = sc.read_10x_h5(pathlib.Path(annotations_path) / "Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5")

    sdata = sd.read_zarr(zarr_path)

    st_data = sdata.tables["table"]
    # sum([x in list(sc_data.var_names) for x in list(st_data.var_names)]) # all in sc data, good!
    
    # open excel
    annotations = pd.read_excel(
        pathlib.Path(annotations_path) / "Cell_Barcode_Type_Matrices.xlsx",
        sheet_name="Xenium R1 Fig1-5 (supervised)",
    )

    st_data.obs = st_data.obs.merge(annotations.set_index('Barcode').loc[[int(x) + 1 for x in st_data.obs_names], 'Cluster'], left_on='cell_id', right_index=True, how='left')

    annotations = pd.read_excel(
        pathlib.Path(annotations_path) / "Cell_Barcode_Type_Matrices.xlsx",
        sheet_name="scFFPE-Seq",
    )

    # add annotation to adata
    sc_data = sc_data[sc_data.obs_names.isin(annotations['Barcode'])]
    sc_data.obs['annotation'] = annotations.set_index('Barcode').loc[list(sc_data.obs_names), 'Annotation']
    
    print('Number of ST spots:', st_data.n_obs)
    print('Number of ST genes:', st_data.n_vars)
    print('Number of SC cells:', sc_data.n_obs)
    print('Number of SC genes:', sc_data.n_vars)

    # Preprocessing as in https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
    # Quality control, normalization, feature selection (highly-variable genes)

    sc.pp.calculate_qc_metrics(sc_data, inplace=True, log1p=True)
    sc.pp.filter_cells(sc_data, min_genes=100)
    sc.pp.filter_genes(sc_data, min_cells=3)
    sc.pp.normalize_total(sc_data, target_sum=1e4)
    adata_sc_2 = sc_data.copy()
    sc.pp.log1p(adata_sc_2)
    sc.pp.highly_variable_genes(adata_sc_2, n_top_genes=5000)
    sc_data = sc_data[:, adata_sc_2.var['highly_variable']]
    adata_sc_2 = adata_sc_2[:, adata_sc_2.var['highly_variable']]

    # sc.pp.calculate_qc_metrics(st_data, inplace=True, log1p=True)
    sc.pp.filter_cells(st_data, min_genes=75)
    sc.pp.filter_genes(st_data, min_cells=3)
    sc.pp.normalize_total(st_data, target_sum=1e4)
    adata_st_2 = st_data.copy()
    sc.pp.log1p(adata_st_2)
    # sc.pp.highly_variable_genes(adata_st_2)
    # st_data = st_data[:, adata_st_2.var['highly_variable']]

    st_data = st_data[:, st_data.var_names.isin(sc_data.var_names)]
    adata_st_2 = adata_st_2[:, adata_st_2.var_names.isin(sc_data.var_names)]

    print('After filtering:')
    print('Number of ST spots:', st_data.n_obs)
    print('Number of ST genes:', st_data.n_vars)
    print('Number of SC cells:', sc_data.n_obs)
    print('Number of SC genes:', sc_data.n_vars)

    # temporary
    sc_data = adata_sc_2.copy()
    st_data = adata_st_2.copy()

    # sc.pp.calculate_qc_metrics(st_data, percent_top=(10, 20, 50, 150), inplace=True)

    # cprobes = (
    #     st_data.obs["control_probe_counts"].sum() / st_data.obs["total_counts"].sum() * 100
    # )
    # cwords = (
    #     st_data.obs["control_codeword_counts"].sum() / st_data.obs["total_counts"].sum() * 100
    # )
    # print(f"Negative DNA probe count % : {cprobes}")
    # print(f"Negative decoding count % : {cwords}")


    # fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    # axs[0].set_title("Total transcripts per cell")
    # sns.histplot(
    #     st_data.obs["total_counts"],
    #     kde=False,
    #     ax=axs[0],
    # )

    # axs[1].set_title("Unique transcripts per cell")
    # sns.histplot(
    #     st_data.obs["n_genes_by_counts"],
    #     kde=False,
    #     ax=axs[1],
    # )


    # axs[2].set_title("Area of segmented cells")
    # sns.histplot(
    #     st_data.obs["cell_area"],
    #     kde=False,
    #     ax=axs[2],
    # )

    # axs[3].set_title("Nucleus ratio")
    # sns.histplot(
    #     st_data.obs["nucleus_area"] / st_data.obs["cell_area"],
    #     kde=False,
    #     ax=axs[3],
    # )

    # plt.tight_layout()
    # plt.show()

    # sc.pp.filter_cells(st_data, min_counts=10)
    # sc.pp.filter_genes(st_data, min_cells=5)

    # st_data.layers["counts"] = st_data.X.copy()
    # sc.pp.normalize_total(st_data, inplace=True)
    # sc.pp.log1p(st_data)
    # sc.pp.pca(st_data)
    # sc.pp.neighbors(st_data)
    # sc.tl.umap(st_data)
    # sc.tl.leiden(st_data, key_added="leiden")

    # sc.pl.umap(
    #     st_data,
    #     color=[
    #         "total_counts",
    #         "n_genes_by_counts",
    #         "leiden",
    #     ],
    #     wspace=0.4,
    # )

    # sq.pl.spatial_scatter(
    #     st_data,
    #     library_id="spatial",
    #     shape=None,
    #     color=[
    #         "leiden",
    #     ],
    #     wspace=0.4,
    # )

    # plt.show()

    pipeline_config = ImputationDefaultPipelineConfig.copy()
    model_config = ImputationDefaultModelConfig.copy()

    PRETRAIN_VERSION = '20231027_85M'
    DEVICE = 'cuda' # 'cpu' or 'mps'

    pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                            overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                            pretrain_directory=pathlib.Path(annotations_path) / './ckpt')

    # create new anndata 
    new_st_data = AnnData(
        X = np.zeros((st_data.n_obs, sc_data.n_vars), dtype='float32'),
        obs = st_data.obs.copy(),
        var = sc_data.var.copy(),
    )

    new_st_data[:, new_st_data.var_names.isin(st_data.var_names)].X = st_data[:, st_data.var_names.isin(new_st_data.var_names)].X

    # Data from your JSON and experiment description
    pixel_size = 0.2125 
    fov_size_microns = 500 
    coords = st_data.obsm['spatial']
    x_microns = coords[:, 0]
    y_microns = coords[:, 1]

    x_tile = (x_microns // fov_size_microns).astype(int)
    y_tile = (y_microns // fov_size_microns).astype(int)
    new_st_data.obs['fov_id'] = x_tile.astype(str) + "_" + y_tile.astype(str)

    # Formula: (Global position - Tile Start) / Pixel Size
    new_st_data.obs['x_FOV_px'] = ((x_microns % fov_size_microns) / pixel_size).astype('float32')
    new_st_data.obs['y_FOV_px'] = ((y_microns % fov_size_microns) / pixel_size).astype('float32')

    new_st_data.obs['batch'] = new_st_data.obs['fov_id']
    new_st_data.obs['platform'] = '10x_xenium'

    # Ensure they are strings to avoid the 'int' vs 'str' error we saw earlier
    sc_data.obs['batch'] = "Flex_Reference_Batch_1"
    sc_data.obs['platform'] = '10x_chromium'

    new_st_data.var['gene_symbol'] = new_st_data.var_names
    sc_data.var['gene_symbol'] = sc_data.var_names
    st_data.var['gene_symbol'] = st_data.var_names
    new_st_data.var_names = new_st_data.var['gene_ids']
    sc_data.var_names = sc_data.var['gene_ids']
    st_data.var_names = st_data.var['gene_ids']

    new_st_data.obsm['truth'] = st_data.X.toarray()

    query_genes = [g for g in sc_data.var.index]
    ref_genes = [g for g in sc_data.var.index if g in st_data.var.index]
    query_batches = [str(x) for x in list(new_st_data.obs['batch'].unique())]
    ref_batches = [str(x) for x in list(sc_data.obs['batch'].unique())]
    batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
        [query_genes]*len(query_batches) + [sc_data.var.index.tolist()]*len(ref_batches)))

    new_st_data = new_st_data[:, new_st_data.var_names.isin(pipeline.model.gene_set)].copy()
    sc_data = sc_data[:, sc_data.var_names.isin(pipeline.model.gene_set)].copy()

    train_data = new_st_data.concatenate(sc_data, join='outer', batch_key=None, index_unique=None)

    train_data.obs['split'] = 'train'
    train_data.obs['split'][train_data.obs['batch']==new_st_data.obs['batch'][-1]] = 'valid'
    train_data.obs['split'][train_data.obs['batch']==sc_data.obs['batch'][-1]] = 'valid'

    with open('genes_to_test_cellplm_all.pkl', 'wb') as f:
        pickle.dump(new_st_data.var_names.tolist(), f)

    with open('observed_st_genes_cellplm_all.pkl', 'wb') as f:
        pickle.dump(st_data.var_names.tolist(), f)

    n_reps = 50

    for rep in tqdm(range(n_reps)):

        pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                    overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                    pretrain_directory=pathlib.Path(annotations_path) / './ckpt')

        set_seed(rep)
        pipeline.fit(train_data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', #  Specify a column in .obs that contains split information
                train_split = 'train',
                valid_split = 'valid',
                batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
                device = DEVICE, # Specify the device to run the model on
                ) 

        pred = pipeline.predict(
            new_st_data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            device = DEVICE,
        )

        # pipeline.score(
        #             new_st_data, # An AnnData object
        #             evaluation_config = {'target_genes': ref_genes}, # The config dictionary we created previously, optional
        #             label_fields = ['truth'], # A field in .obsm that stores the ground-truth for evaluation
        #             device = DEVICE,
        # )  

        ad_ge = new_st_data.copy()
        ad_ge.X = np.array(pred.cpu())

        # obs_total = st_data.X.toarray().sum()
        # pred_total = ad_ge[:, ad_ge.var_names.isin(st_data.var_names)].X.sum()
        # adjustment_factor = obs_total / pred_total
        # ad_ge.X = ad_ge.X * adjustment_factor

        # # sc.pp.normalize_total(ad_ge, target_sum=1e4)

        # adata_sc2 = sc_data.copy()
        # adata_sc2.X = csr_matrix(adata_sc2.X)
        # adata_st2 = st_data[st_data.obs.Cluster.isin(['Invasive_Tumor', 'Prolif_Invasive_Tumor', 'DCIS_1', 'DCIS_2'])].copy()
        # adata_st2 = adata_st2[:, adata_sc2.var_names.intersection(adata_st2.var_names)].copy()
        # ad_ge2 = ad_ge[ad_ge.obs.Cluster.isin(['Invasive_Tumor', 'Prolif_Invasive_Tumor', 'DCIS_1', 'DCIS_2'])].copy()
        # ad_ge2 = ad_ge2[:, adata_sc2.var_names.intersection(ad_ge2.var_names)].copy()

        # adata_st2.obs.loc[adata_st2.obs['Cluster'] == 'DCIS_2', 'Cluster'] = 'DCIS_1'
        # ad_ge2.obs.loc[ad_ge2.obs['Cluster'] == 'DCIS_2', 'Cluster'] = 'DCIS_1'
        # adata_st2.obs.loc[adata_st2.obs['Cluster'] == 'Prolif_Invasive_Tumor', 'Cluster'] = 'Invasive_Tumor'
        # ad_ge2.obs.loc[ad_ge2.obs['Cluster'] == 'Prolif_Invasive_Tumor', 'Cluster'] = 'Invasive_Tumor'

        # importlib.reload(DDSPI)

        # ddspi = DDSPI.DDSPI(
        #     ST_data=adata_st2,
        #     ST_annotations='Cluster',
        #     ST_coordinates='spatial',
        #     SC_data=adata_sc2,
        #     ST_predictions=ad_ge2
        # )

        # ddspi.fit(verbose=True, augmentation_type='complete', n_permutations=9999, alpha=0.1)
        # significant_genes_complete = ddspi.get_significant_genes()

    # query_dataset = './Xenium/HumanLungCancerPatient2_filtered_ensg.h5ad'
    # ref_dataset = './Xenium/GSE131907_Lung_ensg.h5ad'
    # query_data = ad.read_h5ad(query_dataset)
    # ref_data = ad.read_h5ad(ref_dataset)

    # target_genes = stratified_sample_genes_by_sparsity(query_data, seed=11) # This is for reproducing the hold-out gene lists in our paper
    # query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
    # query_data[:, target_genes].X = 0
    # train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

    # train_data.obs['split'] = 'train'
    # train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid'
    # train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid'

    # query_genes = [g for g in query_data.var.index if g not in target_genes]
    # query_batches = list(query_data.obs['batch'].unique())
    # ref_batches = list(ref_data.obs['batch'].unique())
    # batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
    #     [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))
    
    # pipeline_config = ImputationDefaultPipelineConfig.copy()
    # model_config = ImputationDefaultModelConfig.copy()

    # PRETRAIN_VERSION = '20231027_85M'
    # DEVICE = 'cuda' # 'cpu' or 'mps'

    # pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
    #                                     overwrite_config=model_config,  # This is for overwriting part of the pretrain config
    #                                     pretrain_directory=pathlib.Path(annotations_path) / './ckpt')
    
    # pipeline.fit(train_data, # An AnnData object
    #         pipeline_config, # The config dictionary we created previously, optional
    #         split_field = 'split', #  Specify a column in .obs that contains split information
    #         train_split = 'train',
    #         valid_split = 'valid',
    #         batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
    #         device = DEVICE,
    #         ) 

        # save storer
        with open('./results/cellplm3_ddspi_pred' + str(rep) + '_all.pkl', 'wb') as f:
            pickle.dump(ad_ge, f)