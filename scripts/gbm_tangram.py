import scanpy as sc
import numpy as np
import pandas as pd
import tangram as tg
import pickle
from tqdm import tqdm

N_REPS                = 1
N_MARKERS_PER_CLUSTER = 100
SC_PATH   = './Inputs/scRNA_Darmanis/darmanis_adata.h5ad'
META_PATH = './Inputs/general/visium_metadata.csv'

SAMPLES = [
    'MGH258',
    'UKF243', 'UKF248', 'UKF251', 'UKF255', 'UKF259', 'UKF260',
    'UKF266', 'UKF269', 'UKF275', 'UKF296', 'UKF304', 'UKF313', 'UKF334',
    'ZH1007inf', 'ZH1007nec', 'ZH1019inf', 'ZH1019T1',
    'ZH8811Abulk', 'ZH8811Bbulk', 'ZH8812bulk',
    'ZH881inf', 'ZH881T1',
    'ZH916bulk', 'ZH916inf', 'ZH916T1',
]

if __name__ == '__main__':

    meta = pd.read_csv(META_PATH)

    # SC preprocessing is shared across all samples
    print('Loading Darmanis scRNA-seq (GSE84465)...')
    adata_sc = sc.read_h5ad(SC_PATH)
    adata_sc.var_names = pd.Index(adata_sc.var_names.str.lower())
    print('SC shape:', adata_sc.shape)

    sc.pp.normalize_total(adata_sc, target_sum=1e4)

    # Use ground-truth Darmanis cell_type labels (not unsupervised Leiden clusters)
    # so that neuron-specific markers survive into the training gene set.
    adata_sc_for_markers = adata_sc.copy()
    adata_sc_for_markers.obs['cluster'] = adata_sc_for_markers.obs['cell_type'].values
    print(f'SC cell types: {adata_sc_for_markers.obs["cluster"].value_counts().to_dict()}')
    sc.tl.rank_genes_groups(adata_sc_for_markers, groupby='cluster', use_raw=False)
    markers_df = pd.DataFrame(
        adata_sc_for_markers.uns['rank_genes_groups']['names']
    ).iloc[:N_MARKERS_PER_CLUSTER]
    training_genes_all = list(np.unique(markers_df.values.ravel()))

    adata_sc_pristine = adata_sc.copy()

    for sample in SAMPLES:
        print(f'\n{"="*60}\nSample: {sample}\n{"="*60}')

        st_dir = f'./Inputs/general/GBM_data/{sample}/outs/'
        adata_st = sc.read_visium(st_dir)
        adata_st.var_names_make_unique()
        adata_st.var_names = pd.Index(adata_st.var_names.str.lower())

        meta_sample = meta[meta['sample'] == sample].set_index('spot_id')
        shared_spots = adata_st.obs_names.intersection(meta_sample.index)
        adata_st = adata_st[shared_spots].copy()
        print(f'ST spots after metadata filter: {adata_st.n_obs}')

        shared = set(adata_sc.var_names) & set(adata_st.var_names)
        training_genes = [g for g in training_genes_all if g in shared]
        print(f'Training genes: {len(training_genes)}')

        # pp_adatas modifies in-place — work on fresh copies per sample
        adata_sc_copy = adata_sc.copy()
        tg.pp_adatas(adata_sc_copy, adata_st, genes=training_genes)

        for rep in tqdm(range(N_REPS), desc=sample):
            ad_map = tg.map_cells_to_space(
                adata_sc_copy, adata_st,
                mode='cells',
                density_prior='rna_count_based',
                random_state=rep,
                verbose=False,
            )
            ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc_pristine)

            out = f'./results_raw/gbm_tangram_pred_{sample}_{rep}.pkl'
            with open(out, 'wb') as f:
                pickle.dump(ad_ge, f)
            print(f'  Rep {rep}: saved {out}, shape={ad_ge.shape}')
