#
# Mouse Brain data analysis with Tangram
#

# https://tangram-sc.readthedocs.io/en/latest/tutorial_link.html

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import tangram as tg

import pickle

from tqdm import tqdm

if __name__ == "__main__":

    print('Tangram analysis of Mouse Brain data')

    n_reps = 1  # number of repetitions for averaging results

    # Load data
    adata_st = sq.datasets.visium_fluo_adata_crop()
    adata_st = adata_st[
        adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
    ].copy()
    img = sq.datasets.visium_fluo_image_crop()

    adata_sc = sq.datasets.sc_mouse_cortex()

    print('Number of ST spots:', adata_st.n_obs)
    print('Number of ST genes:', adata_st.n_vars)
    print('Number of SC cells:', adata_sc.n_obs)
    print('Number of SC genes:', adata_sc.n_vars)

    sc.pp.normalize_total(adata_sc, target_sum=1e4)

    sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
    markers = list(np.unique(markers_df.melt().value.values))

    tg.pp_adatas(adata_sc, adata_st, genes=markers)

    for rep in tqdm(range(n_reps)):

        ad_map = tg.map_cells_to_space(adata_sc, adata_st,
            mode="cells", # mode="clusters",
            density_prior='rna_count_based',
            random_state=rep
        )

        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)

        with open('./results_raw/tangram_pred' + str(rep) + '.pkl', 'wb') as f:
            pickle.dump(ad_ge, f)        
