"""
Mouse brain (MB) cortical layer PLM using tidest.

Inputs (pre-computed by mb_pseudo_outcome.py + mb_confounding_estimation.py):
  results_raw/tangram_pseudo{rep}.pkl     — pseudo-outcome AnnData (log space)
  results_raw/spatialPCA_pcs{rep}.feather — spatial PCs from SpatialPCA

Output:
  results_raw/plm_results{rep}.csv
"""
import pickle
import pandas as pd
from tidest import tidest

N_REPS         = 1
N_PCS_USE      = 10
N_FOLDS        = 2
N_ESTIMATORS   = 200
CORR_THRESHOLD = 0.5

MARKER_GENES = [
    # Superficial (L2/3/4)
    'Cux1','Cux2','Calb1','Rasgrf2','Pou3f2','Pou3f3','Satb2','Prss12','Sema3c','Lhx2',
    'Rorb','Rspo1','Scnn1a','Plcxd2','Krt73',
    # Deep (L5/6)
    'Bcl11b','Fezf2','Foxp2','Sox5','Tbr1','Tle4','Tshz3','Ldb2','Sulf1','Cdh13',
    'Syt6','Nxph4','Nfe2l1','Etv1','Tdo2','Chrna6','Nxph2','Syt17',
    # L5 specifically
    'Grik1','Trhr','Npsr1','Chrna4','Htr2c',
    # Broad excitatory / layer identity
    'Neurod6','Neurod2','Nrn1','Camk2a','Emx1','Rbfox3',
    # Inhibitory interneurons
    'Pvalb','Sst','Vip','Reln','Lhx6',
    # Other known DE across deep/superficial
    'Crym','Ntng1','Slc17a6','Slc17a7','Gad1','Gad2',
]


if __name__ == '__main__':

    for rep in range(N_REPS):
        print(f'\n=== Rep {rep} ===')

        with open(f'./results_raw/tangram_pseudo{rep}.pkl', 'rb') as f:
            pseudo = pickle.load(f)

        U_df = pd.read_feather(f'./results_raw/spatialPCA_pcs{rep}.feather').set_index('spot')

        model = tidest(
            n_pcs=N_PCS_USE,
            n_folds=N_FOLDS,
            n_estimators=N_ESTIMATORS,
            corr_threshold=CORR_THRESHOLD,
        )
        model.fit(
            pseudo=pseudo,
            U=U_df,
            treatment='cluster',
            treatment_val='Cortex_3',
            genes=MARKER_GENES,
        )

        out_path = f'./results_raw/plm_results{rep}.csv'
        model.results_.to_csv(out_path, index=False)
        print(f'Saved {len(model.results_)} results to {out_path}')
        print(f'Significant (q<0.05): {(model.results_["qval"] < 0.05).sum()} / {len(model.results_)}')
        print(model.results_.head(10).to_string(index=False))
