"""
GBM multi-sample PLM using tidest (consolidated entry point).

Inputs:
  results_raw/gbm_tangram_pred_{sample}_{rep}.pkl   (from gbm_tangram.py)
  Inputs/scRNA_Darmanis/darmanis_adata.h5ad          (SC reference)
  Inputs/general/visium_metadata.csv                 (Ivy GAP annotations)
  Inputs/general/GBM_data/{sample}/outs/             (Visium ST data)

Runs internally per sample:
  1. Pearson correction pseudo-outcome (build_pseudo_outcome)
  2. Confounder gene selection (cv_threshold=0.5, marker_r_thresh=0.3, 2000 genes)
  3. SpatialPCA (50 PCs, via bundled spatialPCA_runner.R)
  4. Robinson PLM (n_folds=5, n_estimators=200, corr_threshold=0.5)

Output:
  results_raw/gbm_plm_results_{sample}_{rep}.csv
  results_raw/gbm_plm_results_all.csv
"""

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from tidest import tidest

# ── Constants ──────────────────────────────────────────────────────────────────
N_REPS             = 1
N_PCS_USE          = 50
N_FOLDS            = 5
N_ESTIMATORS       = 200
CORR_THRESHOLD     = 0.5
CV_THRESHOLD       = 0.5   # match old gbm_confounding_estimation.py
MARKER_R_THRESH    = 0.3
N_CONFOUNDER_GENES = 2000
TOP_K              = 5

SC_PATH   = './Inputs/scRNA_Darmanis/darmanis_adata.h5ad'
META_PATH = './Inputs/general/visium_metadata.csv'
IVYGAP_KEEP = ['CT', 'LE']

SAMPLES = [
    'MGH258',
    'UKF243', 'UKF248', 'UKF251', 'UKF255', 'UKF259', 'UKF260',
    'UKF266', 'UKF269', 'UKF275', 'UKF296', 'UKF304', 'UKF313', 'UKF334',
    'ZH1007inf', 'ZH1007nec', 'ZH1019inf', 'ZH1019T1',
    'ZH8811Abulk', 'ZH8811Bbulk', 'ZH8812bulk',
    'ZH881inf', 'ZH881T1',
    'ZH916bulk', 'ZH916inf', 'ZH916T1',
]

MARKER_GENES = [
    # === ORIGINAL PANEL (56 genes) ===
    # Proliferation / CT-enriched
    'TOP2A', 'MKI67', 'PCNA', 'MCM2', 'MCM6',
    # MES state
    'CD44', 'TNC', 'CHI3L1', 'FN1', 'VIM', 'TGFBI', 'COL1A1', 'SERPINE1',
    # AC state
    'APOE', 'SPARCL1', 'ALDOC', 'SLC1A2', 'GLUL', 'GJA1',
    # OPC state
    'OLIG1', 'OLIG2', 'PDGFRA', 'SOX10',
    # NPC / neural progenitor
    'SOX11', 'ASCL1', 'DCX', 'MAP2', 'TUBB3',
    # Invasion / LE-enriched
    'MMP9', 'MMP2', 'PLAUR', 'ITGB1', 'CXCR4',
    # Normal brain (LE-enriched)
    'MBP', 'MOG', 'MAG', 'PLP1',
    'SYT1', 'SNAP25', 'RBFOX3',
    'GFAP', 'AQP4', 'S100B',
    # Immune / microenvironment
    'AIF1', 'CD68', 'TMEM119', 'P2RY12',
    'PECAM1', 'VWF', 'CLDN5',
    # Hypoxia
    'HIF1A', 'VEGFA', 'LDHA', 'CA9',
    # RTKs
    'EGFR', 'MET',
    # === EXTENDED PANEL (34 new genes) ===
    # Homeostatic microglia (LE-enriched)
    'HEXB', 'SALL1', 'OLFML3', 'CX3CR1',
    # TAM / M2 macrophage (CT-enriched)
    'SPP1', 'LGALS1', 'LGALS3', 'CD163', 'MSR1', 'MRC1',
    # Invasion ECM (LE-enriched)
    'PTPRZ1', 'NRCAM', 'BCAN', 'L1CAM', 'HTRA1',
    # MES / CT ECM
    'VCAN', 'ANXA2', 'POSTN',
    # NPC / stem state (LE-enriched)
    'SOX4', 'EGR1', 'CCND2', 'NES',
    # Metabolic disambiguation
    'LDHB', 'HK2', 'EPAS1',
    # Signaling (contested)
    'STAT3', 'IL6', 'PTPN11', 'CDKN2A', 'PTEN', 'IDH1',
    # Normal brain circuit (LE-enriched)
    'NRXN1', 'DLG4', 'CKB',
]


if __name__ == '__main__':

    meta = pd.read_csv(META_PATH)

    # ── Pearson matrix: computed once on the shared SC reference ──────────────
    print('Loading Darmanis scRNA-seq (GSE84465)...')
    adata_sc = sc.read_h5ad(SC_PATH)
    adata_sc.var_names = pd.Index(adata_sc.var_names.str.lower())
    print(f'SC shape: {adata_sc.shape}')

    from tidest._pseudo_outcome import compute_pearson
    print('Computing Pearson matrix (once)...')
    pearson, pearson_genes = compute_pearson(adata_sc)
    print(f'Pearson: {pearson.shape[0]} genes')

    all_results = []

    for sample in SAMPLES:
        print(f'\n{"="*60}\nSample: {sample}\n{"="*60}')

        meta_sample = meta[meta['sample'] == sample].set_index('spot_id')
        ct_le_spots = meta_sample[meta_sample['ivygap'].isin(IVYGAP_KEEP)].index

        st_dir = f'./Inputs/general/GBM_data/{sample}/outs/'
        adata_st = sc.read_visium(st_dir)
        adata_st.var_names_make_unique()
        adata_st.var_names = pd.Index(adata_st.var_names.str.lower())
        adata_st = adata_st[adata_st.obs_names.isin(ct_le_spots)].copy()
        print(f'ST spots (CT+LE): {adata_st.n_obs}')

        for rep in tqdm(range(N_REPS), desc=sample):
            with open(f'./results_raw/gbm_tangram_pred_{sample}_{rep}.pkl', 'rb') as f:
                ad_ge = pickle.load(f)

            ad_ge.var_names = pd.Index(ad_ge.var_names.str.lower())
            ad_ge = ad_ge[ad_ge.obs_names.isin(ct_le_spots)].copy()

            # Align spot order between pred and ST
            common_spots = ad_ge.obs_names.intersection(adata_st.obs_names).tolist()
            ad_ge   = ad_ge[common_spots].copy()
            adata_st_aligned = adata_st[common_spots].copy()

            # Treatment annotation must be in pred_adata.obs for pseudo to carry it
            ad_ge.obs['ivygap'] = meta_sample.loc[common_spots, 'ivygap'].values

            model = tidest(
                n_pcs=N_PCS_USE,
                n_folds=N_FOLDS,
                n_estimators=N_ESTIMATORS,
                corr_threshold=CORR_THRESHOLD,
                cv_threshold=CV_THRESHOLD,
                marker_r_thresh=MARKER_R_THRESH,
                n_confounder_genes=N_CONFOUNDER_GENES,
                top_k=TOP_K,
                tmp_dir=f'./tidest_tmp/gbm_{sample}_{rep}',
            )
            model.fit(
                pred_adata=ad_ge,
                st_adata=adata_st_aligned,
                pearson=pearson,
                pearson_genes=pearson_genes,
                treatment='ivygap',
                treatment_val='LE',
                genes=MARKER_GENES,
            )

            results = model.results_.copy()
            results.insert(0, 'sample', sample)

            out_path = f'./results_raw/gbm_plm_results_{sample}_{rep}.csv'
            results.to_csv(out_path, index=False)
            print(f'Saved {len(results)} results → {out_path}')
            print(f'Significant (q<0.05): {(results["qval"] < 0.05).sum()} / {len(results)}')

            all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv('./results_raw/gbm_plm_results_all.csv', index=False)
    print(f'\nSaved combined ({len(combined)} rows) → ./results_raw/gbm_plm_results_all.csv')
