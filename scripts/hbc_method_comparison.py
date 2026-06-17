"""
Competitor comparison for the HBC (IBC vs DCIS) application.

Compares five methods on the 30 marker genes observable in the Xenium panel:
  1. TIDEST       — Robinson PLM  (load from hbc_plm_results0.csv)
  2. t-test       — Welch t-test on log-CPM observed counts
  3. SpaGCN       — Wilcoxon rank-sum on log-CPM observed counts
  4. DESpace      — edgeR NB via svg_test (R subprocess)
  5. SpatialGEE   — GEE Poisson via run_gee_gst (R subprocess)

Note: TIDEST additionally tests 39 genes accessible only via the CellPLM
augmented outcome (TFF1, TFF3, AREG, FN1, VIM, CDH2, etc.). Competitor
methods are evaluated only on the 30 genes present in the observed Xenium
313-gene panel; the TIDEST entries in the per-method summary are also
restricted to those same 30 genes for a fair comparison.

Outputs:
  results_raw/hbc_method_comparison.csv    (per-method summary, method as index)
  results_raw/hbc_pergene_allmethods.csv   (per-gene × method wide table)
"""

import os, sys, subprocess, tempfile
import numpy as np
import pandas as pd
import pathlib
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import scanpy as sc
import spatialdata as sd

warnings.filterwarnings('ignore')

_ROOT    = os.path.dirname(os.path.abspath(__file__ + '/..'))
_SIM_DIR = os.path.join(_ROOT, 'scripts', 'simulation')

# ── Constants ──────────────────────────────────────────────────────────────────
ALPHA        = 0.05
ZARR_PATH    = './Xenium.zarr'
ANNOT_PATH   = './Xenium'
PLM_RESULTS  = './results_raw/hbc_plm_results0.csv'

IBC_CLUSTERS  = ['Invasive_Tumor', 'Prolif_Invasive_Tumor']
DCIS_CLUSTERS = ['DCIS_1', 'DCIS_2']
ALL_CLUSTERS  = IBC_CLUSTERS + DCIS_CLUSTERS

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

# Corrected expected directions (IBC=+1, DCIS=-1, contested=0).
# KRT5/KRT14: DCIS-enriched (intact myoepithelial layer in DCIS).
# EGFR, FOXA1, MLPH, GATA3: contested (ER-status / subtype dependent).
EXPECTED_SIGN = {}
for g in ['MKI67','TOP2A','PCNA','CDK1','CCNB1','MCM2','MCM6',
          'VIM','FN1','CDH2','TWIST1','SNAI1','SNAI2','ZEB1','ZEB2',
          'MMP2','MMP9','MMP11','CXCR4',
          'ERBB2','S100A8','S100A9','S100A4','CD44',
          'EZH2','COL11A1','POSTN','CXCL10','THBS2','GREM1',
          'NKG7','GNLY','TYROBP']:
    EXPECTED_SIGN[g.lower()] = +1
for g in ['ESR1','PGR','TFF1','TFF3','AREG',
          'KRT5','KRT14',
          'KRT8','KRT18','KRT19',
          'CDH1','CLDN3','CLDN4','CLDN7',
          'SPDEF','AGR2',
          'CNN1','TP63','MYLK','ITGB6',
          'LAMB3','LAMC2','COL4A1',
          'RUNX1','ERBB4','KIT',
          'TIGIT']:
    EXPECTED_SIGN[g.lower()] = -1
for g in ['CDH3','CA9','CDKN2A','ALDH1A1','CTLA4',
          'EGFR','FOXA1','MLPH','GATA3']:
    EXPECTED_SIGN[g.lower()] = 0


# ── Shared helpers ─────────────────────────────────────────────────────────────

def bh_correct(pvals):
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    return qvals


def score_results(df):
    """Return summary dict for one method's result df (columns: gene, tau, qval).
    Only genes in EXPECTED_SIGN (any value) are included; contested (0) are
    excluded from n_sig_correct but count towards n_genes and n_sig."""
    df = df.copy()
    df['exp'] = df['gene'].map(EXPECTED_SIGN)
    known = df[df['exp'].notna() & (df['exp'] != 0)]  # directional genes only
    sig   = df[df['qval'] < ALPHA]
    sig_known = known[known['qval'] < ALPHA]
    return {
        'n_genes':       len(df),
        'n_sig':         int((df['qval'] < ALPHA).sum()),
        'n_known':       len(known),
        'n_sig_known':   len(sig_known),
        'n_sig_correct': int((sig_known['tau'] * sig_known['exp'] > 0).sum()),
        'dir_acc_all':   float((known['tau'] * known['exp'] > 0).mean()),
        'dir_acc_sig':   float(
            (sig_known['tau'] * sig_known['exp'] > 0).mean()
            if len(sig_known) else np.nan
        ),
    }


def run_r_method(rscript, C_obs_int, A, coords, gene_names, spot_names, timeout=1800):
    """Call a simulation R wrapper with Xenium count data.
    C_obs_int: (N_cells, G) integer array
    Returns DataFrame with columns: gene, tau, pval, qval.
    """
    G = C_obs_int.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        counts_f = os.path.join(tmp, 'counts.csv')
        meta_f   = os.path.join(tmp, 'meta.csv')
        out_f    = os.path.join(tmp, 'results.csv')

        pd.DataFrame(C_obs_int.T, index=gene_names,
                     columns=spot_names).to_csv(counts_f)
        pd.DataFrame({'spot_id': spot_names, 'A': A,
                      'x': coords[:, 0], 'y': coords[:, 1]}).to_csv(meta_f, index=False)

        res = subprocess.run(
            ['Rscript', '--vanilla', rscript, counts_f, meta_f, out_f],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0 or not os.path.exists(out_f):
            print(f'  STDERR (last 600 chars): {res.stderr[-600:]}')
            raise RuntimeError(f'{os.path.basename(rscript)} failed')
        r_df = pd.read_csv(out_f)

    g2i    = {g: i for i, g in enumerate(gene_names)}
    pval   = np.ones(G)
    effect = np.full(G, np.nan)

    eff_col = None
    if 'logfc' in r_df.columns:
        eff_col = 'logfc'
    elif 'gee_coef' in r_df.columns:
        eff_col = 'gee_coef'

    for _, row in r_df.iterrows():
        idx = g2i.get(str(row['gene']))
        if idx is not None:
            v = float(row['pvalue'])
            if not np.isnan(v):
                pval[idx] = v
            if eff_col is not None:
                ev = float(row[eff_col])
                if not np.isnan(ev):
                    effect[idx] = ev

    qval    = bh_correct(pval)
    log_cpm = np.log1p(C_obs_int.astype(np.float64))
    dir_raw = log_cpm[A == 1].mean(0) - log_cpm[A == 0].mean(0)
    tau_out = np.where(np.isfinite(effect), effect, dir_raw)
    return pd.DataFrame({'gene': gene_names, 'tau': tau_out,
                         'pval': pval, 'qval': qval})


# ── Load Xenium ST data ────────────────────────────────────────────────────────
print('Loading Xenium ST data...')
sdata   = sd.read_zarr(ZARR_PATH)
st_data = sdata.tables['table']

annotations_st = pd.read_excel(
    pathlib.Path(ANNOT_PATH) / 'Cell_Barcode_Type_Matrices.xlsx',
    sheet_name='Xenium R1 Fig1-5 (supervised)',
)
st_data.obs = st_data.obs.merge(
    annotations_st.set_index('Barcode').loc[
        [int(x) + 1 for x in st_data.obs_names], 'Cluster'
    ],
    left_on='cell_id', right_index=True, how='left',
)
st_data = st_data[st_data.obs['Cluster'].isin(ALL_CLUSTERS)].copy()
st_data.obs['group'] = st_data.obs['Cluster'].replace(
    {'Prolif_Invasive_Tumor': 'Invasive_Tumor', 'DCIS_2': 'DCIS_1'}
)
print(f'  IBC: {(st_data.obs.group=="Invasive_Tumor").sum()}  '
      f'DCIS: {(st_data.obs.group=="DCIS_1").sum()}')

# Genes in observed Xenium panel
panel_lower = {g.lower(): g for g in st_data.var_names}
obs_markers = [g for g in MARKER_GENES if g.lower() in panel_lower]
obs_markers_lower = [g.lower() for g in obs_markers]
print(f'  Marker genes in Xenium panel: {len(obs_markers)}/{len(MARKER_GENES)}')

# Treatment vector: IBC=1, DCIS=0
A          = (st_data.obs['group'] == 'Invasive_Tumor').astype(int).values
spot_names = list(st_data.obs_names)

# Raw integer counts (N_cells × G) for competitors
X_raw = (st_data[:, [panel_lower[g] for g in obs_markers_lower]].X)
X_raw = X_raw.toarray() if issparse(X_raw) else np.asarray(X_raw)
C_obs = np.round(X_raw).astype(np.int64)   # (N, G) integers

# Log-CPM for t-test / SpaGCN
log_cpm = np.log1p(C_obs.astype(np.float64))

# Spatial coordinates (μm from Xenium)
coords = st_data.obsm['spatial'].astype(np.float64)

results = {}   # method → DataFrame(gene, tau, pval, qval)


# ── 1. TIDEST — load pre-computed results, restrict to in-panel genes ──────────
print('\n[1/5] TIDEST — loading hbc_plm_results0.csv')
df_plm = pd.read_csv(PLM_RESULTS)
df_plm['gene'] = df_plm['gene'].str.lower()
df_plm_panel   = df_plm[df_plm['gene'].isin(obs_markers_lower)].copy()
results['TIDEST'] = df_plm_panel[['gene', 'tau', 'pval', 'qval']]
print(f'  In-panel genes: {len(df_plm_panel)} / {len(obs_markers_lower)}'
      f'  (TIDEST total: {len(df_plm)})')
print(f'  Significant (in-panel): {(df_plm_panel.qval < ALPHA).sum()}')


# ── 2. t-test ─────────────────────────────────────────────────────────────────
print('\n[2/5] t-test (Welch, log-CPM observed)...')
rows = []
for i, g in enumerate(obs_markers_lower):
    g1, g0 = log_cpm[A == 1, i], log_cpm[A == 0, i]
    _, p = stats.ttest_ind(g1, g0, equal_var=False)
    rows.append({'gene': g, 'tau': g1.mean() - g0.mean(), 'pval': p})
df_tt = pd.DataFrame(rows)
df_tt['qval'] = bh_correct(df_tt['pval'])
results['t-test'] = df_tt
print(f'  Significant: {(df_tt.qval < ALPHA).sum()}/{len(df_tt)}')


# ── 3. SpaGCN (Wilcoxon rank-sum) ─────────────────────────────────────────────
print('\n[3/5] SpaGCN / Wilcoxon (log-CPM observed)...')
rows = []
for i, g in enumerate(obs_markers_lower):
    g1, g0 = log_cpm[A == 1, i], log_cpm[A == 0, i]
    z, p   = stats.ranksums(g1, g0)
    rows.append({'gene': g, 'tau': z, 'pval': p})
df_wil = pd.DataFrame(rows)
df_wil['qval'] = bh_correct(df_wil['pval'])
results['SpaGCN'] = df_wil
print(f'  Significant: {(df_wil.qval < ALPHA).sum()}/{len(df_wil)}')


# ── 4. DESpace ────────────────────────────────────────────────────────────────
print('\n[4/5] DESpace (svg_test via Rscript)...')
rscript_despace = os.path.join(_SIM_DIR, 'run_despace.R')
df_despace = run_r_method(rscript_despace, C_obs, A, coords,
                          obs_markers_lower, spot_names)
results['DESpace'] = df_despace
print(f'  Significant: {(df_despace.qval < ALPHA).sum()}/{len(df_despace)}')


# ── 5. SpatialGEE — stratified subsample ──────────────────────────────────────
# SpatialGEE fits a GEE per gene; O(N) per gene makes 62k cells intractable
# (~1h timeout). Use a stratified subsample of 5,000 cells (2,500 per group),
# which is still >>300 spots that SpatialGEE was designed for.
print('\n[5/5] SpatialGEE (run_gee_gst via Rscript, stratified subsample N=20000)...')
N_GEE = 20000
rng_gee = np.random.default_rng(seed=42)
ibc_idx_gee  = rng_gee.choice(np.where(A == 1)[0], N_GEE // 2, replace=False)
dcis_idx_gee = rng_gee.choice(np.where(A == 0)[0], N_GEE // 2, replace=False)
gee_idx = np.concatenate([ibc_idx_gee, dcis_idx_gee])
C_obs_gee   = C_obs[gee_idx]
A_gee       = A[gee_idx]
coords_gee  = coords[gee_idx]
spots_gee   = [spot_names[i] for i in gee_idx]
print(f'  Subsample: {(A_gee==1).sum()} IBC + {(A_gee==0).sum()} DCIS')

rscript_spatialgee = os.path.join(_SIM_DIR, 'run_spatialgee.R')
df_spatialgee = run_r_method(rscript_spatialgee, C_obs_gee, A_gee, coords_gee,
                             obs_markers_lower, spots_gee, timeout=1800)
results['SpatialGEE'] = df_spatialgee
print(f'  Significant: {(df_spatialgee.qval < ALPHA).sum()}/{len(df_spatialgee)}')


# ── Score all methods ──────────────────────────────────────────────────────────
print('\n── Scoring ─────────────────────────────────────────────────────────────')
score_rows = []
for name, df in results.items():
    s = score_results(df)
    s['method'] = name
    score_rows.append(s)
    print(f'{name:<12} sig={s["n_sig"]:2d}/{s["n_genes"]}  '
          f'sig_known={s["n_sig_known"]:2d}  '
          f'sig_correct={s["n_sig_correct"]:2d}/{s["n_known"]}  '
          f'dir_all={s["dir_acc_all"]:.1%}  dir_sig={s["dir_acc_sig"]:.1%}')

score_df = pd.DataFrame(score_rows).set_index('method')
score_df.to_csv('./results_raw/hbc_method_comparison.csv')
print('\nSaved ./results_raw/hbc_method_comparison.csv')

# ── Per-gene wide table ────────────────────────────────────────────────────────
wide = pd.DataFrame({'gene': obs_markers_lower}).set_index('gene')
effect_names = {'TIDEST': 'tau', 't-test': 'tau',
                'SpaGCN': 'z', 'DESpace': 'logfc', 'SpatialGEE': 'coef'}
for name, df in results.items():
    df_i = df.set_index('gene')
    key     = name.lower().replace('-', '').replace(' ', '')
    eff_key = effect_names.get(name, 'tau')
    wide[f'{eff_key}_{key}'] = df_i['tau']
    wide[f'pval_{key}']      = df_i['pval']
    wide[f'qval_{key}']      = df_i['qval']

wide.to_csv('./results_raw/hbc_pergene_allmethods.csv')
print('Saved ./results_raw/hbc_pergene_allmethods.csv')


# ── Figure ────────────────────────────────────────────────────────────────────
COLORS = {
    'TIDEST':     '#EE6677',
    't-test':     '#4477AA',
    'SpaGCN':     '#228833',
    'DESpace':    '#CCBB44',
    'SpatialGEE': '#AA3377',
}
method_order = ['TIDEST', 't-test', 'SpaGCN', 'DESpace', 'SpatialGEE']
labels       = ['TIDEST', 't-test', 'SpaGCN\n(Wilcoxon)', 'DESpace', 'SpatialGEE']
colors       = [COLORS[m] for m in method_order]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(method_order))
w = 0.35

n_known = score_df.loc['TIDEST', 'n_known']

# Panel A: n_sig / n_genes + n_sig_correct / n_known
ax = axes[0]
sig_vals  = [score_df.loc[m, 'n_sig']         for m in method_order]
n_genes   = [score_df.loc[m, 'n_genes']        for m in method_order]
corr_vals = [score_df.loc[m, 'n_sig_correct']  for m in method_order]

ax.bar(x - w/2, sig_vals,  w, color=colors, alpha=0.45, label=f'Sig (q<0.05)')
ax.bar(x + w/2, corr_vals, w, color=colors, alpha=0.95, label='Sig & correct direction')
for xi, sv, ng, cv in zip(x, sig_vals, n_genes, corr_vals):
    ax.text(xi - w/2, sv + 0.15, str(sv),  ha='center', va='bottom', fontsize=8)
    ax.text(xi + w/2, cv + 0.15, str(cv),  ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Number of genes')
ax.set_title(f'A   Detections ({obs_markers_lower.__len__()} in-panel genes)', loc='left', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.spines[['top','right']].set_visible(False)

# Panel B: direction accuracy on significant known genes
ax = axes[1]
dir_sig = [score_df.loc[m, 'dir_acc_sig'] * 100 for m in method_order]
bars = ax.bar(x, dir_sig, color=colors, alpha=0.85)
ax.axhline(50, color='gray', ls='--', lw=1, label='Chance (50%)')
for bar, val, m in zip(bars, dir_sig, method_order):
    nk = score_df.loc[m, 'n_sig_known']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%\n(n={nk})', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Direction accuracy (%)')
ax.set_title(f'B   Direction accuracy among significant genes\n({n_known} directional genes)', loc='left', fontweight='bold')
ax.set_ylim(0, 120)
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

fig.suptitle('HBC — method comparison (IBC vs DCIS, Xenium in-panel genes)',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig('./results_raw/hbc_method_comparison.png', dpi=200, bbox_inches='tight')
print('Saved ./results_raw/hbc_method_comparison.png')
plt.close()
