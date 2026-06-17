"""
Competitor comparison and ablation study for the MB application.

Compares six methods on the 55 cortical layer marker genes with known expected directions:
  1. t-test        (observed ST data, no imputation, no spatial control)
  2. Wilcoxon      (observed ST data, no imputation, no spatial control)
  3. t-test pseudo (pseudo-outcome, no spatial control)
  4. PLM no-impute (raw Tangram prediction + SpatialPCA, no Pearson correction)
  5. PLM no-spatial(pseudo-outcome + log_lib only, no SpatialPCA)
  6. PLM full      (pseudo-outcome + SpatialPCA + log_lib) — our method

Ground truth: EXPECTED_SIGN dict (38 genes with known layer direction).

Outputs:
  ./results_raw/mb_ablation_results.csv
  ./results_raw/mb_ablation_comparison.png
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

import squidpy as sq
import scanpy as sc
from scanpy.preprocessing import normalize_total

# ── Constants ──────────────────────────────────────────────────────────────────
N_FOLDS      = 2
N_ESTIMATORS = 200
ALPHA        = 0.05
CORR_THRESHOLD = 0.5

MARKER_GENES = [
    'Cux1','Cux2','Calb1','Rasgrf2','Pou3f2','Pou3f3','Satb2','Prss12','Sema3c','Lhx2',
    'Rorb','Rspo1','Scnn1a','Plcxd2','Krt73',
    'Bcl11b','Fezf2','Foxp2','Sox5','Tbr1','Tle4','Tshz3','Ldb2','Sulf1','Cdh13',
    'Syt6','Nxph4','Nfe2l1','Etv1','Tdo2','Chrna6','Nxph2','Syt17',
    'Grik1','Trhr','Npsr1','Chrna4','Htr2c',
    'Neurod6','Neurod2','Nrn1','Camk2a','Emx1','Rbfox3',
    'Pvalb','Sst','Vip','Reln','Lhx6',
    'Crym','Ntng1','Slc17a6','Slc17a7','Gad1','Gad2',
]

EXPECTED_SIGN = {}
for g in ['Cux1','Cux2','Calb1','Rasgrf2','Pou3f2','Pou3f3','Satb2','Prss12',
          'Sema3c','Lhx2','Rorb','Rspo1','Scnn1a','Plcxd2','Krt73']:
    EXPECTED_SIGN[g.lower()] = +1
for g in ['Bcl11b','Fezf2','Foxp2','Sox5','Tbr1','Tle4','Tshz3','Ldb2','Sulf1',
          'Cdh13','Syt6','Nxph4','Nfe2l1','Etv1','Tdo2','Chrna6','Nxph2','Syt17',
          'Grik1','Trhr','Npsr1','Chrna4','Htr2c']:
    EXPECTED_SIGN[g.lower()] = -1


# ── PLM helper ─────────────────────────────────────────────────────────────────
def robinson_plm(Y, A, U, n_folds=N_FOLDS, n_estimators=N_ESTIMATORS, seed=42):
    n = len(A)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    A_res = np.zeros(n)
    Y_res = np.zeros_like(Y, dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(kf.split(U)):
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
        clf.fit(U[train_idx], A[train_idx])
        A_res[val_idx] = A[val_idx] - clf.predict_proba(U[val_idx])[:, 1]
        reg = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
        reg.fit(U[train_idx], Y[train_idx])
        Y_res[val_idx] = Y[val_idx] - reg.predict(U[val_idx])
    denom     = float(A_res @ A_res)
    tau       = (A_res @ Y_res) / denom
    influence = A_res[:, None] * (Y_res - A_res[:, None] * tau)
    se        = np.sqrt((influence ** 2).sum(axis=0)) / denom
    return tau, se


def run_plm(Y, A, U, gene_names):
    tau, se = robinson_plm(Y, A, U)
    z    = tau / np.where(se > 0, se, np.nan)
    pval = 2 * stats.norm.sf(np.abs(z))
    _, qval, _, _ = multipletests(pval[~np.isnan(pval)], method='fdr_bh')
    qval_full = np.full(len(pval), np.nan)
    qval_full[~np.isnan(pval)] = qval
    return pd.DataFrame({'gene': gene_names, 'tau': tau, 'se': se,
                         'z': z, 'pval': pval, 'qval': qval_full})


def score_results(df, stat_col='tau', sig_col='qval', expected=EXPECTED_SIGN):
    """Return (n_tested, n_sig, n_sig_correct, n_correct_all, direction_acc)."""
    df = df.copy()
    df['exp'] = df['gene'].map(expected)
    df_known  = df[df['exp'].notna()]
    n_tested  = len(df_known)
    sig       = df_known[df_known[sig_col] < ALPHA] if sig_col else df_known
    # sign: positive tau means same direction as +1 in EXPECTED_SIGN
    sig_corr  = (sig[stat_col] * sig['exp'] > 0).sum()
    all_corr  = (df_known[stat_col] * df_known['exp'] > 0).sum()
    return {
        'n_tested':     n_tested,
        'n_sig':        (df[sig_col] < ALPHA).sum() if sig_col else len(df),
        'n_sig_correct': int(sig_corr),
        'n_correct_all': int(all_corr),
        'dir_acc_all':   all_corr / n_tested if n_tested else np.nan,
        'dir_acc_sig':   sig_corr / len(sig) if len(sig) else np.nan,
    }


# ── Load shared data ───────────────────────────────────────────────────────────
print('Loading data...')

# Observed ST data (Cortex_1 and Cortex_3 only)
adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[adata_st.obs.cluster.isin(['Cortex_1', 'Cortex_3'])].copy()
adata_st.var_names = pd.Index(adata_st.var_names.str.lower())

# Pseudo-outcome (log1p, full SC gene panel)
with open('./results_raw/tangram_pseudo0.pkl', 'rb') as f:
    pseudo = pickle.load(f)

# Raw Tangram prediction (scaled, log1p — for ablation "no Pearson correction")
with open('./results_raw/tangram_pred0.pkl', 'rb') as f:
    ad_ge_raw = pickle.load(f)
ad_ge_raw = ad_ge_raw[ad_ge_raw.obs.cluster.isin(['Cortex_1', 'Cortex_3'])].copy()
ad_ge_raw.var_names = pd.Index(ad_ge_raw.var_names.str.lower())
if ad_ge_raw.var_names.duplicated().any():
    ad_ge_raw = ad_ge_raw[:, ~ad_ge_raw.var_names.duplicated()].copy()
# Per-spot scaling: Tangram weighted-avg → ST count scale, then log1p
adata_st_obs = adata_st[:, adata_st.var_names.isin(ad_ge_raw.var_names)].copy()
N_s = np.asarray(adata_st_obs.X.sum(axis=1)).ravel()
ge_obs_sums = np.asarray(
    ad_ge_raw[:, adata_st_obs.var_names].X.toarray()
    if issparse(ad_ge_raw.X) else
    ad_ge_raw[:, adata_st_obs.var_names].X
).sum(axis=1)
scale = N_s / ge_obs_sums
X_raw = np.asarray(ad_ge_raw.X.toarray() if issparse(ad_ge_raw.X) else ad_ge_raw.X).astype(np.float32)
X_raw_scaled = X_raw * scale[:, None]
log_pred_raw = np.log1p(X_raw_scaled)

# SpatialPCA PCs
U_df = pd.read_feather('./results_raw/spatialPCA_pcs0.feather').set_index('spot')
spot_order = U_df.index.tolist()
U_full = U_df.values.astype(np.float64)

# PC-treatment correlation screening (same as mb_plm.py)
A = (pseudo.obs.loc[spot_order, 'cluster'] == 'Cortex_3').astype(int).values
pc_corrs  = np.array([np.corrcoef(U_full[:, i], A)[0, 1] for i in range(U_full.shape[1])])
keep_mask = np.abs(pc_corrs) < CORR_THRESHOLD
U_pca     = U_full[:, keep_mask]
print(f'SpatialPCA: kept {keep_mask.sum()}/{len(keep_mask)} PCs (|corr|<{CORR_THRESHOLD})')

log_lib = np.log1p(
    np.expm1(np.asarray(pseudo[spot_order].X).astype(np.float64)).sum(axis=1, keepdims=True)
)

pseudo_genes = set(pseudo.var_names)
markers_lower = [g.lower() for g in MARKER_GENES]
keep = [g for g in markers_lower if g in pseudo_genes]
print(f'Marker genes found in pseudo-outcome: {len(keep)}/{len(MARKER_GENES)}')

results_store = {}

# ── Method 1: t-test on observed ST ───────────────────────────────────────────
print('\n[1/6] t-test (observed ST)...')
obs_markers = [g for g in markers_lower if g in adata_st.var_names]
print(f'  Observed markers: {len(obs_markers)}/{len(markers_lower)}')
spot_order_arr = np.array(spot_order)
A_obs = (adata_st[spot_order_arr].obs['cluster'] == 'Cortex_3').astype(int).values
X_obs = np.asarray(
    adata_st[spot_order_arr, obs_markers].X.toarray()
    if issparse(adata_st.X) else
    adata_st[spot_order_arr, obs_markers].X
).astype(np.float64)
log_obs = np.log1p(X_obs)

rows = []
for i, g in enumerate(obs_markers):
    grp1 = log_obs[A_obs == 1, i]
    grp0 = log_obs[A_obs == 0, i]
    t, p = stats.ttest_ind(grp1, grp0)
    rows.append({'gene': g, 'tau': grp1.mean() - grp0.mean(), 'pval': p})
df_ttest_obs = pd.DataFrame(rows)
_, qval, _, _ = multipletests(df_ttest_obs['pval'], method='fdr_bh')
df_ttest_obs['qval'] = qval
results_store['t-test\n(observed)'] = df_ttest_obs
print(f'  Significant: {(df_ttest_obs.qval < ALPHA).sum()}/{len(df_ttest_obs)}')

# ── Method 2: Wilcoxon on observed ST ─────────────────────────────────────────
print('\n[2/6] Wilcoxon (observed ST)...')
rows = []
for i, g in enumerate(obs_markers):
    grp1 = log_obs[A_obs == 1, i]
    grp0 = log_obs[A_obs == 0, i]
    stat, p = stats.ranksums(grp1, grp0)
    rows.append({'gene': g, 'tau': grp1.mean() - grp0.mean(), 'pval': p})
df_wilcox_obs = pd.DataFrame(rows)
_, qval, _, _ = multipletests(df_wilcox_obs['pval'], method='fdr_bh')
df_wilcox_obs['qval'] = qval
results_store['Wilcoxon\n(observed)'] = df_wilcox_obs
print(f'  Significant: {(df_wilcox_obs.qval < ALPHA).sum()}/{len(df_wilcox_obs)}')

# ── Method 3: t-test on pseudo-outcome ────────────────────────────────────────
print('\n[3/6] t-test (pseudo-outcome)...')
Y_pseudo = np.asarray(pseudo[spot_order, keep].X).astype(np.float64)
rows = []
for i, g in enumerate(keep):
    grp1 = Y_pseudo[A == 1, i]
    grp0 = Y_pseudo[A == 0, i]
    t, p = stats.ttest_ind(grp1, grp0)
    rows.append({'gene': g, 'tau': grp1.mean() - grp0.mean(), 'pval': p})
df_ttest_pseudo = pd.DataFrame(rows)
_, qval, _, _ = multipletests(df_ttest_pseudo['pval'], method='fdr_bh')
df_ttest_pseudo['qval'] = qval
results_store['t-test\n(pseudo)'] = df_ttest_pseudo
print(f'  Significant: {(df_ttest_pseudo.qval < ALPHA).sum()}/{len(df_ttest_pseudo)}')

# ── Method 4: PLM — no Pearson correction (raw Tangram + SpatialPCA) ──────────
print('\n[4/6] PLM (no Pearson correction = raw Tangram)...')
raw_markers = [g for g in markers_lower if g in ad_ge_raw.var_names]
print(f'  Markers in raw Tangram: {len(raw_markers)}/{len(markers_lower)}')
# Align spot order: raw Tangram uses same obs as pseudo
raw_spot_order = [s for s in spot_order if s in ad_ge_raw.obs_names]
raw_spot_idx   = [list(ad_ge_raw.obs_names).index(s) for s in raw_spot_order]
Y_raw = log_pred_raw[np.array(raw_spot_idx)][:, [list(ad_ge_raw.var_names).index(g) for g in raw_markers]]
A_raw = (pseudo.obs.loc[raw_spot_order, 'cluster'] == 'Cortex_3').astype(int).values
U_raw_pca = U_df.loc[raw_spot_order].values[:, keep_mask].astype(np.float64)
log_lib_raw = np.log1p(np.expm1(
    np.asarray(pseudo[raw_spot_order].X).astype(np.float64)
).sum(axis=1, keepdims=True))
U_raw = np.hstack([U_raw_pca, log_lib_raw])
df_plm_noimpute = run_plm(Y_raw, A_raw, U_raw, raw_markers)
results_store['PLM\n(no correction)'] = df_plm_noimpute
print(f'  Significant: {(df_plm_noimpute.qval < ALPHA).sum()}/{len(df_plm_noimpute)}')

# ── Method 5: PLM — no SpatialPCA (pseudo-outcome + log_lib only) ─────────────
print('\n[5/6] PLM (no spatial = pseudo-outcome + log_lib only)...')
Y = np.asarray(pseudo[spot_order, keep].X).astype(np.float64)
U_nospatial = log_lib  # only library size, no SpatialPCA
df_plm_nospatial = run_plm(Y, A, U_nospatial, keep)
results_store['PLM\n(no spatial)'] = df_plm_nospatial
print(f'  Significant: {(df_plm_nospatial.qval < ALPHA).sum()}/{len(df_plm_nospatial)}')

# ── Method 6: PLM full — load from existing results ───────────────────────────
print('\n[6/6] PLM full (our method) — loading plm_results0.csv...')
df_plm_full = pd.read_csv('./results_raw/plm_results0.csv')
results_store['PLM full\n(ours)'] = df_plm_full
print(f'  Significant: {(df_plm_full.qval < ALPHA).sum()}/{len(df_plm_full)}')

# ── Score all methods ──────────────────────────────────────────────────────────
print('\n── Scoring ──')
rows_score = []
for name, df in results_store.items():
    s = score_results(df, stat_col='tau', sig_col='qval')
    s['method'] = name
    rows_score.append(s)
    print(f'{name.replace(chr(10), " "):<28} '
          f'tested={s["n_tested"]:2d}  sig={s["n_sig"]:2d}  '
          f'sig_corr={s["n_sig_correct"]:2d}  '
          f'dir_all={s["dir_acc_all"]:.1%}  dir_sig={s["dir_acc_sig"]:.1%}')

score_df = pd.DataFrame(rows_score).set_index('method')
score_df.to_csv('./results_raw/mb_ablation_results.csv')
print('Saved ./results_raw/mb_ablation_results.csv')

# ── Plot ───────────────────────────────────────────────────────────────────────
methods   = list(results_store.keys())
n_methods = len(methods)
x         = np.arange(n_methods)
width     = 0.28

n_sig_vals       = [score_df.loc[m, 'n_sig']         for m in methods]
n_sig_corr_vals  = [score_df.loc[m, 'n_sig_correct'] for m in methods]
n_tested_vals    = [score_df.loc[m, 'n_tested']       for m in methods]
dir_all_vals     = [score_df.loc[m, 'dir_acc_all']    for m in methods]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: n_sig and n_sig_correct bars
ax = axes[0]
bars1 = ax.bar(x - width/2, n_sig_vals,      width, label='Significant (q<0.05)',         color='#4878CF', alpha=0.85)
bars2 = ax.bar(x + width/2, n_sig_corr_vals, width, label='Significant & correct direction', color='#6ACC65', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Number of marker genes', fontsize=11)
ax.set_title('A   Marker gene recovery (55 genes tested\nwhere available, 38 with known direction)',
             fontsize=10, loc='left')
ax.legend(fontsize=9)
ax.set_ylim(0, max(n_sig_vals) * 1.2)
# annotate n_tested above each method label
for i, (nt, ns) in enumerate(zip(n_tested_vals, n_sig_vals)):
    ax.text(i, max(n_sig_vals) * 1.13, f'n={nt}', ha='center', va='bottom', fontsize=8, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: direction accuracy (all genes with known sign)
ax = axes[1]
colors = ['#4878CF' if 'PLM full' not in m else '#D65F5F' for m in methods]
bars = ax.bar(x, [v * 100 for v in dir_all_vals], color=colors, alpha=0.85)
ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Direction accuracy (%, all genes with known sign)', fontsize=10)
ax.set_title('B   Direction accuracy on genes with known expected sign\n(38 genes)',
             fontsize=10, loc='left')
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
for bar, val in zip(bars, dir_all_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./results_raw/mb_ablation_comparison.png', dpi=200, bbox_inches='tight')
print('Saved ./results_raw/mb_ablation_comparison.png')
plt.close()
