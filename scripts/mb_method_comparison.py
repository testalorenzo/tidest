"""
Competitor comparison for the MB (mouse brain) cortical layer application.

Compares five methods on the 55 cortical layer marker genes:
  1. tidest        — Robinson PLM (load from plm_results0.csv)
  2. t-test        — Welch t-test on log1p observed counts
  3. SpaGCN        — Wilcoxon rank-sum on log1p observed counts
  4. DESpace       — edgeR NB via svg_test (R subprocess)
  5. SpatialGEE    — GEE Poisson via run_gee_gst (R subprocess)

Ground truth: EXPECTED_SIGN dict (38 genes with known cortical layer direction).

Outputs:
  results_raw/mb_method_comparison.csv
  results_raw/mb_method_comparison.png
"""

import os, sys, subprocess, tempfile
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import squidpy as sq

_ROOT     = os.path.dirname(os.path.abspath(__file__ + '/..'))
_SIM_DIR  = os.path.join(_ROOT, 'scripts', 'simulation')

# ── Constants ──────────────────────────────────────────────────────────────────
ALPHA          = 0.05
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
MARKERS_LOWER = [g.lower() for g in MARKER_GENES]

EXPECTED_SIGN = {}
for g in ['Cux1','Cux2','Calb1','Rasgrf2','Pou3f2','Pou3f3','Satb2','Prss12',
          'Sema3c','Lhx2','Rorb','Rspo1','Scnn1a','Plcxd2','Krt73']:
    EXPECTED_SIGN[g.lower()] = +1
for g in ['Bcl11b','Fezf2','Foxp2','Sox5','Tbr1','Tle4','Tshz3','Ldb2','Sulf1',
          'Cdh13','Syt6','Nxph4','Nfe2l1','Etv1','Tdo2','Chrna6','Nxph2','Syt17',
          'Grik1','Trhr','Npsr1','Chrna4','Htr2c']:
    EXPECTED_SIGN[g.lower()] = -1


# ── Shared helpers ─────────────────────────────────────────────────────────────

def bh_correct(pvals):
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    return qvals


def score_results(df):
    """Return summary dict for one method's result df (columns: gene, tau, qval)."""
    df = df.copy()
    df['exp'] = df['gene'].map(EXPECTED_SIGN)
    known = df[df['exp'].notna()]
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


def run_r_method(rscript, C_obs_int, A, coords, gene_names, spot_names, timeout=600):
    """
    Call one of the simulation R scripts with the real MB count data.
    Returns DataFrame with columns: gene, tau, pval, qval.

    Native effect estimates:
      - DESpace:    edgeR log2 fold change (logfc column, via verbose=TRUE)
      - SpatialGEE: Poisson GEE coefficient from full model (gee_coef column)
    'tau' stores the native effect estimate for direction scoring and reporting.
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
            raise RuntimeError(f'{os.path.basename(rscript)} failed:\n{res.stderr[-800:]}')
        r_df = pd.read_csv(out_f)

    g2i    = {g: i for i, g in enumerate(gene_names)}
    pval   = np.ones(G)
    effect = np.full(G, np.nan)

    # Detect native effect column
    eff_col = None
    if 'logfc' in r_df.columns:
        eff_col = 'logfc'       # DESpace: log2 fold change (cluster1 vs cluster0)
    elif 'gee_coef' in r_df.columns:
        eff_col = 'gee_coef'    # SpatialGEE: Poisson log coefficient (trt vs ctrl)

    for _, row in r_df.iterrows():
        idx = g2i.get(str(row['gene']))
        if idx is not None:
            if not np.isnan(float(row['pvalue'])):
                pval[idx] = float(row['pvalue'])
            if eff_col is not None:
                v = float(row[eff_col])
                if not np.isnan(v):
                    effect[idx] = v

    qval = bh_correct(pval)
    # For genes where the native effect is NaN (R method failed or filtered the
    # gene), fall back to the sign of the log1p group mean difference so that
    # direction scoring in score_results() is never based on NaN.
    Y_log   = np.log1p(C_obs_int.astype(np.float64))
    dir_raw = Y_log[A == 1].mean(0) - Y_log[A == 0].mean(0)
    tau_out = np.where(np.isfinite(effect), effect, dir_raw)
    return pd.DataFrame({'gene': gene_names, 'tau': tau_out,
                         'pval': pval, 'qval': qval})


# ── Load observed ST data ──────────────────────────────────────────────────────
print('Loading MB observed ST data...')
adata = sq.datasets.visium_fluo_adata_crop()
adata = adata[adata.obs.cluster.isin(['Cortex_1', 'Cortex_3'])].copy()
adata.var_names = pd.Index(adata.var_names.str.lower())

# Observed markers present in data
obs_markers = [g for g in MARKERS_LOWER if g in adata.var_names]
print(f'Marker genes in observed data: {len(obs_markers)}/{len(MARKERS_LOWER)}')

# Treatment vector: Cortex_3 = 1, Cortex_1 = 0
A = (adata.obs['cluster'] == 'Cortex_3').astype(int).values
spot_names = list(adata.obs_names)

# Raw integer counts (genes × spots → spots × genes after transpose)
X_raw = (adata[:, obs_markers].X.toarray()
         if issparse(adata.X) else
         np.asarray(adata[:, obs_markers].X))
C_obs = np.round(X_raw).astype(np.int64)   # (N, G) integers

# Log1p for t-test / Wilcoxon
log_obs = np.log1p(C_obs.astype(np.float64))

# Spatial coordinates
coords = adata.obsm['spatial'].astype(np.float64)   # (N, 2)

results = {}   # method name → DataFrame(gene, tau, pval, qval)


# ── 1. tidest — load pre-computed results ─────────────────────────────────────
print('\n[1/5] tidest — loading plm_results0.csv')
df_plm = pd.read_csv('./results_raw/plm_results0.csv')
df_plm['gene'] = df_plm['gene'].str.lower()
df_plm = df_plm[df_plm['gene'].isin(obs_markers)].copy()
results['tidest'] = df_plm[['gene', 'tau', 'pval', 'qval']]
print(f'  Significant: {(df_plm.qval < ALPHA).sum()}/{len(df_plm)}')


# ── 2. t-test ─────────────────────────────────────────────────────────────────
# Native effect: group mean difference of log1p counts.
print('\n[2/5] t-test (Welch, log1p observed)...')
rows = []
for i, g in enumerate(obs_markers):
    g1, g0 = log_obs[A == 1, i], log_obs[A == 0, i]
    _, p = stats.ttest_ind(g1, g0)
    rows.append({'gene': g, 'tau': g1.mean() - g0.mean(), 'pval': p})
df_tt = pd.DataFrame(rows)
df_tt['qval'] = bh_correct(df_tt['pval'])
results['t-test'] = df_tt
print(f'  Significant: {(df_tt.qval < ALPHA).sum()}/{len(df_tt)}')


# ── 3. SpaGCN (Wilcoxon) ──────────────────────────────────────────────────────
# SpaGCN uses a Wilcoxon rank-sum test internally.
# scipy.stats.ranksums returns a signed z-statistic: z > 0 means group1 (A=1,
# Cortex_3, superficial) has higher ranks. This z-statistic IS the Wilcoxon
# effect measure and determines both direction and relative magnitude.
print('\n[3/5] SpaGCN / Wilcoxon (log1p observed)...')
rows = []
for i, g in enumerate(obs_markers):
    g1, g0 = log_obs[A == 1, i], log_obs[A == 0, i]
    z, p   = stats.ranksums(g1, g0)   # z is signed: >0 means g1 > g0 in rank
    rows.append({'gene': g, 'tau': z, 'pval': p})
df_wil = pd.DataFrame(rows)
df_wil['qval'] = bh_correct(df_wil['pval'])
results['SpaGCN'] = df_wil
print(f'  Significant: {(df_wil.qval < ALPHA).sum()}/{len(df_wil)}')


# ── 4. DESpace ────────────────────────────────────────────────────────────────
print('\n[4/5] DESpace (svg_test via Rscript)...')
# DESpace uses cluster labels as spatial clusters; treatment A encodes cluster.
rscript_despace = os.path.join(_SIM_DIR, 'run_despace.R')
df_despace = run_r_method(rscript_despace, C_obs, A, coords, obs_markers, spot_names)
results['DESpace'] = df_despace
print(f'  Significant: {(df_despace.qval < ALPHA).sum()}/{len(df_despace)}')


# ── 5. SpatialGEE ─────────────────────────────────────────────────────────────
print('\n[5/5] SpatialGEE (run_gee_gst via Rscript)...')
rscript_spatialgee = os.path.join(_SIM_DIR, 'run_spatialgee.R')
df_spatialgee = run_r_method(rscript_spatialgee, C_obs, A, coords, obs_markers, spot_names)
results['SpatialGEE'] = df_spatialgee
print(f'  Significant: {(df_spatialgee.qval < ALPHA).sum()}/{len(df_spatialgee)}')


# ── Score all methods ──────────────────────────────────────────────────────────
print('\n── Scoring ──────────────────────────────────────────────────────────────')
score_rows = []
for name, df in results.items():
    s = score_results(df)
    s['method'] = name
    score_rows.append(s)
    print(f'{name:<12} sig={s["n_sig"]:2d}  sig_known={s["n_sig_known"]:2d}  '
          f'sig_correct={s["n_sig_correct"]:2d}  '
          f'dir_all={s["dir_acc_all"]:.1%}  dir_sig={s["dir_acc_sig"]:.1%}')

score_df = pd.DataFrame(score_rows).set_index('method')
score_df.to_csv('./results_raw/mb_method_comparison.csv')
print('\nSaved ./results_raw/mb_method_comparison.csv')

# ── Per-gene wide table (tau + qval for every method × gene) ──────────────────
# Indexed by gene; columns: tau_<method>, qval_<method> for each method.
# Used by scientific findings documents to look up individual gene results.
wide = pd.DataFrame({'gene': obs_markers}).set_index('gene')
# Effect column guide (each method's native statistic):
#   tau_tidest    — PLM debiased τ (log pseudo-outcome scale)
#   z_spagcn      — Wilcoxon signed z-statistic (z>0: superficial enrichment)
#   logfc_despace — edgeR log2 fold change (cluster1 vs cluster0; >0: superficial)
#   coef_spatialgee — GEE Poisson log-coefficient (trt vs ctrl; >0: superficial)
#   tau_ttest     — log1p group mean difference (Cortex_3 - Cortex_1)
effect_names = {'tidest': 'tau', 'ttest': 'tau',
                'spagcn': 'z', 'despace': 'logfc', 'spatialgee': 'coef'}
for name, df in results.items():
    df_indexed = df.set_index('gene')
    key     = name.lower().replace('-', '').replace(' ', '')
    eff_key = effect_names.get(key, 'tau')
    wide[f'{eff_key}_{key}'] = df_indexed['tau']
    wide[f'pval_{key}']      = df_indexed['pval']
    wide[f'qval_{key}']      = df_indexed['qval']

wide.to_csv('./results_raw/mb_pergene_allmethods.csv')
print('Saved ./results_raw/mb_pergene_allmethods.csv')


# ── Figure ────────────────────────────────────────────────────────────────────
COLORS = {
    'tidest':     '#EE6677',
    't-test':     '#4477AA',
    'SpaGCN':     '#228833',
    'DESpace':    '#CCBB44',
    'SpatialGEE': '#AA3377',
}
method_order = ['tidest', 't-test', 'SpaGCN', 'DESpace', 'SpatialGEE']
labels       = ['tidest', 't-test', 'SpaGCN\n(Wilcoxon)', 'DESpace', 'SpatialGEE']
colors       = [COLORS[m] for m in method_order]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.rcParams.update({'font.size': 10})
x = np.arange(len(method_order))
w = 0.35

# Panel A: n_sig total (bars) + n_sig_known (darker inner bars)
ax = axes[0]
sig_vals  = [score_df.loc[m, 'n_sig']       for m in method_order]
known_vals= [score_df.loc[m, 'n_sig_known'] for m in method_order]
corr_vals = [score_df.loc[m, 'n_sig_correct'] for m in method_order]
ax.bar(x - w/2, sig_vals,   w, color=colors, alpha=0.5,  label='Sig (q<0.05, all genes)')
ax.bar(x + w/2, corr_vals,  w, color=colors, alpha=0.95, label='Sig & correct direction')
for xi, sv, kv in zip(x, sig_vals, known_vals):
    ax.text(xi - w/2, sv + 0.3, str(sv),  ha='center', va='bottom', fontsize=8)
    ax.text(xi + w/2, kv + 0.3 if kv else sv + 0.3,
            str(int(score_df.loc[method_order[int(xi)], 'n_sig_correct'])),
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Number of marker genes'); ax.set_ylim(0, max(sig_vals)*1.25)
ax.set_title('A   Significant detections\n(55 genes tested)', loc='left', fontweight='bold')
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

# Panel B: direction accuracy on ALL 38 known genes
ax = axes[1]
dir_all = [score_df.loc[m, 'dir_acc_all'] * 100 for m in method_order]
bars = ax.bar(x, dir_all, color=colors, alpha=0.85)
ax.axhline(50, color='gray', ls='--', lw=1, label='Chance (50%)')
for bar, val in zip(bars, dir_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Direction accuracy (%)'); ax.set_ylim(0, 110)
ax.set_title('B   Direction accuracy\n(38 genes with known sign)', loc='left', fontweight='bold')
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

# Panel C: direction accuracy on SIGNIFICANT genes only
ax = axes[2]
dir_sig = [score_df.loc[m, 'dir_acc_sig'] * 100 for m in method_order]
bars = ax.bar(x, dir_sig, color=colors, alpha=0.85)
ax.axhline(50, color='gray', ls='--', lw=1, label='Chance (50%)')
for bar, val, m in zip(bars, dir_sig, method_order):
    nk = score_df.loc[m, 'n_sig_known']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%\n(n={nk})', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Direction accuracy (%)'); ax.set_ylim(0, 120)
ax.set_title('C   Direction accuracy\n(significant genes with known sign)', loc='left', fontweight='bold')
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

fig.suptitle('MB application — competitor comparison (55 cortical layer marker genes)',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('./results_raw/mb_method_comparison.png', dpi=200, bbox_inches='tight')
print('Saved ./results_raw/mb_method_comparison.png')
plt.close()
