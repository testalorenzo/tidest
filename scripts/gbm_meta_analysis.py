"""
Random-effects (DerSimonian-Laird) + fixed-effects meta-analysis across 26 GBM samples.

Inputs:
  ./results_raw/gbm_plm_results_all.csv   (sample, gene, tau, se, z, pval, qval)

Outputs:
  ./results_raw/gbm_meta_results.csv           — RE + FE pooled τ, SE, z, q, I²
  ./results_raw/gbm_sample_heatmap.png         — cross-sample τ heatmap
  ./results_raw/gbm_meta_summary.png           — pooled effect sizes lollipop (RE)
  ./results_raw/gbm_meta_forest.png            — forest plot for key genes (RE)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Expected directions (all 90 panel genes) ──────────────────────────────────
# +1 = LE-enriched  |  -1 = CT-enriched  |  0 = no clear prior
EXPECTED_SIGN = {
    # Neuronal / normal-brain LE markers
    'syt1': +1, 'snap25': +1, 'rbfox3': +1, 'tubb3': +1, 'dcx': +1,
    'map2': +1, 'nrxn1': +1, 'dlg4': +1, 'ckb': +1,
    # Normal-brain oligodendrocyte / myelin
    'mbp': +1, 'plp1': +1, 'mag': +1, 'mog': +1,
    # Normal-brain astrocyte
    'sparcl1': +1, 'slc1a2': +1,
    # OPC / NPC tumor states (expected LE by cell-type argument;
    # empirically show CT-enrichment = OPC paradox)
    'olig1': +1, 'olig2': +1, 'pdgfra': +1, 'sox10': +1,
    'sox11': +1, 'ascl1': +1, 'sox4': +1, 'egr1': +1, 'ccnd2': +1,
    # Invasion / LE ECM
    'mmp9': +1, 'mmp2': +1, 'plaur': +1, 'itgb1': +1, 'cxcr4': +1,
    'l1cam': +1, 'nrcam': +1, 'ptprz1': +1, 'bcan': +1, 'htra1': +1,
    'ldhb': +1,
    # Homeostatic microglia (LE-enriched, replaced by TAMs in CT)
    'hexb': +1, 'sall1': +1, 'olfml3': +1, 'cx3cr1': +1,
    'tmem119': +1, 'p2ry12': +1,
    # Vascular / pericyte
    'pecam1': +1, 'vwf': +1, 'cldn5': +1,
    # Immune (microglial/NK in LE)
    'aif1': +1, 'cd68': +1,
    # Proliferation (CT core)
    'top2a': -1, 'mki67': -1, 'pcna': -1, 'mcm2': -1, 'mcm6': -1,
    # AC / GBM-astrocyte state (CT-enriched)
    'gfap': -1, 'aqp4': -1, 's100b': -1,
    'apoe': -1, 'aldoc': -1, 'glul': -1, 'gja1': -1,
    # MES state (CT-enriched)
    'cd44': -1, 'tnc': -1, 'chi3l1': -1, 'fn1': -1, 'vim': -1,
    'tgfbi': -1, 'col1a1': -1, 'serpine1': -1,
    # RTK / EGFR amplification (CT)
    'egfr': -1,
    # Hypoxia (CT core)
    'hif1a': -1, 'ca9': -1, 'ldha': -1, 'hk2': -1,
    # TAM / M2 macrophage (CT-enriched)
    'spp1': -1, 'lgals1': -1, 'lgals3': -1, 'cd163': -1, 'msr1': -1, 'mrc1': -1,
    # CT ECM / invasion scaffold
    'vcan': -1, 'anxa2': -1, 'postn': -1, 'epas1': -1,
    # No clear prior / contested
    'met':    0,   # invasion receptor but also expressed in CT
    'vegfa':  0,   # hypoxia-CT but also angiogenesis at invasion front
    'il6':    0,   # inflammatory, both compartments
    'cdkn2a': 0,   # tumor suppressor loss, absent in most GBM
    'nes':    0,   # stem-cell marker, both CT and LE
    'stat3':  0,   # signaling hub, both compartments
    'ptpn11': 0,   # RTK signaling, ambiguous
    'idh1':   0,   # IDH1 R132H rare in GBM; expression itself not directional
    'pten':   0,   # tumor suppressor, ambiguous
}

SAMPLES = [
    'MGH258',
    'UKF243','UKF248','UKF251','UKF255','UKF259','UKF260',
    'UKF266','UKF269','UKF275','UKF296','UKF304','UKF313','UKF334',
    'ZH1007inf','ZH1007nec','ZH1019inf','ZH1019T1',
    'ZH8811Abulk','ZH8811Bbulk','ZH8812bulk',
    'ZH881inf','ZH881T1',
    'ZH916bulk','ZH916inf','ZH916T1',
]

ALPHA = 0.05

# ── Load combined results ──────────────────────────────────────────────────────
print('Loading combined GBM results...')
df = pd.read_csv('./results_raw/gbm_plm_results_all.csv')
df['se']  = df['se'].clip(lower=1e-6)   # guard against numerical zeros
genes     = sorted(df['gene'].unique())
n_genes   = len(genes)
n_samples = len(SAMPLES)
print(f'{n_genes} genes × {n_samples} samples')

# ── DerSimonian-Laird random-effects meta-analysis ────────────────────────────
def dersimonian_laird(tau_vec, se_vec):
    """
    DerSimonian-Laird random-effects meta-analysis.
    Returns: tau_FE, se_FE, tau_RE, se_RE, tau2 (between-study variance), I2, Q, Q_pval
    """
    k    = len(tau_vec)
    w_fe = 1.0 / se_vec**2

    # Fixed-effects pooled estimate
    tau_fe = (w_fe * tau_vec).sum() / w_fe.sum()
    se_fe  = 1.0 / np.sqrt(w_fe.sum())

    # Cochran's Q
    Q      = (w_fe * (tau_vec - tau_fe)**2).sum()
    Q_p    = 1 - stats.chi2.cdf(Q, df=k - 1)

    # DL between-study variance estimate
    c      = w_fe.sum() - (w_fe**2).sum() / w_fe.sum()
    tau2   = max(0.0, (Q - (k - 1)) / c)

    # I² (proportion of variance due to heterogeneity)
    I2     = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0

    # Random-effects weights and pooled estimate
    w_re   = 1.0 / (se_vec**2 + tau2)
    tau_re = (w_re * tau_vec).sum() / w_re.sum()
    se_re  = 1.0 / np.sqrt(w_re.sum())

    return tau_fe, se_fe, tau_re, se_re, tau2, I2, Q, Q_p


print('Running DerSimonian-Laird random-effects meta-analysis...')
meta_rows = []
for gene in genes:
    sub   = df[df['gene'] == gene].copy()
    tau_v = sub['tau'].values
    se_v  = sub['se'].values

    tau_fe, se_fe, tau_re, se_re, tau2, I2, Q, Q_p = dersimonian_laird(tau_v, se_v)

    # z and p for both models
    z_fe  = tau_fe / se_fe;  p_fe = 2 * stats.norm.sf(abs(z_fe))
    z_re  = tau_re / se_re;  p_re = 2 * stats.norm.sf(abs(z_re))

    n_sig   = (sub['qval'] < ALPHA).sum()
    pct_sig = n_sig / len(sub)

    meta_rows.append({
        'gene': gene,
        # Random-effects (primary)
        'tau_pool': tau_re, 'se_pool': se_re, 'z_pool': z_re, 'pval_pool': p_re,
        # Fixed-effects (secondary / diagnostic)
        'tau_fe': tau_fe, 'se_fe': se_fe, 'z_fe': z_fe, 'pval_fe': p_fe,
        # Heterogeneity
        'tau2': tau2, 'I2': I2, 'Q': Q, 'Q_pval': Q_p,
        # Cross-sample counts
        'n_sig': n_sig, 'pct_sig': pct_sig,
        'n_samples': len(sub),
        'tau_mean': sub['tau'].mean(), 'tau_std': sub['tau'].std(),
    })

meta = pd.DataFrame(meta_rows)

# BH correction on RE p-values (primary)
_, qval_re, _, _ = multipletests(meta['pval_pool'], method='fdr_bh')
meta['qval_pool'] = qval_re
# BH correction on FE p-values (diagnostic)
_, qval_fe, _, _ = multipletests(meta['pval_fe'], method='fdr_bh')
meta['qval_fe'] = qval_fe

meta = meta.sort_values('pval_pool')
meta.to_csv('./results_raw/gbm_meta_results.csv', index=False)

print(f"Random-effects significant (q<0.05):  {(meta['qval_pool'] < ALPHA).sum()}/{n_genes}")
print(f"Fixed-effects significant (q<0.05):   {(meta['qval_fe'] < ALPHA).sum()}/{n_genes}")
meta_known   = meta[meta['gene'].isin(EXPECTED_SIGN)]
dir_correct  = (meta_known['tau_pool'] * meta_known['gene'].map(EXPECTED_SIGN) > 0).sum()
print(f"Direction accuracy (known sign, RE):  {dir_correct}/{len(meta_known)}")
print(f"\nHeterogeneity summary (median I²):    {meta['I2'].median():.2%}")
print(f"Genes with I²>0.75:                   {(meta['I2'] > 0.75).sum()}/{n_genes}")
print(f"\nTop 15 by RE z-score:")
print(meta[['gene','tau_pool','se_pool','z_pool','qval_pool','I2','pct_sig']].head(15).to_string())

# ── Figure 1: Cross-sample τ heatmap ─────────────────────────────────────────
print('\nBuilding cross-sample heatmap...')

# Gene order: by fixed-effects z-score
gene_order = meta.sort_values('z_pool', ascending=False)['gene'].tolist()

tau_mat  = np.full((n_genes, n_samples), np.nan)
sig_mat  = np.zeros((n_genes, n_samples), dtype=bool)
for i, gene in enumerate(gene_order):
    for j, sample in enumerate(SAMPLES):
        row = df[(df['gene'] == gene) & (df['sample'] == sample)]
        if len(row):
            tau_mat[i, j] = row['tau'].values[0]
            sig_mat[i, j] = row['qval'].values[0] < ALPHA

vabs = np.nanpercentile(np.abs(tau_mat), 97)
cmap = plt.cm.RdBu_r

fig, ax = plt.subplots(figsize=(max(10, n_samples * 0.38), max(10, n_genes * 0.22)))
im = ax.imshow(tau_mat, aspect='auto', cmap=cmap, vmin=-vabs, vmax=vabs)

# Asterisks for significant cells
for i in range(n_genes):
    for j in range(n_samples):
        if sig_mat[i, j]:
            ax.text(j, i, '✦', ha='center', va='center', fontsize=5, color='white',
                    fontweight='bold')

ax.set_xticks(range(n_samples))
ax.set_xticklabels(SAMPLES, rotation=55, ha='right', fontsize=7.5)
ax.set_yticks(range(n_genes))

# Gene labels with expected direction marker
ylabels = []
for gene in gene_order:
    exp = EXPECTED_SIGN.get(gene, 0)
    row = meta[meta['gene'] == gene].iloc[0]
    sig = '*' if row['qval_pool'] < ALPHA else ''
    exp_mark = '→LE' if exp > 0 else '→CT' if exp < 0 else ''
    ylabels.append(f"{gene}{sig}  {exp_mark}")
ax.set_yticklabels(ylabels, fontsize=7.5)

cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label('τ (positive = LE-enriched)', fontsize=9)
ax.set_title('GBM PLM results across 26 samples\n'
             '(✦ q<0.05 per sample; gene* = significant in random-effects meta-analysis)',
             fontsize=10)
plt.tight_layout()
plt.savefig('./results_raw/gbm_sample_heatmap.png', dpi=180, bbox_inches='tight')
print('Saved ./results_raw/gbm_sample_heatmap.png')
plt.close()

# ── Figure 2: Pooled effect size lollipop ─────────────────────────────────────
print('Building pooled effect size plot...')
meta_sorted = meta.sort_values('tau_pool')
y_pos = np.arange(len(meta_sorted))

def gene_color_gbm(gene, sig):
    exp = EXPECTED_SIGN.get(gene, 0)
    if not sig:
        return '#BBBBBB'
    if exp == +1:
        return '#E87F5A'  # LE
    if exp == -1:
        return '#5A9ED4'  # CT
    return '#888888'

meta_sorted['sig']   = meta_sorted['qval_pool'] < ALPHA
meta_sorted['color'] = [gene_color_gbm(g, s) for g, s in
                        zip(meta_sorted['gene'], meta_sorted['sig'])]
meta_sorted['ci95']  = 1.96 * meta_sorted['se_pool']

fig, ax = plt.subplots(figsize=(5.5, 12))
for i, row in enumerate(meta_sorted.itertuples()):
    ax.plot([row.tau_pool - row.ci95, row.tau_pool + row.ci95], [i, i],
            color=row.color, linewidth=1.3, alpha=0.8)
    ax.scatter([row.tau_pool], [i], color=row.color,
               s=35 if row.sig else 15, zorder=5,
               marker='D' if row.sig else 'o')

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(y_pos)
ylabels = [
    f"{row.gene}{'*' if row.sig else ''}  {row.pct_sig:.0%}sig"
    for row in meta_sorted.itertuples()
]
ax.set_yticklabels(ylabels, fontsize=7.5)
ax.set_xlabel('Random-effects pooled τ ± 95% CI\n(positive = LE-enriched; DerSimonian-Laird)', fontsize=10)
ax.set_title('GBM cross-sample meta-analysis (n=26)\n'
             '(* q<0.05; orange=expected LE, blue=expected CT, gray=NS)',
             fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_els = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#E87F5A', markersize=8,
           label='Significant, expected LE'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#5A9ED4', markersize=8,
           label='Significant, expected CT'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#BBBBBB', markersize=7,
           label='Not significant'),
]
ax.legend(handles=legend_els, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('./results_raw/gbm_meta_summary.png', dpi=200, bbox_inches='tight')
print('Saved ./results_raw/gbm_meta_summary.png')
plt.close()

# ── Figure 3: Forest plot for key genes ───────────────────────────────────────
print('Building forest plot...')
KEY_GENES = ['snap25', 'syt1', 'tubb3', 'rbfox3',     # LE expected
             'gfap', 's100b', 'aqp4', 'egfr', 'mki67', # CT expected
             'mog']                                      # discordant

key_genes_present = [g for g in KEY_GENES if g in genes]
n_key = len(key_genes_present)
n_cols = 2
n_rows = (n_key + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, n_rows * 2.2), squeeze=False)

for idx, gene in enumerate(key_genes_present):
    row_i, col_i = divmod(idx, n_cols)
    ax = axes[row_i][col_i]

    sub = df[df['gene'] == gene].copy()
    sub = sub.set_index('sample').reindex(SAMPLES).reset_index()
    sub.columns = ['sample'] + list(sub.columns[1:])

    y_pos_f  = np.arange(n_samples)
    exp      = EXPECTED_SIGN.get(gene, 0)
    col_base = '#E87F5A' if exp > 0 else '#5A9ED4' if exp < 0 else '#888888'

    for j, srow in enumerate(sub.itertuples()):
        if np.isnan(srow.tau):
            continue
        ci = 1.96 * srow.se
        col = col_base if srow.qval < ALPHA else '#CCCCCC'
        ax.plot([srow.tau - ci, srow.tau + ci], [j, j], color=col, linewidth=1)
        ax.scatter([srow.tau], [j], color=col, s=15, zorder=5)

    # Pooled estimates — RE (primary) + FE (secondary)
    pmeta = meta[meta['gene'] == gene].iloc[0]
    ax.axvline(pmeta['tau_pool'], color='black', linewidth=1.8, linestyle='-',
               label=f"RE τ={pmeta['tau_pool']:.3f} z={pmeta['z_pool']:.1f} "
                     f"(q={pmeta['qval_pool']:.2e}, I²={pmeta['I2']:.0%})")
    ax.axvline(pmeta['tau_fe'], color='dimgray', linewidth=1.0, linestyle=':',
               label=f"FE τ={pmeta['tau_fe']:.3f} z={pmeta['z_fe']:.1f}")
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')

    ax.set_yticks(y_pos_f)
    ax.set_yticklabels(SAMPLES, fontsize=5.5)
    exp_str = '→LE' if exp > 0 else '→CT' if exp < 0 else 'ambiguous'
    ax.set_title(f'{gene}  [{exp_str}]', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.0, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('τ', fontsize=8)

# Hide unused subplots
for idx in range(n_key, n_rows * n_cols):
    row_i, col_i = divmod(idx, n_cols)
    axes[row_i][col_i].set_visible(False)

plt.suptitle('GBM forest plots — per-sample PLM τ with random-effects pooled estimate\n'
             '(solid black = RE DerSimonian-Laird; dotted = FE)',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig('./results_raw/gbm_meta_forest.png', dpi=180, bbox_inches='tight')
print('Saved ./results_raw/gbm_meta_forest.png')
plt.close()

print('\nDone. Outputs:')
print('  ./results_raw/gbm_meta_results.csv')
print('  ./results_raw/gbm_sample_heatmap.png')
print('  ./results_raw/gbm_meta_summary.png')
print('  ./results_raw/gbm_meta_forest.png')
