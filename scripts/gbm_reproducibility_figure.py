"""
Orthogonal cross-sample reproducibility figure for the GBM application (referee #6).

Unlike the curated marker-direction comparisons used elsewhere, this figure
evaluates TIDEST purely on internal cross-sample consistency: for each of the
90 panel genes, what fraction of the 26 individual-sample PLM estimates agree
in sign with the pooled (random-effects) effect? This is an objective
reproducibility metric that does not depend on any prior expectation about
the gene's direction.

Inputs:
  ./results_raw/gbm_plm_results_all.csv  (sample, gene, tau, se, z, pval, qval)
  ./results_raw/gbm_meta_results.csv     (gene, tau_pool, qval_pool, I2, ...)

Outputs:
  ./results_raw/gbm_reproducibility_figure.pdf
  ./results_raw/gbm_reproducibility_figure.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05

per_sample = pd.read_csv('./results_raw/gbm_plm_results_all.csv')
meta = pd.read_csv('./results_raw/gbm_meta_results.csv').set_index('gene')

records = []
for gene, grp in per_sample.groupby('gene'):
    pooled_sign = np.sign(meta.loc[gene, 'tau_pool'])
    sign_consistency = (np.sign(grp['tau']) == pooled_sign).mean()
    records.append({
        'gene': gene,
        'sign_consistency': sign_consistency,
        'qval_pool': meta.loc[gene, 'qval_pool'],
        'I2': meta.loc[gene, 'I2'],
        'sig': meta.loc[gene, 'qval_pool'] < ALPHA,
    })
res = pd.DataFrame(records)

n_sig = res['sig'].sum()
n_total = len(res)
mean_sig = res.loc[res['sig'], 'sign_consistency'].mean()
mean_nonsig = res.loc[~res['sig'], 'sign_consistency'].mean()
u_stat, u_pval = stats.mannwhitneyu(
    res.loc[res['sig'], 'sign_consistency'],
    res.loc[~res['sig'], 'sign_consistency'],
    alternative='greater',
)

print(f'Genes significant in meta-analysis (q<{ALPHA}): {n_sig}/{n_total}')
print(f'Mean sign-consistency, significant genes:     {mean_sig:.3f}')
print(f'Mean sign-consistency, non-significant genes: {mean_nonsig:.3f}')
print(f'Mann-Whitney U (significant > non-significant): U={u_stat:.1f}, p={u_pval:.2e}')

fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))

# Panel A: sign-consistency vs meta-analysis significance
ax = axes[0]
colors = np.where(res['sig'], '#EE6677', '#BBBBBB')
neglog10q = -np.log10(res['qval_pool'].clip(lower=1e-300))
ax.scatter(neglog10q, res['sign_consistency'], c=colors, s=18, alpha=0.85,
           edgecolors='none')
ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--', label='Chance (0.5)')
ax.axvline(-np.log10(ALPHA), color='gray', linewidth=0.8, linestyle=':', label=r'$q=0.05$')
ax.set_xlabel(r'$-\log_{10}(q_{\mathrm{pool}})$')
ax.set_ylabel('Cross-sample sign consistency')
ax.set_title('A. Sign consistency vs.\nmeta-analysis significance')
ax.legend(fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: sign-consistency distribution, significant vs non-significant genes
ax = axes[1]
groups = [res.loc[~res['sig'], 'sign_consistency'], res.loc[res['sig'], 'sign_consistency']]
bp = ax.boxplot(groups, tick_labels=[f'Not sig.\n(n={n_total - n_sig})', f'Sig. ($q<0.05$)\n(n={n_sig})'],
                 patch_artist=True, widths=0.5, showmeans=True)
for patch, color in zip(bp['boxes'], ['#BBBBBB', '#EE6677']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Cross-sample sign consistency')
ax.set_title(f'B. Significant vs. non-significant genes\n'
              f'(Mann-Whitney $p={u_pval:.1e}$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./results_raw/gbm_reproducibility_figure.pdf', bbox_inches='tight')
plt.savefig('./results_raw/gbm_reproducibility_figure.png', dpi=180, bbox_inches='tight')
print('Saved ./results_raw/gbm_reproducibility_figure.pdf')
