"""
Supplementary stress-tests figure (referee #3).

Reads results_raw/sim_stress_tests_raw.csv (from run_sim_stress_tests.py) and
plots, for each of the 4 stress tests (one per column):
  Row 1: Power (TPR), TIDEST and t-test
  Row 2: FPR (type-I error), TIDEST and t-test, with 5% nominal level marked
  Row 3: AUC, TIDEST and t-test
  Row 4: Reconstruction RMSE -- raw imputation (grey) vs augmented
         pseudo-outcome (red)

Columns:
  A. Weak gene-gene correlation (corr_strength)
  B. Reference mismatch (mismatch_frac)
  C. Non-smooth confounding (confounder_type)
  D. Small SC reference (n_cells)

Same color/style conventions as figures.py / sim_sensitivity_figure.py.

Saves:
  results_raw/sim_stress_tests_figure.pdf
  results_raw/sim_stress_tests_figure.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_SIM_DIR, '..', '..')

COL = {
    'tidest': '#EE6677',
    'ttest':  '#4477AA',
    'pred':   '#BBBBBB',
    'pseudo': '#EE6677',
}
LABELS = {
    'tidest': 'TIDEST',
    'ttest':  '$t$-test',
    'pred':   'Raw imputation',
    'pseudo': 'Augmented (TIDEST)',
}

FONT = 13
plt.rcParams.update({
    'font.size':        FONT,
    'axes.labelsize':   FONT,
    'axes.titlesize':   FONT,
    'legend.fontsize':  FONT - 2,
    'xtick.labelsize':  FONT - 1,
    'ytick.labelsize':  FONT - 1,
    'figure.dpi':       150,
})

STRESS_TESTS = ['A_corr_strength', 'B_mismatch_frac', 'C_confounder_type', 'D_sc_size']
COL_TITLES = {
    'A_corr_strength':   'A. Weak gene-gene\ncorrelation',
    'B_mismatch_frac':   'B. Reference\nmismatch',
    'C_confounder_type': 'C. Non-smooth\nconfounding',
    'D_sc_size':         'D. Small SC\nreference',
}
XLABELS = {
    'A_corr_strength':   'corr\\_strength',
    'B_mismatch_frac':   'mismatch\\_frac',
    'C_confounder_type': 'confounder type',
    'D_sc_size':         '# SC reference cells',
}
# Categorical x labels for stress test C (preserve sweep order)
CAT_LABELS = {
    'C_confounder_type': ['smooth', 'discont.', 'hotspot', 'multiscale'],
}

ROWS = [
    ('tpr',  'Power (TPR)'),
    ('fpr',  'FPR (type-I error)'),
    ('auc',  'AUC'),
    ('rmse', 'Reconstruction RMSE'),
]


def _agg(df, value_col):
    g = df.groupby('value', sort=False, observed=True)[value_col].agg(['mean', 'sem'])
    ci = 1.96 * g['sem']
    return g.index.values, g['mean'].values, ci.values


def _xpos(stress, values):
    """Return x positions: numeric values plotted as-is, categorical as 0..k-1."""
    if stress == 'C_confounder_type':
        return np.arange(len(values))
    if stress == 'D_sc_size':
        return np.arange(len(values))  # evenly spaced despite 2000/500/100
    return np.array(values, dtype=float)


def main():
    df = pd.read_csv(os.path.join(_ROOT, 'results_raw', 'sim_stress_tests_raw.csv'))

    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(len(ROWS), len(STRESS_TESTS), figure=fig,
                            hspace=0.45, wspace=0.32,
                            top=0.93, bottom=0.06, left=0.07, right=0.98)

    for col_i, stress in enumerate(STRESS_TESTS):
        sub = df[df['stress'] == stress]
        # preserve the sweep order as it appears in the CSV
        order = sub['value'].drop_duplicates().tolist()
        sub = sub.copy()
        sub['value'] = pd.Categorical(sub['value'], categories=order, ordered=True)

        for row_i, (metric, title) in enumerate(ROWS):
            ax = fig.add_subplot(gs[row_i, col_i])

            if metric == 'rmse':
                vals, y_pred, ci_pred = _agg(sub, 'rmse_pred')
                _, y_pseudo, ci_pseudo = _agg(sub, 'rmse_pseudo')
                x = _xpos(stress, vals)
                for y, ci, key in [(y_pred, ci_pred, 'pred'), (y_pseudo, ci_pseudo, 'pseudo')]:
                    ax.plot(x, y, color=COL[key], ls='-', marker='o', ms=4.5,
                            lw=1.8, zorder=3, label=LABELS[key])
                    ax.fill_between(x, y - ci, y + ci, color=COL[key], alpha=0.15, zorder=2)
            else:
                vals, y_t, ci_t = _agg(sub, f'{metric}_tidest')
                _, y_c, ci_c = _agg(sub, f'{metric}_ttest')
                x = _xpos(stress, vals)
                for y, ci, key in [(y_t, ci_t, 'tidest'), (y_c, ci_c, 'ttest')]:
                    ax.plot(x, y, color=COL[key], ls='-', marker='o', ms=4.5,
                            lw=1.8, zorder=3, label=LABELS[key])
                    ax.fill_between(x, y - ci, y + ci, color=COL[key], alpha=0.15, zorder=2)
                if metric == 'fpr':
                    ax.axhline(0.05, color='gray', ls='--', lw=1, zorder=1)

            ax.spines[['top', 'right']].set_visible(False)

            if stress in CAT_LABELS or stress == 'D_sc_size':
                ax.set_xticks(_xpos(stress, vals))
                if stress == 'D_sc_size':
                    ax.set_xticklabels([str(int(v)) for v in vals])
                else:
                    ax.set_xticklabels(CAT_LABELS[stress], rotation=20, ha='right')

            if row_i == 0:
                ax.set_title(COL_TITLES[stress], fontweight='bold', pad=8)
                ax.legend(frameon=False, fontsize=FONT - 3, loc='best')
            if metric == 'rmse' and col_i == 0:
                ax.legend(frameon=False, fontsize=FONT - 3, loc='best')
            if col_i == 0:
                ax.text(-0.32, 0.5, title, transform=ax.transAxes, fontweight='bold',
                        fontsize=FONT, rotation=90, va='center', ha='center')
            if row_i == len(ROWS) - 1:
                ax.set_xlabel(XLABELS[stress])

    out = os.path.join(_ROOT, 'results_raw', 'sim_stress_tests_figure')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'Saved {out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
