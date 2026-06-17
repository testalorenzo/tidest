"""
Supplementary hyperparameter-sensitivity figure (referees #2, #7).

Reads results_raw/sim_sensitivity_raw.csv (from run_sim_sensitivity.py) and
plots, for each of the 5 swept TIDEST hyperparameters (one per column):
  Row 1: Power (TPR)
  Row 2: FPR (type-I error), with the 5% nominal level marked
  Row 3: AUC
  Row 4: Reconstruction RMSE of the augmented pseudo-outcome (rmse_pseudo)

Same color/style conventions as sim_dgp_figure.py (TIDEST = '#EE6677').

Saves:
  results_raw/sim_sensitivity_figure.pdf
  results_raw/sim_sensitivity_figure.png
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

TIDEST_COLOR = '#EE6677'

FONT = 13
plt.rcParams.update({
    'font.size':        FONT,
    'axes.labelsize':   FONT,
    'axes.titlesize':   FONT,
    'legend.fontsize':  FONT,
    'xtick.labelsize':  FONT - 1,
    'ytick.labelsize':  FONT - 1,
    'figure.dpi':       150,
})

SWEEPS = ['top_k', 'min_corr', 'pearson_noise_sd', 'n_pcs', 'corr_threshold']
SWEEP_TITLES = {
    'top_k':            'Augmentation\ntop-$k$ neighbours',
    'min_corr':         'Augmentation\ncorrelation floor',
    'pearson_noise_sd': 'Noise on SC reference\ncorrelations (SD)',
    'n_pcs':            'SpatialPCA\ncomponents',
    'corr_threshold':   'PC-vs-treatment\nscreen $|\\rho|$ threshold',
}
SWEEP_DEFAULT = {
    'top_k': 5, 'min_corr': 0.0, 'pearson_noise_sd': 0.0,
    'n_pcs': 20, 'corr_threshold': 0.5,
}
ROWS = ['tpr_tidest', 'fpr_tidest', 'auc_tidest', 'rmse_pseudo']
ROW_TITLES = {
    'tpr_tidest':  'Power (TPR)',
    'fpr_tidest':  'FPR (type-I error)',
    'auc_tidest':  'AUC',
    'rmse_pseudo': 'Pseudo-outcome\nreconstruction RMSE',
}


def _agg(df, value_col):
    g = df.groupby('value')[value_col].agg(['mean', 'sem']).reset_index()
    g = g.sort_values('value')
    ci = 1.96 * g['sem']
    return g['value'].values, g['mean'].values, ci.values


def main():
    df = pd.read_csv(os.path.join(_ROOT, 'results_raw', 'sim_sensitivity_raw.csv'))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(len(ROWS), len(SWEEPS), figure=fig,
                            hspace=0.45, wspace=0.32,
                            top=0.94, bottom=0.06, left=0.07, right=0.98)

    for col_i, sweep in enumerate(SWEEPS):
        sub = df[df['sweep'] == sweep]
        for row_i, row_key in enumerate(ROWS):
            ax = fig.add_subplot(gs[row_i, col_i])
            x, y, err = _agg(sub, row_key)
            ax.plot(x, y, color=TIDEST_COLOR, ls='-', marker='o', ms=4.5,
                    lw=1.8, zorder=3)
            ax.fill_between(x, y - err, y + err, color=TIDEST_COLOR,
                             alpha=0.15, zorder=2)
            ax.spines[['top', 'right']].set_visible(False)

            if row_key == 'fpr_tidest':
                ax.axhline(0.05, color='gray', ls='--', lw=1, zorder=1)

            ax.axvline(SWEEP_DEFAULT[sweep], color='black', ls=':', lw=1, zorder=1)

            if row_i == 0:
                ax.set_title(SWEEP_TITLES[sweep], fontweight='bold', pad=8)
            if col_i == 0:
                ax.text(-0.32, 0.5, ROW_TITLES[row_key],
                        transform=ax.transAxes, fontweight='bold',
                        fontsize=FONT, rotation=90, va='center', ha='center')
            if row_i == len(ROWS) - 1:
                ax.set_xlabel(sweep)

    fig.text(0.005, 0.98,
             'Dotted vertical line: pipeline default value used in main analyses.',
             fontsize=FONT - 2, style='italic', va='top')

    out = os.path.join(_ROOT, 'results_raw', 'sim_sensitivity_figure')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'Saved {out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
