"""
Supplementary 3×3 figure: performance across all three DGP variants.

Rows:    two-region | GRF | latent-factor
Columns: TPR        | FPR | AUC

Each cell shows all 5 methods as a function of confounder strength α.
Same style (colors, line widths, fonts) as the main sim_figure.py.

Saves:
  results_raw/sim_dgp_figure.pdf
  results_raw/sim_dgp_figure.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_SIM_DIR, '..', '..')

# ── Same style as figures.py ───────────────────────────────────────────────────
COL = {
    'tidest':     '#EE6677',
    'ttest':      '#4477AA',
    'spagcn':     '#228833',
    'despace':    '#CCBB44',
    'spatialgee': '#AA3377',
}
LABELS = {
    'tidest':     'TIDEST',
    'ttest':      '$t$-test',
    'spagcn':     'SpaGCN',
    'despace':    'DESpace',
    'spatialgee': 'SpatialGEE',
}
METHODS = ['tidest', 'ttest', 'spagcn', 'despace', 'spatialgee']

DGP_VARIANTS = ['two-region', 'grf', 'latent-factor']
DGP_LABELS   = ['Two-region', 'GRF', 'Latent-factor']

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

METRIC_TITLES = {
    'tpr': 'Power (TPR)',
    'fpr': 'FPR (type-I error)',
    'auc': 'AUC',
}

# Per (dgp_variant, metric) y-limits — rescaled where data range is narrow
YLIM = {
    ('two-region',    'tpr'): (0.83, 1.01),
    ('grf',           'tpr'): (0.55, 1.02),
    ('latent-factor', 'tpr'): (0.92, 1.01),
    ('two-region',    'fpr'): (0.0,  0.62),
    ('grf',           'fpr'): (0.0,  0.62),
    ('latent-factor', 'fpr'): (0.0,  0.12),
    ('two-region',    'auc'): (0.78, 1.01),
    ('grf',           'auc'): (0.78, 1.01),
    ('latent-factor', 'auc'): (0.975, 1.005),
}


def _agg(df, group_col, value_col, band='se'):
    g = df.groupby(group_col)[value_col].agg(['mean', 'sem', 'std']).reset_index()
    g = g.sort_values(group_col)
    ci = 1.96 * g['sem'] if band == 'se' else g['std']
    return g[group_col].values, g['mean'].values, ci.values


def _line(ax, x, y, err, color, label):
    ax.plot(x, y, color=color, label=label, ls='-',
            marker='o', ms=4.5, lw=1.8, zorder=3)
    ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, zorder=2)
    ax.spines[['top', 'right']].set_visible(False)


def _fill_subplot(ax, sub, variant, metric, show_xlabel=True):
    col_map = {'tpr': 'tpr_{m}', 'fpr': 'fpr_{m}', 'auc': 'auc_{m}'}

    for key in METHODS:
        col = col_map[metric].format(m=key)
        if col not in sub.columns or sub[col].isna().all():
            continue
        x, y, err = _agg(sub, 'alpha_conf', col)
        _line(ax, x, y, err, COL[key], LABELS[key])

    if metric == 'fpr':
        ax.axhline(0.05, color='gray', ls='--', lw=1, zorder=1)

    ax.set_ylim(*YLIM[(variant, metric)])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    if show_xlabel:
        ax.set_xlabel('Confounder strength α')
    else:
        ax.set_xticklabels([])


def main():
    csv_main = os.path.join(_ROOT, 'results_raw', 'sim_results.csv')
    df = pd.read_csv(csv_main)

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.38, wspace=0.32,
                            top=0.93, bottom=0.13,
                            left=0.10, right=0.97)

    metrics = ['tpr', 'fpr', 'auc']
    first_ax = None

    for row_i, (variant, dgp_label) in enumerate(zip(DGP_VARIANTS, DGP_LABELS)):
        sub = df[
            (df['n_spots']    == 300) &
            (df['dgp_variant']== variant) &
            (df['tau_level']  == 1.0) &
            (df['sigma_imp']  == 0.5)
        ].copy()

        for col_i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[row_i, col_i])
            if first_ax is None:
                first_ax = ax

            _fill_subplot(ax, sub, variant, metric, show_xlabel=(row_i == 2))

            # Column titles on top row only
            if row_i == 0:
                ax.set_title(METRIC_TITLES[metric], fontweight='bold', pad=8)

            # Row labels: rotated, just outside the left edge of the leftmost column
            if col_i == 0:
                ax.text(-0.18, 0.5, dgp_label,
                        transform=ax.transAxes,
                        fontweight='bold', fontsize=FONT,
                        rotation=90, va='center', ha='center')

    # Shared legend at the bottom of the figure
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.01),
               ncol=5,
               framealpha=0.9,
               fontsize=FONT)

    out = os.path.join(_ROOT, 'results_raw', 'sim_dgp_figure')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'Saved {out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
