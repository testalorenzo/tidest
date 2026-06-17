"""
Calibration diagnostics figure (referees #4, #5).

Reads results_raw/sim_calibration_raw.csv and sim_calibration_pvals.csv (from
run_calibration.py) and plots, for the main figure's high-confounding setting
(two-region DGP, alpha_conf=1.0, sigma_imp=0.5, n=300, 200 replicates):

  Panel A: null p-value histograms (TIDEST, t-test, PPI)
  Panel B: QQ plot of null p-values, -log10(observed) vs -log10(expected),
           all six methods
  Panel C: empirical 95% Wald CI coverage for null genes (should contain 0)
           and DE genes (should contain tau_true), all six methods

Same color/style conventions as figures.py.

Saves:
  results_raw/sim_calibration_figure.pdf
  results_raw/sim_calibration_figure.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_SIM_DIR, '..', '..')

FONT = 13
plt.rcParams.update({
    'font.size':        FONT,
    'axes.labelsize':   FONT,
    'axes.titlesize':   FONT,
    'legend.fontsize':  FONT - 1,
    'xtick.labelsize':  FONT - 1,
    'ytick.labelsize':  FONT - 1,
    'figure.dpi':       150,
})

COL = {
    'tidest':     '#EE6677',   # red
    'ttest':      '#4477AA',   # blue
    'spagcn':     '#228833',   # green
    'despace':    '#CCBB44',   # yellow-gold
    'spatialgee': '#AA3377',   # purple
    'ppi':        '#66CCEE',   # cyan
}
LABELS = {
    'tidest':     'TIDEST',
    'ttest':      't-test',
    'spagcn':     'SpaGCN (Wilcoxon)',
    'despace':    'DESpace',
    'spatialgee': 'SpatialGEE',
    'ppi':        'PPI',
}
METHOD_ORDER = ['tidest', 'ttest', 'spagcn', 'despace', 'spatialgee', 'ppi']


def _qq_points(pvals, n_points=2000):
    """Subsample (log-spaced in rank) -log10(observed) vs -log10(expected)."""
    p = np.sort(np.asarray(pvals))
    n = len(p)
    # log-spaced ranks emphasize the tail (small p-values)
    ranks = np.unique(np.geomspace(1, n, n_points).astype(int)) - 1
    expected = (ranks + 1) / (n + 1)
    observed = p[ranks]
    return -np.log10(expected), -np.log10(np.clip(observed, 1e-300, 1.0))


def main():
    raw = pd.read_csv(os.path.join(_ROOT, 'results_raw', 'sim_calibration_raw.csv'))
    pvals = pd.read_csv(os.path.join(_ROOT, 'results_raw', 'sim_calibration_pvals.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))

    # ── Panel A: null p-value histograms ────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    for method in ['ttest', 'ppi', 'tidest']:
        p = pvals.loc[pvals['method'] == method, 'pval'].values
        ax.hist(p, bins=bins, density=True, histtype='step', lw=2.2,
                color=COL[method], label=LABELS[method])
    ax.axhline(1.0, color='gray', ls='--', lw=1, zorder=1, label='Uniform (well-calibrated)')
    ax.set_xlabel('Null-gene $p$-value')
    ax.set_ylabel('Density')
    ax.set_title('A. Null $p$-value distribution', fontweight='bold', loc='left')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel B: QQ plot ─────────────────────────────────────────────────────
    # y-axis capped at 25: TIDEST/PPI/t-test/SpatialGEE stay within this range,
    # while DESpace and SpaGCN (severely anti-conservative) run off the top of
    # the panel -- itself a visual indicator of their miscalibration.
    ax = axes[1]
    xlim, ylim = 5, 25
    for method in METHOD_ORDER:
        p = pvals.loc[pvals['method'] == method, 'pval'].values
        x, y = _qq_points(p)
        ax.plot(x, y, color=COL[method], lw=1.8, label=LABELS[method])
    ax.plot([0, xlim], [0, xlim], color='black', ls=':', lw=1, zorder=1)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xlabel(r'Expected $-\log_{10}(p)$')
    ax.set_ylabel(r'Observed $-\log_{10}(p)$')
    ax.set_title('B. QQ plot (null genes)', fontweight='bold', loc='left')
    ax.legend(frameon=False, fontsize=FONT - 3, ncol=2, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel C: CI coverage ─────────────────────────────────────────────────
    ax = axes[2]
    g = raw.groupby('method')[['coverage_null', 'coverage_de']].agg(['mean', 'sem'])
    width = 0.38
    x = np.arange(len(METHOD_ORDER))
    for offset, col, label, hatch in [(-width / 2, 'coverage_null', 'Null genes (covers 0)', None),
                                       (width / 2, 'coverage_de', r'DE genes (covers $\tau_{\rm true}$)', '//')]:
        means = [g.loc[m, (col, 'mean')] for m in METHOD_ORDER]
        errs  = [1.96 * g.loc[m, (col, 'sem')] for m in METHOD_ORDER]
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=3,
                       color=[COL[m] for m in METHOD_ORDER], hatch=hatch,
                       edgecolor='black', linewidth=0.6, label=label)
    ax.axhline(0.95, color='gray', ls='--', lw=1, zorder=1, label='Nominal 95%')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=30, ha='right')
    ax.set_ylabel('Empirical 95% CI coverage')
    ax.set_ylim(0, 1.05)
    ax.set_title('C. CI coverage', fontweight='bold', loc='left')
    ax.legend(frameon=False, loc='lower left', fontsize=FONT - 3)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    out = os.path.join(_ROOT, 'results_raw', 'sim_calibration_figure')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
        print(f'Saved {out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
