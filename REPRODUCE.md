# Reproducing the TIDEST paper

Every analysis and figure can be regenerated with the [`Makefile`](Makefile).
Run all targets **from the repository root** — the scripts use relative paths
(`./Inputs`, `./data`, `./Xenium*`) and write intermediates and figures to
`./results_raw/`.

## 0. Setup

```bash
conda activate spatrans            # Python 3.11 environment used for the paper
make install                       # pip install -e tidest
make vignette                      # sanity check: synthetic example, no data needed
make test                          # smoke test
```

Some steps additionally require:

- **R + SpatialPCA** for confounder estimation in the real-data pipelines, and
  **R + DESpace + SpatialGEE** for the simulation competitors
  (`make install-r-deps`). The simulation can skip these with `make sim-no-r`.
- **CellPLM** (+ checkpoint), `spatialdata`, `torch` for the HBC pipeline only.

Obtain the raw datasets first — see [`DATA.md`](DATA.md).

## 1. Mouse brain (MB)

```bash
make mb        # = mb-tangram → mb-plm → mb-comparison → mb-figure
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `mb-tangram` | `scripts/mb_tangram_raw.py` | `results_raw/tangram_pred*.pkl` | Tangram mapping |
| `mb-plm` | `scripts/mb_plm.py` | `results_raw/plm_results*.csv` | **41/55** sig (q<0.05); **28/38** correct direction |
| `mb-comparison` | `scripts/mb_method_comparison.py` | `results_raw/mb_method_comparison.csv` | TIDEST 28/38 vs SpatialGEE 24 / t-test 19 / DESpace 19 / SpaGCN 15 |
| `mb-figure` | `scripts/mb_main_figure.py` (**Fig. 3**), `mb_supplementary_figure.py` (**Fig. C.3**) | `results_raw/mb_main_figure.pdf`, `mb_supplementary_figure.pdf` | main + supplementary figures |

## 2. GBM

```bash
make gbm-prep  # one-time: download Darmanis scRNA-seq (GSE84465)
make gbm       # = gbm-tangram → gbm-plm → gbm-meta → gbm-comparison → gbm-figure
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `gbm-tangram` | `scripts/gbm_tangram.py` | `results_raw/gbm_tangram_pred_<sample>_*.pkl` | per-sample Tangram (26 samples) |
| `gbm-plm` | `scripts/gbm_plm.py` | per-sample CSVs + `gbm_plm_results_all.csv` | **48/90** sig in meta-analysis; **26/81** correct direction |
| `gbm-meta` | `scripts/gbm_meta_analysis.py` | `gbm_meta_results.csv`, forest/heatmap | DerSimonian-Laird random-effects meta-analysis |
| `gbm-comparison` | `scripts/gbm_method_comparison_full.py` | comparison CSVs | cross-method comparison (Fig. 4e, Table D.3) |
| `gbm-figure` | `scripts/gbm_main_figure.py` (**Fig. 4**), `gbm_reproducibility_figure.py` (**Fig. D.4**) | `results_raw/gbm_*figure*.pdf` | main + supplementary figures |

## 3. HBC (Xenium breast cancer)

```bash
make hbc-zarr     # one-time ingestion: raw Xenium → Xenium.zarr (CellPLM env)
make hbc          # = hbc-cellplm → hbc-plm → hbc-comparison → hbc-figure
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `hbc-cellplm` | `scripts/hbc_cellplm_raw.py` | `results_raw/hbc_cellplm_pred*.pkl` | CellPLM imputation |
| `hbc-plm` | `scripts/hbc_plm.py` | `results_raw/hbc_plm_results*.csv` | **65/68** sig (q<0.05); **20/25** correct directional in-panel |
| `hbc-comparison` | `scripts/hbc_method_comparison.py` | comparison CSVs | PLM vs t-test/Wilcoxon on observed panel (Fig. 5b, Table E.4) |
| `hbc-figure` | `scripts/hbc_main_figure.py` (**Fig. 5**) | `results_raw/hbc_main_figure.pdf` | 6-panel HBC figure |

## 4. Simulation study (no downloads)

```bash
make sim             # full grid (5 methods × parameter settings); needs R for competitors
make sim-no-r        # full grid, TIDEST + t-test only (no R)
make sim-figure      # Fig. 2 (main) + Fig. B.1 (per-DGP supplementary)
make sim-sensitivity # Fig. B.2 (hyperparameter sensitivity sweep)
```

| Step | Script | Output |
|---|---|---|
| `sim` | `scripts/simulation/run_sim.py --full-grid` | `results_raw/sim_results.csv` |
| `sim-figure` | `scripts/simulation/figures.py` (**Fig. 2**), `sim_dgp_figure.py` (**Fig. B.1**) | `results_raw/sim_figure.pdf`, `sim_dgp_figure.pdf` |
| `sim-sensitivity` | `scripts/simulation/run_sim_sensitivity.py`, `sim_sensitivity_figure.py` (**Fig. B.2**) | sensitivity sweep + figure |

Headline (α = 1.0, two-region DGP): TIDEST holds FPR at 13.7% vs 42.8-50.0% for
competitors, with AUC = 0.932 vs ≤ 0.832; augmentation cuts reconstruction RMSE
by up to 72% at high imputation noise.

## 5. Everything

```bash
make all        # mb + gbm + hbc + sim + figures (heavy; all data required)
make figures    # just regenerate every figure from existing intermediates
```

> Wall-clock note: MB runs in minutes; the simulation grid and the 26-sample GBM
> Tangram/PLM loop and the HBC CellPLM step are the heavy steps (hours, GPU
> recommended for CellPLM).
