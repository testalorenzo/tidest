# Reproducing the TIDEST paper

Every analysis can be regenerated with the [`Makefile`](Makefile).
Run all targets **from the repository root** — the scripts use relative paths
(`./Inputs`, `./data`, `./Xenium*`) and write intermediates to `./results_raw/`.

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
make mb        # = mb-tangram → mb-plm
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `mb-tangram` | `scripts/mb_tangram_raw.py` | `results_raw/tangram_pred*.pkl` | Tangram mapping |
| `mb-plm` | `scripts/mb_plm.py` | `results_raw/plm_results*.csv` | **41/55** sig (q<0.05); **28/38** correct direction |

## 2. GBM

```bash
make gbm-prep  # one-time: download Darmanis scRNA-seq (GSE84465)
make gbm       # = gbm-tangram → gbm-plm → gbm-comparison
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `gbm-tangram` | `scripts/gbm_tangram.py` | `results_raw/gbm_tangram_pred_<sample>_*.pkl` | per-sample Tangram (26 samples) |
| `gbm-plm` | `scripts/gbm_plm.py` | per-sample CSVs + `gbm_plm_results_all.csv` | **48/90** sig in meta-analysis; **26/81** correct direction |
| `gbm-comparison` | `scripts/gbm_method_comparison_full.py` | comparison CSVs | cross-method comparison (Table D.3) |

## 3. HBC (Xenium breast cancer)

```bash
make hbc-zarr     # one-time ingestion: raw Xenium → Xenium.zarr (CellPLM env)
make hbc          # = hbc-cellplm → hbc-plm
```

| Step | Script | Output | Expect |
|---|---|---|---|
| `hbc-cellplm` | `scripts/hbc_cellplm_raw.py` | `results_raw/hbc_cellplm_pred*.pkl` | CellPLM imputation |
| `hbc-plm` | `scripts/hbc_plm.py` | `results_raw/hbc_plm_results*.csv` | **65/68** sig (q<0.05); **20/25** correct directional in-panel |

## 4. Simulation study (no downloads)

```bash
make sim             # full grid (5 methods × parameter settings); needs R for competitors
make sim-no-r        # full grid, TIDEST + t-test only (no R)
make sim-sensitivity # hyperparameter sensitivity sweep
```

| Step | Script | Output |
|---|---|---|
| `sim` | `scripts/simulation/run_sim.py --full-grid` | `results_raw/sim_results.csv` |
| `sim-sensitivity` | `scripts/simulation/run_sim_sensitivity.py` | sensitivity sweep CSV |

Headline (α = 1.0, two-region DGP): TIDEST holds FPR at 13.7% vs 42.8-50.0% for
competitors, with AUC = 0.932 vs ≤ 0.832; augmentation cuts reconstruction RMSE
by up to 72% at high imputation noise.

## 5. Everything

```bash
make all        # mb + gbm + hbc + sim (heavy; all data required)
```

> Wall-clock note: MB runs in minutes; the simulation grid and the 26-sample GBM
> Tangram/PLM loop and the HBC CellPLM step are the heavy steps (hours, GPU
> recommended for CellPLM).
