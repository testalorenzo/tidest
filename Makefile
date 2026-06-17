# TIDEST — reproduction Makefile
#
# Run every target from the repository root (scripts use relative ./Inputs,
# ./data, ./Xenium*, and write to ./results_raw).
#
#   make install      install the tidest package (editable)
#   make vignette     run the self-contained synthetic quick-start
#   make test         run the smoke test
#   make mb|gbm|hbc   run a full real-data pipeline (analysis + figures)
#   make sim          run the full simulation grid
#   make figures      regenerate every paper figure
#   make all          everything (heavy; needs all data downloaded)
#
# See DATA.md for how to obtain the raw data and REPRODUCE.md for run order,
# expected outputs, and headline numbers.

PYTHON ?= python

.PHONY: all install test vignette clean \
        mb mb-tangram mb-plm mb-comparison mb-figure \
        gbm gbm-prep gbm-tangram gbm-plm gbm-meta gbm-comparison gbm-figure \
        hbc hbc-zarr hbc-cellplm hbc-plm hbc-comparison hbc-figure \
        sim sim-quick sim-no-r sim-figure \
        sim-calibration sim-sensitivity sim-stress figures

# ── Package & example ───────────────────────────────────────────────────────
install:
	pip install -e tidest

vignette:
	$(PYTHON) examples/quickstart.py

test:
	$(PYTHON) tests/test_smoke.py

# ── Mouse brain (MB) ────────────────────────────────────────────────────────
mb: mb-tangram mb-plm mb-comparison mb-figure

mb-tangram:
	$(PYTHON) scripts/mb_tangram_raw.py

mb-plm:
	$(PYTHON) scripts/mb_plm.py

mb-comparison:
	$(PYTHON) scripts/mb_method_comparison.py
	$(PYTHON) scripts/mb_competitor_ablation.py

mb-figure:
	$(PYTHON) scripts/mb_main_figure.py
	$(PYTHON) scripts/mb_supplementary_figure.py

# ── GBM ─────────────────────────────────────────────────────────────────────
gbm: gbm-tangram gbm-plm gbm-meta gbm-comparison gbm-figure

gbm-prep:
	$(PYTHON) preprocessing/prepare_darmanis_gbm.py

gbm-tangram:
	$(PYTHON) scripts/gbm_tangram.py

gbm-plm:
	$(PYTHON) scripts/gbm_plm.py

gbm-meta:
	$(PYTHON) scripts/gbm_meta_analysis.py

gbm-comparison:
	$(PYTHON) scripts/gbm_method_comparison_full.py

gbm-figure:
	$(PYTHON) scripts/gbm_main_figure.py
	$(PYTHON) scripts/gbm_samples_figure.py
	$(PYTHON) scripts/gbm_reproducibility_figure.py

# ── HBC (Xenium breast cancer) ──────────────────────────────────────────────
hbc: hbc-cellplm hbc-plm hbc-comparison hbc-figure

# Build the SpatialData zarr from raw Xenium output (needs CellPLM env; see DATA.md)
hbc-zarr:
	$(PYTHON) preprocessing/build_xenium_zarr.py

hbc-cellplm:
	$(PYTHON) scripts/hbc_cellplm_raw.py

hbc-plm:
	$(PYTHON) scripts/hbc_plm.py

hbc-comparison:
	$(PYTHON) scripts/hbc_method_comparison.py

hbc-figure:
	$(PYTHON) scripts/hbc_main_figure.py

# ── Simulation study ────────────────────────────────────────────────────────
sim:
	$(PYTHON) scripts/simulation/run_sim.py --full-grid

sim-quick:
	$(PYTHON) scripts/simulation/run_sim.py --quick

# Skip the R competitors (DESpace / SpatialGEE) for environments without R
sim-no-r:
	$(PYTHON) scripts/simulation/run_sim.py --full-grid --skip-r-methods

sim-calibration:
	$(PYTHON) scripts/simulation/run_calibration.py
	$(PYTHON) scripts/simulation/sim_calibration_figure.py

sim-sensitivity:
	$(PYTHON) scripts/simulation/run_sim_sensitivity.py
	$(PYTHON) scripts/simulation/sim_sensitivity_figure.py

sim-stress:
	$(PYTHON) scripts/simulation/run_sim_stress_tests.py
	$(PYTHON) scripts/simulation/sim_stress_tests_figure.py

sim-figure:
	$(PYTHON) scripts/simulation/figures.py
	$(PYTHON) scripts/simulation/sim_dgp_figure.py

# ── Aggregate ───────────────────────────────────────────────────────────────
figures: mb-figure gbm-figure hbc-figure sim-figure

all: mb gbm hbc sim sim-figure

clean:
	rm -rf tidest_tmp/*.feather

# ── R dependencies for the simulation competitors (one-time) ────────────────
install-r-deps:
	Rscript scripts/simulation/install_r_deps.R
