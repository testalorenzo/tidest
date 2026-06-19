# TIDEST — reproduction Makefile
#
# Run every target from the repository root (scripts use relative ./Inputs,
# ./data, ./Xenium*, and write to ./results_raw).
#
#   make install      install the tidest package (editable)
#   make vignette     run the self-contained synthetic quick-start
#   make test         run the smoke test
#   make mb|gbm|hbc   run a full real-data pipeline
#   make sim          run the full simulation grid
#   make all          everything (heavy; needs all data downloaded)
#
# See DATA.md for how to obtain the raw data and REPRODUCE.md for run order,
# expected outputs, and headline numbers.

PYTHON ?= python

.PHONY: all install test vignette clean \
        mb mb-tangram mb-plm \
        gbm gbm-prep gbm-tangram gbm-plm gbm-comparison \
        hbc hbc-zarr hbc-cellplm hbc-plm \
        sim sim-quick sim-no-r sim-sensitivity \
        install-r-deps

# ── Package & example ───────────────────────────────────────────────────────
install:
	pip install -e tidest

vignette:
	$(PYTHON) examples/quickstart.py

test:
	$(PYTHON) tests/test_smoke.py

# ── Mouse brain (MB) ────────────────────────────────────────────────────────
mb: mb-tangram mb-plm

mb-tangram:
	$(PYTHON) scripts/mb_tangram_raw.py

mb-plm:
	$(PYTHON) scripts/mb_plm.py

# ── GBM ─────────────────────────────────────────────────────────────────────
gbm: gbm-tangram gbm-plm gbm-comparison

gbm-prep:
	$(PYTHON) preprocessing/prepare_darmanis_gbm.py

gbm-tangram:
	$(PYTHON) scripts/gbm_tangram.py

gbm-plm:
	$(PYTHON) scripts/gbm_plm.py

gbm-comparison:
	$(PYTHON) scripts/gbm_method_comparison_full.py

# ── HBC (Xenium breast cancer) ──────────────────────────────────────────────
hbc: hbc-cellplm hbc-plm

# Build the SpatialData zarr from raw Xenium output (needs CellPLM env; see DATA.md)
hbc-zarr:
	$(PYTHON) preprocessing/build_xenium_zarr.py

hbc-cellplm:
	$(PYTHON) scripts/hbc_cellplm_raw.py

hbc-plm:
	$(PYTHON) scripts/hbc_plm.py

# ── Simulation study ────────────────────────────────────────────────────────
sim:
	$(PYTHON) scripts/simulation/run_sim.py --full-grid

sim-quick:
	$(PYTHON) scripts/simulation/run_sim.py --quick

# Skip the R competitors (DESpace / SpatialGEE) for environments without R
sim-no-r:
	$(PYTHON) scripts/simulation/run_sim.py --full-grid --skip-r-methods

# Hyperparameter sensitivity sweep
sim-sensitivity:
	$(PYTHON) scripts/simulation/run_sim_sensitivity.py

# ── Aggregate ───────────────────────────────────────────────────────────────
all: mb gbm hbc sim

clean:
	rm -rf tidest_tmp/*.feather

# ── R dependencies for the simulation competitors (one-time) ────────────────
install-r-deps:
	Rscript scripts/simulation/install_r_deps.R
