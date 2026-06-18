<p align="center">
  <img src="assets/tidest_logo.png" alt="TIDEST logo" width="200"/>
</p>

<h1 align="center">TIDEST</h1>

<p align="center"><b>Testing Imputed Differential Expressions for Spatial Transcriptomics</b></p>

TIDEST estimates the causal effect of a binary spatial treatment (cortical layer,
tumor region, histological compartment, ...) on gene expression. It builds a
Pearson-corrected *augmented outcome* from imputed expression, controls for
spatial structure with SpatialPCA components and a library-size term, and fits a
cross-fitted Robinson (1988) partially linear model to return an effect estimate
`tau`, a sandwich standard error, and BH-adjusted q-values per gene.

This repository contains the `tidest` Python package and everything needed to
reproduce the paper's analyses, simulations, and figures.

## Install

```bash
pip install "git+https://github.com/testalorenzo/tidest.git#subdirectory=tidest"
```

Or, for development:

```bash
git clone https://github.com/testalorenzo/tidest.git
cd tidest
make install          # pip install -e tidest
```

The core method needs only the standard scientific Python stack (numpy, pandas,
scipy, scikit-learn, statsmodels, anndata, scanpy). **R + SpatialPCA** is needed
only when TIDEST estimates spatial confounders itself; pass `U=` to bypass it.

## 30-second quick start (synthetic, no downloads)

```bash
python examples/quickstart.py
```

```python
from tidest import tidest
from _synthetic import make_dataset          # in examples/

data = make_dataset(N=400, G_DE=30, G_null=70)
model = tidest(n_pcs=20, n_folds=2).fit(
    sc_adata=data["sc_adata"],
    st_adata=data["st_adata"],
    pred_adata=data["pred_adata"],
    U=data["U"],                # precomputed spatial PCs → no R needed
    treatment=data["A"],
    genes=data["genes"],
    pred_is_log=True,
)
print(model.results_)           # gene, tau, se, z, pval, qval
```

On this synthetic example TIDEST recovers 90% of the DE genes at q<0.05 with a
1% false-positive rate and the correct effect sign on every detected gene —
versus a 70% false-positive rate for a naive t-test under the same spatial
confounding. See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for an
annotated, figure-by-figure walk-through (its outputs are rendered on GitHub, so
you can read it without running anything).

## Repository layout

| Path | Contents |
|---|---|
| [`tidest/`](tidest/) | the installable `tidest` package (+ its own README) |
| [`examples/`](examples/) | self-contained synthetic quick-start (script + notebook) |
| [`scripts/`](scripts/) | analysis pipelines (MB, GBM, HBC) and figure scripts |
| [`scripts/simulation/`](scripts/simulation/) | the simulation study |
| [`preprocessing/`](preprocessing/) | scripts that build the raw-data inputs |
| [`tests/`](tests/) | smoke test |
| [`Makefile`](Makefile) | one entry point for every analysis and figure |

## Reproduce the paper

- **Data:** [`DATA.md`](DATA.md) — original public sources and preprocessing.
- **Analyses & figures:** [`REPRODUCE.md`](REPRODUCE.md) — run order, expected
  outputs, and headline numbers, all driven by the [`Makefile`](Makefile).

```bash
make vignette     # synthetic sanity check (no data)
make mb           # mouse-brain pipeline + figures
make gbm          # GBM pipeline + figures
make hbc          # HBC (Xenium) pipeline + figures
make sim          # simulation study
make figures      # regenerate every figure
```

## Citation

If you use TIDEST, please cite the accompanying paper (citation to be added).

## License

MIT — see [LICENSE](LICENSE).
