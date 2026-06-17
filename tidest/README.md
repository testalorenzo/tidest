# TIDEST

**Testing Imputed Differential Expressions for Spatial Transcriptomics**

TIDEST estimates the causal effect of a binary spatial treatment (e.g. cortical
layer identity, tumour region, histological compartment) on gene expression. It
combines three ideas:

1. **Augmented outcome** — imputed expression (from Tangram, CellPLM, …) is
   corrected toward the observed counts using a single-cell gene-gene Pearson
   matrix, yielding a denoised per-spot outcome.
2. **Spatial confounder control** — spatial structure is summarised by SpatialPCA
   components (or any user-supplied confounder matrix) and a per-spot library-size
   term.
3. **Robinson (1988) partially linear model** — cross-fitted random forests
   residualise both treatment and outcome on the confounders; the treatment
   effect `tau` and a sandwich standard error follow, with BH-adjusted q-values.

## Installation

```bash
pip install "git+https://github.com/testalorenzo/tidest.git#subdirectory=tidest"
```

For local development:

```bash
git clone https://github.com/testalorenzo/tidest.git
pip install -e tidest
```

**Optional R dependency.** SpatialPCA-based confounder estimation calls `Rscript`
with the R package [`SpatialPCA`](https://github.com/shangll123/SpatialPCA). This
is only needed when you let TIDEST estimate the confounders itself. If you pass a
precomputed confounder matrix via `U=`, **no R install is required**.

## Quick start

```python
from tidest import tidest

model = tidest(n_pcs=20, n_folds=2, corr_threshold=0.5, seed=0)
model.fit(
    sc_adata=sc_adata,        # single-cell reference (raw counts)
    st_adata=st_adata,        # spatial counts + .obsm['spatial']
    pred_adata=pred_adata,    # imputed expression (Tangram/CellPLM)
    treatment="region",       # obs column ...
    treatment_val="tumor",    # ... value encoding A = 1
    genes=marker_genes,       # genes to test
)
print(model.results_)         # gene, tau, se, z, pval, qval
```

A runnable, self-contained synthetic example (no downloads, no R) lives in
[`examples/quickstart.py`](../examples/quickstart.py) and
[`examples/quickstart.ipynb`](../examples/quickstart.ipynb) in the repository.

## Key parameters

`tidest(...)` constructor:

| Parameter | Default | Meaning |
|---|---|---|
| `n_pcs` | 10 | SpatialPCA components used as confounders |
| `n_folds` | 2 | Cross-fitting folds for the Robinson PLM |
| `n_estimators` | 200 | Trees per random-forest nuisance model |
| `corr_threshold` | 0.5 | Drop PCs with `|corr(PC, A)| ≥ this` (prevents over-control) |
| `cv_threshold` | 0.3 | Max CV of log-expression for a confounder-gene candidate |
| `marker_r_thresh` | 0.3 | Max `|Pearson r|` to any outcome gene for a confounder candidate |
| `n_confounder_genes` | 2000 | Top spatial-variance genes passed to SpatialPCA |
| `top_k` | 5 | Pearson neighbours used in the augmented-outcome correction |
| `spatialPCA_rscript` | bundled | Path to a custom SpatialPCA R script |
| `tmp_dir` | `./tidest_tmp` | Feather exchange directory for the R subprocess |
| `seed` | 42 | Random seed |

`fit(...)` shortcuts for precomputed intermediates:

- `pseudo=` — supply a pre-built augmented-outcome AnnData (skips construction).
- `U=` — supply a confounder matrix (ndarray or DataFrame; skips SpatialPCA / R).
- `pearson=`, `pearson_genes=` — supply a precomputed Pearson matrix.
- `pred_is_log=True` — `pred_adata.X` is already log-transformed (e.g. CellPLM).
- `treatment=` may be an array instead of a column name (then `treatment_val` is
  unnecessary).

Results are in `model.results_` (a DataFrame sorted by p-value).

## Citation

If you use TIDEST, please cite the accompanying paper (citation to be added).

## License

MIT — see [LICENSE](LICENSE).
