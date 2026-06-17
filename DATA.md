# Data sources & preprocessing

TIDEST's analyses use three public spatial-transcriptomics datasets plus
single-cell references. Raw data are **not** redistributed in this repository. 
This document lists the original public sources and the
preprocessing scripts we used to turn them into the inputs the analysis scripts
expect. All analysis scripts are run from the repository root and read from
`./Inputs/` (human glioblastoma), `./data/` (mouse brain), and `./Xenium*/` 
(human breast cancer).

The fully synthetic simulation study needs **no** downloads — see
[`scripts/simulation/`](scripts/simulation/) and the quick-start in
[`examples/`](examples/).

| Dataset | Role | Source | Preprocessing | Expected output |
|---|---|---|---|---|
| Mouse brain Visium (fluorescence crop) | MB spatial | `squidpy.datasets.visium_fluo_adata_crop()` / `visium_fluo_image_crop()` | none (auto-download + cache) | `data/anndata/visium_fluo_adata_crop.h5ad`, `data/images/visium_fluo_image_crop.tiff` |
| Mouse cortex scRNA-seq | MB reference | `squidpy.datasets.sc_mouse_cortex()` | none (auto-download + cache) | `data/anndata/sc_mouse_cortex.h5ad` |
| Darmanis GBM scRNA-seq (GSE84465) | GBM reference | NCBI GEO accession **GSE84465** | [`preprocessing/prepare_darmanis_gbm.py`](preprocessing/prepare_darmanis_gbm.py) | `Inputs/scRNA_Darmanis/darmanis_adata.h5ad` |
| GBM Visium spatial atlas (26 samples) | GBM spatial | NCBI GEO accession **GSE237183** (MGH / UKF / ZH cohorts; includes the per-spot metadata) | provided metadata; per-sample `outs/` from source | `Inputs/general/GBM_data/<sample>/outs/`, `Inputs/general/visium_metadata.csv` |
| 10x Xenium human breast cancer | HBC spatial | NCBI GEO accession **GSE243280** | [`preprocessing/build_xenium_zarr.py`](preprocessing/build_xenium_zarr.py) | `Xenium.zarr`, `Xenium/` annotations |

See [`preprocessing/README.md`](preprocessing/README.md) for per-script run
instructions and dependencies.

## Mouse brain (MB)

No manual download. The first run of `scripts/mb_tangram_raw.py` calls
`squidpy.datasets.*`, which downloads and caches the Visium fluorescence crop,
its image, and the mouse-cortex single-cell reference. Cached copies already
present in `data/` are reused.

## GBM

**Single-cell reference (Darmanis, GSE84465).**
`preprocessing/prepare_darmanis_gbm.py` downloads the expression matrix and
series matrix from the NCBI GEO FTP server and writes
`Inputs/scRNA_Darmanis/darmanis_adata.h5ad` (raw counts, lowercase gene names,
cell-type / tissue annotations). Run once:

```bash
python preprocessing/prepare_darmanis_gbm.py
```

**Visium spatial (26 samples).**
The samples (prefixes `MGH`, `UKF`, `ZH`; see `Inputs/general/GBM_samples.txt`)
come from the GBM Visium spatial atlas deposited at NCBI GEO under accession
[**GSE237183**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237183).
Each sample is expected as a standard 10x Space Ranger output at
`Inputs/general/GBM_data/<sample>/outs/` (loadable with `scanpy.read_visium`).
The per-spot annotations used as the treatment (`ivygap` region: leading edge
`LE` vs cellular tumor, plus meta-program / CNA columns) are distributed with
the same GEO record and stored here as `Inputs/general/visium_metadata.csv`.

## HBC (Xenium breast cancer)

The human breast-cancer Xenium dataset (NCBI GEO accession
[**GSE243280**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280))
is ingested by `preprocessing/build_xenium_zarr.py`, which reads the raw Xenium
output (`./Xenium/outs`) plus the supervised cell-type annotation spreadsheet
(in `./Xenium/`) and writes a `SpatialData` store at `./Xenium.zarr`. The HBC
pipeline then imputes with CellPLM (`scripts/hbc_cellplm_raw.py`) and runs TIDEST
(`scripts/hbc_plm.py`).

> **Dependencies:** `build_xenium_zarr.py` and `hbc_cellplm_raw.py` require
> `spatialdata`, `hdf5plugin`, `torch`, and [`CellPLM`](https://github.com/OmicsML/CellPLM)
> (with its pretrained checkpoint). See `preprocessing/README.md`.
