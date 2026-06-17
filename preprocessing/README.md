# Preprocessing scripts

These are the scripts we used to turn the raw public datasets into the inputs the
analysis pipeline expects. See [`../DATA.md`](../DATA.md) for the original data
sources. Run all scripts from the repository root.

The mouse-brain (MB) dataset needs no preprocessing — `squidpy.datasets.*`
auto-downloads and caches it on first use.

## `prepare_darmanis_gbm.py`

Downloads the Darmanis et al. GBM scRNA-seq dataset (NCBI GEO **GSE84465**) and
writes the single-cell reference used by the GBM pipeline.

- **Source:** NCBI GEO FTP (URLs hard-coded in the script).
- **Output:** `Inputs/scRNA_Darmanis/darmanis_adata.h5ad` (raw counts; lowercase
  gene names; `obs`: cell_id, tissue, cell_type, patient_id).
- **Dependencies:** `scanpy`, `pandas`, `numpy`, `scipy` (all in the package deps).
- **Run:** `python preprocessing/prepare_darmanis_gbm.py`

## `build_xenium_zarr.py`

Ingests the 10x Genomics human breast-cancer Xenium dataset into a `SpatialData`
zarr store used by the HBC pipeline. (This is the original ingestion script; it
also contains the CellPLM imputation prototype — the maintained imputation entry
point is `scripts/hbc_cellplm_raw.py`.)

- **Source:** 10x Xenium raw output at `./Xenium/outs`; supervised annotation
  spreadsheet under `./Xenium/`.
- **Output:** `./Xenium.zarr` (SpatialData store).
- **Dependencies (beyond the package deps):** `spatialdata`, `squidpy`,
  `hdf5plugin`, `torch`, and [`CellPLM`](https://github.com/OmicsML/CellPLM)
  with its pretrained checkpoint. These are **not** TIDEST package dependencies;
  install them separately to reproduce the HBC ingestion.
- **Run:** `python preprocessing/build_xenium_zarr.py`

> The GBM Visium spatial data (`Inputs/general/GBM_data/<sample>/outs/`) and the
> per-spot annotation table (`Inputs/general/visium_metadata.csv`) are obtained
> directly from the published GBM spatial atlas; there is no conversion script.
> See the *GBM* section of [`../DATA.md`](../DATA.md).
