"""
Download and prepare Darmanis et al. 2017 (GSE84465) scRNA-seq dataset.
Cells from GBM tumor core AND tumor periphery (migrating front), with
normal brain cell types (neurons, oligodendrocytes, astrocytes, OPCs, immune, vascular).

Output: ./Inputs/scRNA_Darmanis/darmanis_adata.h5ad
  - X: raw integer counts (cells x genes)
  - obs: cell_id, tissue (Tumor/Periphery), cell_type, patient_id
  - var: gene names (lowercase)
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

OUT_DIR = './Inputs/scRNA_Darmanis'
os.makedirs(OUT_DIR, exist_ok=True)

EXPR_URL  = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/suppl/GSE84465_GBM_All_data.csv.gz'
MATRIX_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/matrix/GSE84465_series_matrix.txt.gz'

EXPR_LOCAL   = f'{OUT_DIR}/GSE84465_GBM_All_data.csv.gz'
MATRIX_LOCAL = f'{OUT_DIR}/GSE84465_series_matrix.txt.gz'


def download(url, path):
    if os.path.exists(path):
        print(f'  Already exists: {path}')
        return
    print(f'  Downloading {url} ...')
    urllib.request.urlretrieve(url, path)
    print(f'  Saved: {path}')


# ── 1. Download files ────────────────────────────────────────────────────────
print('=== Downloading GSE84465 ===')
download(EXPR_URL,   EXPR_LOCAL)
download(MATRIX_URL, MATRIX_LOCAL)

# ── 2. Parse metadata from series matrix ────────────────────────────────────
print('\n=== Parsing metadata ===')

meta_rows = []
with gzip.open(MATRIX_LOCAL, 'rt') as fh:
    # The series matrix has one column per sample; we parse block by block.
    # Each sample block contributes one row to metadata.
    # Key rows: !Sample_description (2nd occurrence = plateID.well cell ID),
    #           !Sample_characteristics_ch1 (multiple rows per sample)
    header_done = False
    n_samples = 0
    titles = []
    descriptions = []  # list of lists (multiple !Sample_description per sample)
    chars = []         # list of lists of characteristics per sample

    for line in fh:
        line = line.rstrip('\n')
        if line.startswith('!Sample_title'):
            parts = line.split('\t')
            titles = [p.strip('"') for p in parts[1:]]
            n_samples = len(titles)
            descriptions = [[] for _ in range(n_samples)]
            chars = [[] for _ in range(n_samples)]
        elif line.startswith('!Sample_description'):
            parts = line.split('\t')
            vals = [p.strip('"') for p in parts[1:]]
            for i, v in enumerate(vals):
                if i < n_samples:
                    descriptions[i].append(v)
        elif line.startswith('!Sample_characteristics_ch1'):
            parts = line.split('\t')
            vals = [p.strip('"') for p in parts[1:]]
            for i, v in enumerate(vals):
                if i < n_samples:
                    chars[i].append(v)
        elif line.startswith('!series_matrix_table_begin'):
            break

def parse_char(char_list, key):
    for c in char_list:
        if c.startswith(key + ':'):
            return c.split(':', 1)[1].strip()
    return None

records = []
for i in range(n_samples):
    # Cell ID: second !Sample_description entry is "plateID.well"
    cell_id = descriptions[i][1] if len(descriptions[i]) > 1 else descriptions[i][0]
    tissue    = parse_char(chars[i], 'tissue')
    cell_type = parse_char(chars[i], 'cell type')
    patient   = parse_char(chars[i], 'patient id')
    plate_id  = parse_char(chars[i], 'plate id')
    well      = parse_char(chars[i], 'well')
    records.append({'cell_id': cell_id, 'tissue': tissue,
                    'cell_type': cell_type, 'patient_id': patient,
                    'plate_id': plate_id, 'well': well})

meta_df = pd.DataFrame(records).set_index('cell_id')
print(f'  Cells parsed: {len(meta_df)}')
print('  Tissue breakdown:')
print(meta_df['tissue'].value_counts().to_string())
print('  Cell type breakdown:')
print(meta_df['cell_type'].value_counts().to_string())

# ── 3. Load expression matrix ────────────────────────────────────────────────
print('\n=== Loading expression matrix ===')
# Format: genes (rows, quoted) × cells (columns, named plateID.well), space-separated
df = pd.read_csv(EXPR_LOCAL, sep=' ', index_col=0)
print(f'  Raw shape (genes x cells): {df.shape}')

# Cell IDs in CSV columns should match meta_df index
df.index = df.index.str.strip('"')
df.columns = [c.strip('"') for c in df.columns]

# Align to metadata (keep only cells present in both)
shared_cells = [c for c in df.columns if c in meta_df.index]
missing = [c for c in df.columns if c not in meta_df.index]
if missing:
    print(f'  WARNING: {len(missing)} cells in expression matrix not in metadata — dropping')
df = df[shared_cells]
meta_df = meta_df.loc[shared_cells]

# ── 4. Build AnnData ─────────────────────────────────────────────────────────
print('\n=== Building AnnData ===')
# Transpose: cells x genes; X = raw counts
X = csr_matrix(df.values.T.astype(np.float32))
gene_names = pd.Index(df.index.str.lower())  # lowercase gene names

adata = sc.AnnData(
    X=X,
    obs=meta_df,
    var=pd.DataFrame(index=gene_names),
)
adata.obs_names = pd.Index(shared_cells)

print(f'  AnnData shape: {adata.shape}  (cells x genes)')
print(f'  Tissue — Tumor: {(adata.obs["tissue"]=="Tumor").sum()}  Periphery: {(adata.obs["tissue"]=="Periphery").sum()}')

# Basic QC: remove cells with zero total counts
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.filter_genes(adata, min_cells=1)
print(f'  After QC filter: {adata.shape}')

# ── 5. Save ──────────────────────────────────────────────────────────────────
out_path = f'{OUT_DIR}/darmanis_adata.h5ad'
adata.write_h5ad(out_path)
print(f'\n=== Saved: {out_path} ===')
print(adata)
