#!/usr/bin/env Rscript
# DESpace svg_test wrapper for the simulation study.
#
# Usage:
#   Rscript run_despace.R <counts.csv> <meta.csv> <out.csv>
#
# counts.csv : genes x spots matrix; row names = gene IDs, col names = spot IDs
# meta.csv   : spot_id, A (0/1), x, y
# out.csv    : gene, pvalue  (one row per gene; NaN if gene was filtered out)

suppressPackageStartupMessages({
    library(DESpace)
    library(SpatialExperiment)
    library(SingleCellExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop("Usage: Rscript run_despace.R <counts.csv> <meta.csv> <out.csv>")
}
counts_file <- args[1]
meta_file   <- args[2]
out_file    <- args[3]

# ── Read data ──────────────────────────────────────────────────────────────────
counts_mat <- as.matrix(read.csv(counts_file, row.names = 1, check.names = FALSE))
mode(counts_mat) <- "integer"   # svg_test requires integer counts
gene_names <- rownames(counts_mat)

meta <- read.csv(meta_file, stringsAsFactors = FALSE)
spot_ids <- as.character(meta$spot_id)
colnames(counts_mat) <- spot_ids

# ── Filter zero-library cells (edgeR calcNormFactors fails on them) ────────────
lib_sizes  <- colSums(counts_mat)
keep_cells <- lib_sizes > 0
if (sum(!keep_cells) > 0)
    message(sprintf("Filtering %d zero-library cells before DESpace", sum(!keep_cells)))
counts_mat <- counts_mat[, keep_cells, drop = FALSE]
spot_ids   <- spot_ids[keep_cells]
meta       <- meta[keep_cells, , drop = FALSE]

# ── Build SpatialExperiment ────────────────────────────────────────────────────
col_data <- data.frame(
    treatment = as.character(meta$A),
    row.names = spot_ids
)
spatial_coords <- as.matrix(meta[, c("x", "y")])
rownames(spatial_coords) <- spot_ids

spe <- SpatialExperiment(
    assays        = list(counts = counts_mat),
    colData       = col_data,
    spatialCoords = spatial_coords
)

# ── Run DESpace svg_test ────────────────────────────────────────────────────────
# cluster_col = "treatment": two clusters (A=0 and A=1) as spatial regions.
# Disable gene filters because simulation genes are all expressed by construction.
set.seed(123)
res <- tryCatch(
    svg_test(
        spe               = spe,
        cluster_col       = "treatment",
        min_counts        = 1,
        min_non_zero_spots = 1,
        filter_gene       = FALSE,
        verbose           = TRUE    # needed to access glmLrt$table$logFC
    ),
    error = function(e) {
        cat("svg_test error:", conditionMessage(e), "\n", file = stderr())
        NULL
    }
)

# ── Extract p-values and logFC, write output ──────────────────────────────────
# gene_results: gene_id, LR, logCPM, PValue, FDR  (no logFC)
# glmLrt$table: logFC, logCPM, LR, PValue         (logFC is here)
if (!is.null(res) && !is.null(res$gene_results)) {
    gr   <- res$gene_results    # rownames = gene IDs
    glrt <- res$glmLrt$table    # rownames = gene IDs, contains logFC

    pvals <- setNames(rep(1.0,       length(gene_names)), gene_names)
    logfc <- setNames(rep(NA_real_,  length(gene_names)), gene_names)

    matched_gr   <- intersect(rownames(gr),   gene_names)
    matched_glrt <- intersect(rownames(glrt), gene_names)

    pvals[matched_gr] <- gr[matched_gr, "PValue"]
    if (!is.null(glrt) && "logFC" %in% colnames(glrt))
        logfc[matched_glrt] <- glrt[matched_glrt, "logFC"]

    out_df <- data.frame(gene = gene_names, pvalue = pvals, logfc = logfc, row.names = NULL)
} else {
    out_df <- data.frame(gene = gene_names, pvalue = rep(NA_real_, length(gene_names)),
                         logfc = rep(NA_real_, length(gene_names)))
}

write.csv(out_df, out_file, row.names = FALSE)
