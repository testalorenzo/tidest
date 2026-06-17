#!/usr/bin/env Rscript
# SpatialGEE run_gee_gst wrapper for the simulation study.
#
# Usage:
#   Rscript run_spatialgee.R <counts.csv> <meta.csv> <out.csv>
#
# counts.csv : genes x spots matrix; row names = gene IDs, col names = spot IDs
# meta.csv   : spot_id, A (0/1), x, y
# out.csv    : gene, pvalue

suppressPackageStartupMessages(library(SpatialGEE))
suppressPackageStartupMessages(library(dplyr))     # for %>%, filter, mutate, group_by
suppressPackageStartupMessages(library(geepack))   # for geeglm (full model coefficient)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop("Usage: Rscript run_spatialgee.R <counts.csv> <meta.csv> <out.csv>")
}
counts_file <- args[1]
meta_file   <- args[2]
out_file    <- args[3]

# ── Read data ──────────────────────────────────────────────────────────────────
counts_mat <- as.matrix(read.csv(counts_file, row.names = 1, check.names = FALSE))
gene_names <- rownames(counts_mat)   # length G
N <- ncol(counts_mat)                # number of spots

meta <- read.csv(meta_file, stringsAsFactors = FALSE)

# ── Build SpatialGEE data.frame ───────────────────────────────────────────────
# Required columns (in this order): Barcodes, x, y, Pathology.Annotations
# Then gene expression columns (spots x genes).
annotations <- ifelse(meta$A == 1, "trt", "ctrl")

gene_df <- as.data.frame(t(counts_mat))  # spots x genes, col names = gene IDs
colnames(gene_df) <- gene_names

df <- cbind(
    data.frame(
        Barcodes              = meta$spot_id,
        x                     = meta$x,
        y                     = meta$y,
        Pathology.Annotations = annotations,
        stringsAsFactors      = FALSE
    ),
    gene_df
)

# k must be < N/2 (each cluster needs >1 spot); default 100 is fine for N>=300
k <- min(100L, as.integer(floor(N / 3L)))

# ── Run SpatialGEE ─────────────────────────────────────────────────────────────
res <- tryCatch(
    run_gee_gst(
        data           = df,
        compare_levels = c("ctrl", "trt"),
        k              = k,
        family         = poisson,
        corstr         = "independence",
        cores          = 1L
    ),
    error = function(e) {
        cat("run_gee_gst error:", conditionMessage(e), "\n", file = stderr())
        NULL
    }
)

# ── Full GEE model for treatment coefficient ───────────────────────────────────
# The GST tests H0: beta1=0 but doesn't estimate beta1. We fit the full model
# geeglm(geneexp ~ Pathology.Annotations, ...) to extract the treatment coefficient
# (log-scale Poisson coefficient for "trt" vs "ctrl").
df_full <- df %>%
    filter(Pathology.Annotations %in% c("ctrl", "trt")) %>%
    mutate(Pathology.Annotations = factor(Pathology.Annotations, levels = c("ctrl", "trt")))

set.seed(123)
kmeans_full <- kmeans(df_full[, c("x", "y")], centers = k, nstart = 1)
df_full$Clusters <- kmeans_full$cluster
df_full <- df_full %>%
    group_by(Clusters) %>%
    filter(n() > 1) %>%
    ungroup() %>%
    arrange(Clusters)

gee_coef <- setNames(rep(NA_real_, length(gene_names)), gene_names)
gene_cols_full <- intersect(gene_names, colnames(df_full))
for (g in gene_cols_full) {
    coef_val <- tryCatch({
        fit <- geeglm(
            df_full[[g]] ~ Pathology.Annotations,
            family  = poisson,
            data    = df_full,
            id      = df_full$Clusters,
            corstr  = "independence"
        )
        coef(fit)[2]   # log Poisson coefficient for "trt" vs "ctrl"
    }, error = function(e) NA_real_)
    gee_coef[g] <- coef_val
}

# ── Extract p-values and write output ─────────────────────────────────────────
# res has columns: gene, p_value  (GST p-value)
# gee_coef: treatment log-coefficient from full GEE model
if (!is.null(res)) {
    pvals <- setNames(rep(1.0, length(gene_names)), gene_names)
    matched <- intersect(res$gene, gene_names)
    pvals[matched] <- res$p_value[match(matched, res$gene)]
    out_df <- data.frame(gene = gene_names, pvalue = pvals,
                         gee_coef = gee_coef, row.names = NULL)
} else {
    out_df <- data.frame(gene = gene_names, pvalue = rep(NA_real_, length(gene_names)),
                         gee_coef = gee_coef)
}

write.csv(out_df, out_file, row.names = FALSE)
