#!/usr/bin/env Rscript
# Direct edgeR NB 2-group LRT wrapper — used for the HBC competitor comparison.
# This is the same statistical model as DESpace 1.x (svg_test internally used
# edgeR glmLRT); bypasses DESpace 2.x wrapper which errors on sparse Xenium data.
#
# Usage:
#   Rscript run_edger_nb.R <counts.csv> <meta.csv> <out.csv>
#
# counts.csv : genes x cells matrix; row names = gene IDs, col names = cell IDs
# meta.csv   : spot_id, A (0/1), x, y
# out.csv    : gene, pvalue, logfc

suppressPackageStartupMessages(library(edgeR))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: Rscript run_edger_nb.R counts.csv meta.csv out.csv")

counts_mat <- as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE))
mode(counts_mat) <- "integer"
gene_names <- rownames(counts_mat)

meta     <- read.csv(args[2], stringsAsFactors = FALSE)
group    <- factor(as.character(meta$A), levels = c("0", "1"))

# Filter cells with zero library size
lib_sizes <- colSums(counts_mat)
keep_cells <- lib_sizes > 0
if (sum(!keep_cells) > 0)
    cat(sprintf("Filtering %d zero-library cells\n", sum(!keep_cells)))
counts_mat <- counts_mat[, keep_cells, drop = FALSE]
group      <- group[keep_cells]

# edgeR GLM NB — TMM normalisation + common dispersion + gene-wise dispersion
dge  <- DGEList(counts = counts_mat, group = group)
dge  <- calcNormFactors(dge, method = "TMM")
design <- model.matrix(~ group)
dge  <- estimateDisp(dge, design, robust = TRUE)
fit  <- glmFit(dge, design)
lrt  <- glmLRT(fit, coef = 2)   # coef 2 = group1 vs group0

tab  <- lrt$table   # logFC, logCPM, LR, PValue; rownames = gene_names
logfc <- setNames(rep(NA_real_, length(gene_names)), gene_names)
pvals <- setNames(rep(1.0,      length(gene_names)), gene_names)
matched <- intersect(rownames(tab), gene_names)
logfc[matched] <- tab[matched, "logFC"]
pvals[matched] <- tab[matched, "PValue"]

write.csv(data.frame(gene = gene_names, pvalue = pvals, logfc = logfc,
                     row.names = NULL),
          args[3], row.names = FALSE)
cat(sprintf("edgeR NB done: %d genes tested, %d with p<0.05 (raw)\n",
            length(gene_names), sum(pvals < 0.05, na.rm = TRUE)))
