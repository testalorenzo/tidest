library(SpatialPCA)
library(Matrix)
library(arrow)

# Called as: Rscript spatialPCA_runner.R <tmp_dir> <n_pcs>
args   <- commandArgs(trailingOnly = TRUE)
tmp_dir <- args[1]
n_pcs   <- as.integer(args[2])

cat(sprintf("spatialPCA_runner: tmp_dir=%s, n_pcs=%d\n", tmp_dir, n_pcs))

# Build a minimal SpatialPCA object without Seurat/SPARK dependency.
# normalized_expr is set to the confounder-gene subset; SpatialPCA_buildKernel
# row-standardizes internally so no additional normalization is needed.
make_spatialPCA_object <- function(counts, location, gene_list) {
    keep <- intersect(gene_list, rownames(counts))
    cat(sprintf("  Matched %d / %d confounder genes\n", length(keep), length(gene_list)))
    expr <- counts[keep, , drop = FALSE]
    new("SpatialPCA",
        counts             = Matrix(counts, sparse = TRUE),
        normalized_expr    = expr,
        location           = location,
        project            = "SpatialPCA",
        covariate          = NULL,
        kernelmat          = NULL,
        kerneltype         = character(0),
        bandwidthtype      = character(0),
        bandwidth          = numeric(0),
        sparseKernel       = FALSE,
        sparseKernel_tol   = 1e-20,
        sparseKernel_ncore = 1,
        fast               = FALSE,
        eigenvecnum        = numeric(0),
        SpatialPCnum       = numeric(0),
        tau                = numeric(0),
        sigma2_0           = numeric(0),
        W                  = NULL,
        SpatialPCs         = NULL,
        highPCs            = NULL,
        highPos            = NULL,
        expr_pred          = NULL,
        params             = list()
    )
}

counts_tbl  <- read_feather(file.path(tmp_dir, "pseudo_counts.feather"))
counts      <- as.matrix(counts_tbl[, -1])
rownames(counts) <- counts_tbl[[1]]

loc_tbl     <- read_feather(file.path(tmp_dir, "pseudo_location.feather"))
location    <- as.matrix(loc_tbl[, c("x", "y")])
rownames(location) <- loc_tbl[["spot"]]

gene_list   <- readLines(file.path(tmp_dir, "confounder_genes.txt"))
cat(sprintf("  Using %d confounder genes\n", length(gene_list)))

spca <- make_spatialPCA_object(counts, location, gene_list)
spca <- SpatialPCA_buildKernel(spca, kerneltype = "gaussian", bandwidthtype = "SJ")
spca <- SpatialPCA_EstimateLoading(spca, SpatialPCnum = n_pcs, fast = TRUE, eigenvecnum = n_pcs)
spca <- SpatialPCA_SpatialPCs(spca, fast = TRUE)

pcs_df          <- as.data.frame(t(spca@SpatialPCs))
rownames(pcs_df) <- colnames(counts)
colnames(pcs_df) <- paste0("SpatialPC", seq_len(n_pcs))

out_path <- file.path(tmp_dir, "spatialPCA_pcs.feather")
write_feather(cbind(spot = rownames(pcs_df), pcs_df), out_path)
cat(sprintf("  Saved PCs to %s\n", out_path))
