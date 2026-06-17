library(SpatialPCA)
library(Matrix)
library(arrow)

n_reps <- 1
n_pcs  <- 50
CHUNK  <- 10000   # cells per chunk for Nyström kernel computation

# ---------------------------------------------------------------------------
# Helper: Gaussian kernel between two location matrices.
# Returns an (n1 x n2) matrix using the ||a-b||^2 = ||a||^2+||b||^2-2a'b trick.
# ---------------------------------------------------------------------------
gaussian_kernel <- function(loc1, loc2, bw) {
    d2_1  <- rowSums(loc1^2)
    d2_2  <- rowSums(loc2^2)
    cross <- tcrossprod(loc1, loc2)               # n1 x n2
    D2    <- outer(d2_1, d2_2, "+") - 2 * cross  # n1 x n2
    exp(-D2 / (2 * bw^2))
}

# ---------------------------------------------------------------------------
# Nyström extension: project new_loc onto the SpatialPCA eigenvectors learned
# from the training subsample.
#
# Formula: Z_new = K(new, train) %*% solve(K_train + eps*I, Z_train)
#   K_train  = spca@kernelmat          (N_train x N_train)
#   Z_train  = t(spca@SpatialPCs)      (N_train x k)
#   K(new,.) = Gaussian kernel between new_loc and train_loc
#
# alpha = K_train^{-1} Z_train is precomputed once via Cholesky factorisation.
# New locations are processed in chunks of CHUNK rows to stay memory-safe.
# ---------------------------------------------------------------------------
nystrom_project <- function(spca, train_loc, all_loc, bw, chunk = CHUNK) {

    K_train <- as.matrix(spca@kernelmat)   # N_train x N_train
    Z_train <- t(spca@SpatialPCs)          # N_train x k

    # Regularised Cholesky solve: alpha = (K + eps*I)^{-1} Z_train
    eps   <- 1e-8 * max(diag(K_train))
    K_reg <- K_train + eps * diag(nrow(K_train))
    alpha <- solve(K_reg, Z_train)         # N_train x k

    N_all <- nrow(all_loc)
    k     <- ncol(Z_train)
    Z_all <- matrix(0.0, nrow = N_all, ncol = k)

    cat(sprintf("  Projecting %d cells in chunks of %d...\n", N_all, chunk))
    starts <- seq(1, N_all, by = chunk)
    for (s in starts) {
        e       <- min(s + chunk - 1, N_all)
        K_chunk <- gaussian_kernel(all_loc[s:e, , drop = FALSE], train_loc, bw)
        Z_all[s:e, ] <- K_chunk %*% alpha
        if ((which(starts == s) %% 5) == 0 || e == N_all) {
            cat(sprintf("    processed %d / %d cells\n", e, N_all))
        }
    }
    Z_all
}


make_spatialPCA_object <- function(counts, location, gene_list) {
    keep <- intersect(gene_list, rownames(counts))
    cat(sprintf("  Matched %d / %d non-HVG genes in counts matrix\n",
                length(keep), length(gene_list)))
    expr <- counts[keep, , drop = FALSE]
    new("SpatialPCA",
        counts          = Matrix(counts, sparse = TRUE),
        normalized_expr = expr,
        location        = location,
        project         = "SpatialPCA",
        covariate       = NULL,
        kernelmat       = NULL,
        kerneltype      = character(0),
        bandwidthtype   = character(0),
        bandwidth       = numeric(0),
        sparseKernel    = FALSE,
        sparseKernel_tol   = 1e-20,
        sparseKernel_ncore = 1,
        fast            = FALSE,
        eigenvecnum     = numeric(0),
        SpatialPCnum    = numeric(0),
        tau             = numeric(0),
        sigma2_0        = numeric(0),
        W               = NULL,
        SpatialPCs      = NULL,
        highPCs         = NULL,
        highPos         = NULL,
        expr_pred       = NULL,
        params          = list()
    )
}


for (rep in seq_len(n_reps) - 1) {

    cat(sprintf("\n=== Rep %d ===\n", rep))

    # ── Load training data (3k-cell subsample) ────────────────────────────────
    cat("Loading training data...\n")
    counts_tbl <- read_feather(sprintf("./results_raw/hbc_pseudo_counts%d.feather", rep))
    counts <- as.matrix(counts_tbl[, -1])
    rownames(counts) <- counts_tbl[[1]]

    loc_tbl    <- read_feather(sprintf("./results_raw/hbc_pseudo_location%d.feather", rep))
    train_loc  <- as.matrix(loc_tbl[, c("x", "y")])
    train_ids  <- loc_tbl[["spot"]]
    rownames(train_loc) <- train_ids

    non_hvg_genes <- readLines(sprintf("./results_raw/hbc_non_hvg_genes%d.txt", rep))
    cat(sprintf("  Training cells: %d,  non-HVG genes: %d\n",
                nrow(train_loc), length(non_hvg_genes)))

    # ── Fit SpatialPCA on training subsample ─────────────────────────────────
    cat("Fitting SpatialPCA on training subsample...\n")
    spca <- make_spatialPCA_object(counts, train_loc, non_hvg_genes)
    spca <- SpatialPCA_buildKernel(spca, kerneltype = "gaussian", bandwidthtype = "SJ")
    spca <- SpatialPCA_EstimateLoading(spca, SpatialPCnum = n_pcs,
                                       fast = TRUE, eigenvecnum = n_pcs)
    spca <- SpatialPCA_SpatialPCs(spca, fast = TRUE)
    bw   <- spca@bandwidth
    cat(sprintf("  Fitted SpatialPCA: %d PCs, bandwidth = %.4f\n", n_pcs, bw))

    # ── Load ALL cell locations ───────────────────────────────────────────────
    cat("Loading all-cell locations for Nyström projection...\n")
    all_loc_tbl <- read_feather(sprintf("./results_raw/hbc_all_location%d.feather", rep))
    all_loc     <- as.matrix(all_loc_tbl[, c("x", "y")])
    all_ids     <- all_loc_tbl[["spot"]]
    rownames(all_loc) <- all_ids
    cat(sprintf("  All cells: %d\n", nrow(all_loc)))

    # ── Nyström projection onto all cells ─────────────────────────────────────
    cat("Running Nyström projection...\n")
    Z_all <- nystrom_project(spca, train_loc, all_loc, bw)

    # ── Save projected PCs for all cells ─────────────────────────────────────
    pcs_df <- as.data.frame(Z_all)
    rownames(pcs_df) <- all_ids
    colnames(pcs_df) <- paste0("SpatialPC", seq_len(n_pcs))

    out_path <- sprintf("./results_raw/hbc_spatialPCA_pcs%d.feather", rep)
    write_feather(cbind(spot = rownames(pcs_df), pcs_df), out_path)
    cat(sprintf("  Saved projected PCs (%d cells × %d PCs) to %s\n",
                nrow(pcs_df), n_pcs, out_path))
}
