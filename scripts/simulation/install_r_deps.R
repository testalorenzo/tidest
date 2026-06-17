#!/usr/bin/env Rscript
# One-time setup: install R dependencies for the simulation study competitors.
# Run from spatrans/: Rscript scripts/simulation/install_r_deps.R

required <- c("dplyr", "geepack", "parallel")
for (pkg in required) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg, repos = "https://cloud.r-project.org")
    }
}

if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org")
}

devtools::install_github("yishan03/SpatialGEE")

cat("R dependency installation complete.\n")
cat("DESpace is assumed to already be installed (BiocManager::install('DESpace')).\n")
if (requireNamespace("DESpace", quietly = TRUE))    cat("  DESpace    : OK\n")
if (requireNamespace("SpatialGEE", quietly = TRUE)) cat("  SpatialGEE : OK\n")
