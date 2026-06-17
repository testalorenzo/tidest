"""
Data generating process for the tidest simulation study.

Design:
- G genes split into M modules; genes within a module share a latent factor.
- Synthetic SC reference (C cells) is built from the same modules — only used to compute
  the Pearson gene-gene correlation matrix (no cell-type deconvolution).
- Synthetic ST spots (N spots) have:
    - A spatial grid
    - A binary treatment A (deep vs superficial, three variants)
    - A spatial confounder Z (independent GRF correlated with A via spatial proximity)
    - True log-expression: Y_true_gi = tau_g * A_i + alpha * beta_g * Z_i + eps_gi
    - Observed counts: Poisson(exp(Y_true))
- Imputed predictions have module-structured noise (mimics Tangram systematically
  missing a module due to SC reference gaps). Pearson correction exploits this structure.
- SpatialPCA surrogate: precomputed kernel-PCA eigenvectors from the spatial coordinates.
"""

import numpy as np
import anndata as ad
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


# ── Gene-module structure ──────────────────────────────────────────────────────

def generate_module_loadings(G, M, seed=0, corr_strength=1.0):
    """
    Return loadings (G, M): each gene belongs to exactly one module.
    Loading value drawn from Uniform(0.5, 1.5) * corr_strength, so modules
    have varying strength.

    corr_strength : float in (0, 1] -- scales all module loadings. Smaller
    values weaken both the within-module gene-gene correlations (computed
    from generate_sc_adata) and the module-structured imputation noise
    (simulate_imputation), diluting the block structure that augmentation
    exploits. corr_strength=1.0 reproduces the original (strong-correlation)
    DGP.
    """
    rng = np.random.default_rng(seed)
    loadings = np.zeros((G, M), dtype=np.float32)
    genes_per_module = G // M
    for m in range(M):
        start = m * genes_per_module
        end = start + genes_per_module if m < M - 1 else G
        strengths = rng.uniform(0.5, 1.5, size=end - start).astype(np.float32)
        loadings[start:end, m] = strengths * corr_strength
    return loadings  # (G, M)


# ── Synthetic SC reference ─────────────────────────────────────────────────────

def generate_sc_adata(G, M, loadings, C=2000, seed=1):
    """
    Generate a synthetic scRNA-seq reference (C cells × G genes) sharing the
    same module loadings as the ST data. Used only for compute_pearson().

    Returns sc_adata with raw Poisson counts (as required by compute_pearson).
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"gene{g}" for g in range(G)]

    # Per-cell latent module activations (standard normal)
    F = rng.standard_normal((C, M)).astype(np.float32)
    mu_log = F @ loadings.T  # (C, G)

    # Add a baseline so most genes have mean exp ~ e^1 = 2.7 counts
    mu_log += 1.0
    mu = np.exp(mu_log)

    # Poisson counts
    counts = rng.poisson(mu).astype(np.float32)

    sc = ad.AnnData(counts)
    sc.var_names = gene_names
    sc.obs_names = [f"cell{c}" for c in range(C)]
    return sc


def generate_sc_adata_mismatched(G, M, loadings, C=2000, seed=1, mismatch_frac=0.3):
    """
    Like generate_sc_adata, but a fraction `mismatch_frac` of genes have their
    module loadings permuted before generating the SC reference.

    The Pearson gene-gene correlation matrix computed from this reference no
    longer matches the module structure that drives the ST imputation noise
    (simulate_imputation, which uses the original `loadings`) for these genes
    -- mimicking a reference that is missing cell types or otherwise
    batch-shifted relative to the tissue. mismatch_frac=0.0 reproduces
    generate_sc_adata exactly.
    """
    rng = np.random.default_rng(seed)
    G_total = loadings.shape[0]
    n_mismatch = int(round(mismatch_frac * G_total))

    loadings_sc = loadings.copy()
    if n_mismatch > 0:
        idx = rng.choice(G_total, size=n_mismatch, replace=False)
        shuffled = rng.permutation(idx)
        loadings_sc[idx] = loadings[shuffled]

    return generate_sc_adata(G, M, loadings_sc, C=C, seed=seed)


# ── Spatial layout ─────────────────────────────────────────────────────────────

def generate_grid(N, seed=0):
    """
    Place N spots on a roughly sqrt(N) x sqrt(N) regular grid in [0, 1]^2.
    Returns coords (N, 2).
    """
    n_side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0, 1, n_side)
    ys = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    # subsample deterministically to exactly N spots
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=N, replace=False)
    return coords[idx].astype(np.float32)


def _gaussian_kernel(coords, ell=0.3):
    """
    Gaussian kernel matrix: K_ij = exp(-||x_i - x_j||^2 / (2*ell^2)).
    """
    D2 = cdist(coords, coords, metric='sqeuclidean').astype(np.float64)
    return np.exp(-D2 / (2 * ell ** 2))


def _sample_grf(coords, ell=0.3, seed=0):
    """
    Sample one realisation of a zero-mean Gaussian random field with Gaussian kernel.
    Returns array of shape (N,).
    """
    K = _gaussian_kernel(coords, ell)
    K += 1e-6 * np.eye(len(K))
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(seed)
    z = L @ rng.standard_normal(len(K))
    return z.astype(np.float32)


def _sample_confounder(coords, confounder_type='smooth', ell=0.3, seed=0):
    """
    Sample one realisation of the spatial confounder field Z, standardised to
    unit variance.

    confounder_type:
      'smooth'        -- Gaussian random field, length-scale ell (original DGP).
      'discontinuous' -- sharp step function along a random axis-aligned split.
      'hotspot'       -- sum of a few narrow Gaussian bumps at random centres
                          (sparse, highly localised confounding).
      'multiscale'    -- sum of a fine-scale (ell/4) and coarse-scale (4*ell)
                          GRF, so confounding has structure at two length scales.
    """
    rng = np.random.default_rng(seed)

    if confounder_type == 'smooth':
        return _sample_grf(coords, ell=ell, seed=seed)

    elif confounder_type == 'discontinuous':
        axis = rng.integers(0, 2)
        thresh = rng.uniform(0.3, 0.7)
        z = np.where(coords[:, axis] > thresh, 1.0, -1.0).astype(np.float32)
        return z

    elif confounder_type == 'hotspot':
        n_hotspots = 3
        centers = rng.uniform(0, 1, size=(n_hotspots, 2))
        z = np.zeros(len(coords), dtype=np.float64)
        for c in centers:
            d2 = ((coords - c) ** 2).sum(axis=1)
            z += np.exp(-d2 / (2 * (ell / 4) ** 2))
        z = (z - z.mean()) / (z.std() + 1e-8)
        return z.astype(np.float32)

    elif confounder_type == 'multiscale':
        z_fine = _sample_grf(coords, ell=ell / 4, seed=seed)
        z_coarse = _sample_grf(coords, ell=ell * 4, seed=seed + 500)
        z = z_fine.astype(np.float64) + z_coarse.astype(np.float64)
        z = (z - z.mean()) / (z.std() + 1e-8)
        return z.astype(np.float32)

    else:
        raise ValueError(f"Unknown confounder_type: {confounder_type!r}")


# ── Treatment assignment ───────────────────────────────────────────────────────

def assign_treatment(coords, variant='two-region', ell=0.3, noise_std=0.1, seed=2):
    """
    Assign binary treatment A (1 = treated/superficial, 0 = control/deep).

    variant : 'two-region' | 'grf' | 'latent-factor'
    """
    rng = np.random.default_rng(seed)
    x = coords[:, 0]

    if variant == 'two-region':
        # Split along x axis; add a small band of noise around the boundary
        noise = rng.normal(0, noise_std, size=len(x)).astype(np.float32)
        A = (x + noise > 0.5).astype(np.int32)

    elif variant == 'grf':
        z = _sample_grf(coords, ell=ell, seed=seed)
        A = (z > 0).astype(np.int32)

    elif variant == 'latent-factor':
        # Correlate A with the first spatial PC of the kernel
        K = _gaussian_kernel(coords, ell=ell)
        vals, vecs = eigh(K, subset_by_index=[len(K) - 1, len(K) - 1])
        pc1 = vecs[:, 0].astype(np.float32)
        noise = rng.normal(0, noise_std, size=len(x)).astype(np.float32)
        A = (pc1 + noise > 0).astype(np.int32)

    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return A


# ── ST expression DGP ─────────────────────────────────────────────────────────

def generate_st_expression(
    coords, A, loadings,
    G_DE, G_null,
    tau_level=1.0,
    alpha=0.5,
    sigma=1.0,
    ell=0.3,
    confounder_type='smooth',
    seed=3,
):
    """
    Generate true latent log-expression and observed ST counts.

    Y_true_gi = tau_g * A_i  +  alpha * beta_g * Z_i  +  eps_gi   (DE gene)
    Y_true_gi =                  alpha * beta_g * Z_i  +  eps_gi   (null gene)

    Parameters
    ----------
    G_DE    : number of DE genes (first G_DE columns of Y_true)
    G_null  : number of null genes
    tau_level : mean |tau| for DE genes (scale of effect size)
    alpha   : confounder strength ∈ [0, 1]
    sigma   : SD of individual noise eps
    ell     : GRF length scale for spatial confounder
    confounder_type : 'smooth' | 'discontinuous' | 'hotspot' | 'multiscale'
        (see _sample_confounder)

    Returns
    -------
    Y_true  : (N, G) float32 — latent log-expression
    C_obs   : (N, G) float32 — Poisson counts (observed ST)
    tau_true: (G,)   float32 — true treatment effects (0 for null genes)
    """
    G = G_DE + G_null
    N = len(A)
    rng = np.random.default_rng(seed)

    # Spatial confounder (independent of treatment-driving field)
    Z = _sample_confounder(coords, confounder_type=confounder_type, ell=ell, seed=seed + 1000)  # (N,)

    # Per-gene confounder loadings
    beta = rng.standard_normal(G).astype(np.float32)

    # Per-gene treatment effects
    tau_true = np.zeros(G, dtype=np.float32)
    sign = rng.choice([-1, 1], size=G_DE)
    tau_true[:G_DE] = (sign * rng.normal(tau_level, 0.1 * tau_level, size=G_DE)).astype(np.float32)

    # Individual spot-gene noise
    eps = rng.normal(0, sigma, size=(N, G)).astype(np.float32)

    # True log-expression: baseline + treatment effect + confounder + noise
    baseline = 1.0  # shifts mean count to ~e^1 ≈ 2.7
    Y_true = (
        baseline
        + tau_true[None, :] * A[:, None]
        + alpha * beta[None, :] * Z[:, None]
        + eps
    )

    # Observed counts ~ Poisson(exp(Y_true))
    mu = np.exp(np.clip(Y_true, -5, 8))
    C_obs = rng.poisson(mu).astype(np.float32)

    return Y_true, C_obs, tau_true


def build_st_adata(C_obs, coords, A, G_DE, G_null, treatment_col='treatment'):
    """
    Pack Poisson counts and spatial coordinates into an AnnData.
    st_adata.X = log1p(C_obs)  (pred_is_log=True convention used downstream)
    """
    G = G_DE + G_null
    gene_names = [f"gene{g}" for g in range(G)]
    spot_names = [f"spot{i}" for i in range(len(A))]

    st = ad.AnnData(np.log1p(C_obs))
    st.var_names = gene_names
    st.obs_names = spot_names
    st.obs[treatment_col] = A.astype(np.int32)
    st.obsm['spatial'] = coords.astype(np.float32)
    return st


# ── Imputation simulation ──────────────────────────────────────────────────────

def simulate_imputation(Y_true, loadings, sigma_imp, seed=4):
    """
    Simulate noisy imputation output (mimics Tangram output in log space).

    Noise is module-structured: the imputer systematically over/under-predicts
    all genes in a module together (because the SC reference misses a cell type).
    Without structured noise, the Pearson correction provides no benefit; this
    structured noise is what makes augmentation meaningful.

    Y_pred_gi = Y_true_gi + sigma_imp * (loadings[g,:] @ eta_i) + epsilon_gi

    where eta_i ~ N(0, I_M) is a per-spot module error (same for all genes in the
    module at spot i), and epsilon_gi is small independent jitter.

    Parameters
    ----------
    sigma_imp : float — imputation noise magnitude (0 = perfect, >1 = poor)

    Returns
    -------
    Y_pred : (N, G) float32
    pred_adata : AnnData with Y_pred as .X (pred_is_log=True)
    """
    N, G = Y_true.shape
    M = loadings.shape[1]
    rng = np.random.default_rng(seed)
    gene_names = [f"gene{g}" for g in range(G)]
    spot_names = [f"spot{i}" for i in range(N)]

    # Module-level errors: (N, M) → correlated gene errors via loadings
    eta = rng.standard_normal((N, M)).astype(np.float32)
    module_noise = eta @ loadings.T  # (N, G)

    # Small independent jitter (10% of structured noise)
    eps = rng.normal(0, 0.1 * sigma_imp, size=(N, G)).astype(np.float32)

    Y_pred = Y_true + sigma_imp * module_noise + eps

    pred = ad.AnnData(Y_pred.astype(np.float32))
    pred.var_names = gene_names
    pred.obs_names = spot_names
    return Y_pred, pred


# ── Spatial PCA surrogate ──────────────────────────────────────────────────────

def precompute_spatial_pcs(coords, n_pcs=20, ell=0.3):
    """
    Compute the top n_pcs eigenvectors of the Gaussian spatial kernel matrix.
    Equivalent to the kernel PCA step inside SpatialPCA.

    Returns U : (N, n_pcs) float64
    """
    K = _gaussian_kernel(coords, ell=ell)
    K += 1e-8 * np.eye(len(K))
    n = len(K)
    k = min(n_pcs, n - 1)
    # eigh returns ascending order; we want the top-k (largest)
    vals, vecs = eigh(K, subset_by_index=[n - k, n - 1])
    U = vecs[:, ::-1]  # (N, k) descending eigenvalue order
    return U.astype(np.float64)
