"""
Self-contained synthetic data generator for the TIDEST quick-start example.

This is a trimmed, standalone copy of the data-generating process used in the
TIDEST simulation study (``scripts/simulation/dgp.py``). It is reproduced here
so the quick-start example has *no* dependency on the simulation harness or on
any downloaded data.

Design (see the paper's simulation section for details):
- ``G`` genes are split into ``M`` modules; genes in a module share a latent factor.
- A synthetic single-cell reference shares the same module structure and is used
  only to compute the gene-gene Pearson matrix (no deconvolution).
- Spatial spots lie on a grid, receive a binary treatment ``A`` (two regions),
  and carry a smooth spatial confounder ``Z``.
- True log-expression:  ``Y = baseline + tau_g * A + alpha * beta_g * Z + noise``.
- Observed ST counts are ``Poisson(exp(Y))``.
- "Imputation" adds module-structured noise (what makes the Pearson correction
  worthwhile), mimicking an imputer that misses a cell type / module.
- A SpatialPCA surrogate ``U`` is the top eigenvectors of a Gaussian spatial
  kernel, so the example never needs R.
"""

import numpy as np
import anndata as ad
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


def generate_module_loadings(G, M, seed=0):
    """Return (G, M) loadings: each gene belongs to exactly one module."""
    rng = np.random.default_rng(seed)
    loadings = np.zeros((G, M), dtype=np.float32)
    per = G // M
    for m in range(M):
        start = m * per
        end = start + per if m < M - 1 else G
        loadings[start:end, m] = rng.uniform(0.5, 1.5, size=end - start)
    return loadings


def generate_grid(N, seed=0):
    """Place N spots on a near-sqrt(N) grid in the unit square. Returns (N, 2)."""
    n_side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(xs, xs)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=N, replace=False)
    return coords[idx].astype(np.float32)


def _gaussian_kernel(coords, ell=0.3):
    D2 = cdist(coords, coords, metric="sqeuclidean").astype(np.float64)
    return np.exp(-D2 / (2 * ell ** 2))


def _sample_grf(coords, ell=0.3, seed=0):
    """One realisation of a zero-mean Gaussian random field on the coordinates."""
    K = _gaussian_kernel(coords, ell) + 1e-6 * np.eye(len(coords))
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(seed)
    return (L @ rng.standard_normal(len(coords))).astype(np.float32)


def assign_treatment(coords, noise_std=0.1, seed=2):
    """Binary two-region treatment split along x (1 = treated, 0 = control)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=len(coords)).astype(np.float32)
    return (coords[:, 0] + noise > 0.5).astype(np.int32)


def generate_sc_adata(G, M, loadings, C=2000, seed=1):
    """Synthetic SC reference (raw Poisson counts) sharing the module loadings."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((C, M)).astype(np.float32)
    mu = np.exp(F @ loadings.T + 1.0)
    sc = ad.AnnData(rng.poisson(mu).astype(np.float32))
    sc.var_names = [f"gene{g}" for g in range(G)]
    sc.obs_names = [f"cell{c}" for c in range(C)]
    return sc


def generate_st_expression(coords, A, G_DE, G_null, tau_level=1.0,
                           alpha=0.5, sigma=1.0, ell=0.3, seed=3):
    """Return Y_true (log), C_obs (Poisson counts), tau_true (per-gene truth)."""
    G = G_DE + G_null
    N = len(A)
    rng = np.random.default_rng(seed)

    Z = _sample_grf(coords, ell=ell, seed=seed + 1000)          # spatial confounder
    beta = rng.standard_normal(G).astype(np.float32)            # confounder loadings

    tau_true = np.zeros(G, dtype=np.float32)
    sign = rng.choice([-1, 1], size=G_DE)
    tau_true[:G_DE] = sign * rng.normal(tau_level, 0.1 * tau_level, size=G_DE)

    eps = rng.normal(0, sigma, size=(N, G)).astype(np.float32)
    Y_true = (1.0
              + tau_true[None, :] * A[:, None]
              + alpha * beta[None, :] * Z[:, None]
              + eps)
    C_obs = rng.poisson(np.exp(np.clip(Y_true, -5, 8))).astype(np.float32)
    return Y_true, C_obs, tau_true


def simulate_imputation(Y_true, loadings, sigma_imp=1.0, seed=4):
    """Module-structured noisy imputation (log space). Returns AnnData (pred_is_log)."""
    N, G = Y_true.shape
    M = loadings.shape[1]
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((N, M)).astype(np.float32)
    module_noise = eta @ loadings.T
    eps = rng.normal(0, 0.1 * sigma_imp, size=(N, G)).astype(np.float32)
    Y_pred = Y_true + sigma_imp * module_noise + eps
    pred = ad.AnnData(Y_pred.astype(np.float32))
    pred.var_names = [f"gene{g}" for g in range(G)]
    pred.obs_names = [f"spot{i}" for i in range(N)]
    return pred


def precompute_spatial_pcs(coords, n_pcs=20, ell=0.3):
    """Top-n_pcs eigenvectors of the Gaussian spatial kernel (a SpatialPCA surrogate)."""
    K = _gaussian_kernel(coords, ell=ell) + 1e-8 * np.eye(len(coords))
    n = len(K)
    k = min(n_pcs, n - 1)
    _, vecs = eigh(K, subset_by_index=[n - k, n - 1])
    return vecs[:, ::-1].astype(np.float64)


def make_dataset(N=300, G_DE=30, G_null=70, M=10, n_pcs=20, seed=0):
    """Build a complete synthetic TIDEST dataset.

    Returns
    -------
    dict with keys:
      sc_adata, st_adata, pred_adata : AnnData inputs for ``tidest.fit``
      A          : (N,) treatment indicator
      U          : (N, n_pcs) spatial-PC confounder surrogate (skips R)
      genes      : list of all gene names
      tau_true   : (G,) ground-truth effects (0 for null genes)
      is_de      : (G,) boolean mask of true DE genes
    """
    G = G_DE + G_null
    loadings = generate_module_loadings(G, M, seed=seed)
    coords = generate_grid(N, seed=seed)
    A = assign_treatment(coords, seed=seed + 2)
    Y_true, C_obs, tau_true = generate_st_expression(
        coords, A, G_DE, G_null, seed=seed + 3)

    gene_names = [f"gene{g}" for g in range(G)]
    spot_names = [f"spot{i}" for i in range(N)]

    # ST AnnData: log-CPM-style values (pred_is_log convention), with coordinates
    st = ad.AnnData(np.log1p(C_obs))
    st.var_names = gene_names
    st.obs_names = spot_names
    st.obs["treatment"] = A.astype(np.int32)
    st.obsm["spatial"] = coords

    sc = generate_sc_adata(G, M, loadings, seed=seed + 1)
    pred = simulate_imputation(Y_true, loadings, seed=seed + 4)

    U = precompute_spatial_pcs(coords, n_pcs=n_pcs)

    is_de = np.zeros(G, dtype=bool)
    is_de[:G_DE] = True

    return dict(sc_adata=sc, st_adata=st, pred_adata=pred, A=A, U=U,
                genes=gene_names, tau_true=tau_true, is_de=is_de)
