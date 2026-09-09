"""
copula_common.py — shared numerics for the conditional-copula six-hourly
control (see generate_6h_conditional_copula_control.py and
fit_conditional_copula.py).

Three pieces, used identically by the fitting step (on the day 1-7 training
window) and the application step (on the day 10-end test window), so the
covariate definition can never drift between the two:

1. compute_covariates(): the 3 local, forecast-only covariates used to
   condition the copula's dependence parameter.
2. build_toeplitz_from_lag_corrs() / psd_clip_correlation() /
   sample_correlated_uniforms_from_corr(): draws rank-correlated uniforms
   from a Gaussian copula given an empirically-fit multi-lag Toeplitz
   correlation matrix (one rho per lag, not assumed to follow any ARMA
   decay formula -- see fit_conditional_copula.py). ar1_correlation_matrix()
   / sample_correlated_uniforms() are kept alongside for reference/
   comparison against that earlier (rejected) AR(1) assumption.
3. invert_zig_mixture(): inverts the zero-inflated two-component Gamma
   mixture CDF (uniform -> physical value) via vectorized bisection on
   torch.special.gammainc, so correlated uniforms can be mapped through
   each hour's own (correct) marginal distribution.

Degenerate (effectively-dry) cases: if the *most likely* hour still has
less than a 5% chance of precip (c_min_p0 > DEGENERATE_P0_THRESHOLD),
dependence structure is irrelevant -- nothing is likely to happen in any
hour, so the copula and the naive independence assumption agree. Such
cases are excluded from the copula fit entirely (they're uninformative,
mostly ties) and, at application time, routed to the cheap independence
code path instead of the copula machinery.

Tom Hamill, Jul 2026
"""
import numpy as np
import torch

EPS = 1e-7
DEGENERATE_P0_THRESHOLD = 0.95   # min_k p0_k above this -> skip copula, use independence
IMPORTANCE_ACONST = 0.01          # matches aconst in sample_6hourly_prob_mrms.py


def importance_weight(c_min_p0):
    """Same importance-sampling formula used elsewhere in this pipeline
    (sample_6hourly_prob_mrms.py's psamp), reused here to concentrate the
    copula-fitting sample on informative (more-likely-wet) cases."""
    return IMPORTANCE_ACONST + (1.0 - IMPORTANCE_ACONST) * (1.0 - c_min_p0)


def is_degenerate(c_min_p0, threshold=DEGENERATE_P0_THRESHOLD):
    """True where dependence structure doesn't matter (every hour near-certain dry)."""
    return c_min_p0 > threshold


def compute_covariates(frac_zero_6, mix_weight_6, gshape1_6, gscale1_6,
                        gshape2_6, gscale2_6):
    """
    Three local, forecast-only covariates for conditioning the copula's
    dependence parameter, computed identically on the historical (fitting)
    side and the application side.

    Parameters
    ----------
    *_6 : arrays, shape (6, ...) — hourly gamma-mixture parameters,
          any trailing shape (works for a flat point sample or a full
          (ny, nx) field).

    Returns
    -------
    c_min_p0   : min_k p0_k        -- peak-hour wetness proxy
    c_max_p0   : max_k p0_k        -- temporal-intermittency proxy
    c_mean_amt : mean_k[(1-p0_k) * (w_k*mu1_k + (1-w_k)*mu2_k)] -- magnitude
    """
    mu1 = gshape1_6 * gscale1_6
    mu2 = gshape2_6 * gscale2_6
    hourly_mean = (1.0 - frac_zero_6) * (mix_weight_6 * mu1 + (1.0 - mix_weight_6) * mu2)

    c_min_p0   = frac_zero_6.min(axis=0)
    c_max_p0   = frac_zero_6.max(axis=0)
    c_mean_amt = hourly_mean.mean(axis=0)
    return c_min_p0, c_max_p0, c_mean_amt


def ar1_correlation_matrix(rho, n=6, device='cpu'):
    """6x6 AR(1)/Toeplitz correlation matrix: corr(i,j) = rho^|i-j|."""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    lag = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return rho.reshape(-1, 1, 1) ** lag if rho.dim() > 0 else rho ** lag


def build_toeplitz_from_lag_corrs(rho_lags, n=6):
    """
    Build a batch of nxn Toeplitz correlation matrices from per-pixel
    empirical lag correlations (the multi-lag counterpart of
    ar1_correlation_matrix(), used when rho(lag) is fit directly per lag
    rather than assumed to decay as rho^lag).

    rho_lags : (npix, n-1) tensor -- rho_lags[:, k-1] is the lag-k
               correlation, k = 1..n-1.
    Returns  : (npix, n, n) tensor, corr[:,i,j] = 1 if i==j else
               rho_lags[:, |i-j|-1].
    """
    npix = rho_lags.shape[0]
    device = rho_lags.device
    idx = torch.arange(n, device=device)
    lag = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # (n, n), values 0..n-1
    corr = torch.ones(npix, n, n, device=device, dtype=rho_lags.dtype)
    for k in range(1, n):
        corr[:, lag == k] = rho_lags[:, k - 1].unsqueeze(-1)
    return corr


def psd_clip_correlation(corr, min_eig=1e-6):
    """
    Project a batch of symmetric matrices onto the nearest valid
    correlation matrix: clip eigenvalues below `min_eig`, reconstruct, and
    renormalize the diagonal back to 1. An empirically-estimated
    multi-lag Toeplitz matrix (unlike one derived from an AR(1)/AR(2)
    formula) is not guaranteed positive semi-definite, so this guard is
    required before Cholesky.
    """
    eigvals, eigvecs = torch.linalg.eigh(corr)
    eigvals = torch.clamp(eigvals, min=min_eig)
    corr_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    d = torch.sqrt(torch.diagonal(corr_psd, dim1=-2, dim2=-1))
    return corr_psd / (d.unsqueeze(-1) * d.unsqueeze(-2))


def sample_correlated_uniforms_from_corr(corr, n_members, device, ridge=1e-5):
    """
    Draw (npix, n_hours, n_members) correlated uniforms from a Gaussian
    copula given an already-PSD per-pixel correlation matrix `corr`
    (npix, n_hours, n_hours). Shared sampling core for both the AR(1) path
    (sample_correlated_uniforms()) and the production empirical multi-lag
    Toeplitz path (build_toeplitz_from_lag_corrs() + psd_clip_correlation()).
    """
    npix, n_hours, _ = corr.shape
    # Regularize for numerical stability (correlations close to +-1 can
    # make the matrix near-singular).
    corr = corr + ridge * torch.eye(n_hours, device=device).unsqueeze(0)
    L = torch.linalg.cholesky(corr)                                 # (npix, n, n)

    z = torch.randn(npix, n_hours, n_members, device=device)
    z_corr = torch.matmul(L, z)                                     # (npix, n, members)
    normal = torch.distributions.Normal(0.0, 1.0)
    u = normal.cdf(z_corr)
    return u.clamp(1e-6, 1 - 1e-6)


def sample_correlated_uniforms(rho, n_members, device, n_hours=6):
    """
    Draw (npix, n_hours, n_members) correlated uniforms from a Gaussian
    copula with AR(1) correlation `rho` (rho: (npix,) tensor, one value
    per pixel). Kept for reference/comparison against the production
    empirical multi-lag Toeplitz path -- see
    sample_correlated_uniforms_from_corr().
    """
    corr = ar1_correlation_matrix(rho, n=n_hours, device=device)   # (npix, n, n)
    return sample_correlated_uniforms_from_corr(corr, n_members, device)


def invert_zig_mixture(u, frac_zero, mix_weight, shape1, scale1, shape2, scale2,
                        n_iter=20, x_max=300.0):
    """
    Invert the zero-inflated two-component Gamma-mixture CDF at uniform
    values `u`, i.e. return x such that F(x) = u, via vectorized bisection.
    All inputs broadcast to a common shape; `u` carries the ensemble-member
    dimension.

    F(x) = frac_zero + (1-frac_zero) * [w * GammaCDF(x;a1,s1)
                                          + (1-w) * GammaCDF(x;a2,s2)],  x > 0
    F(0) = frac_zero  (point mass)
    """
    a1 = torch.clamp(shape1, min=EPS)
    s1 = torch.clamp(scale1, min=EPS)
    a2 = torch.clamp(shape2, min=EPS)
    s2 = torch.clamp(scale2, min=EPS)

    dry_mask = u <= frac_zero
    u_pos = ((u - frac_zero) / torch.clamp(1.0 - frac_zero, min=EPS)).clamp(1e-6, 1 - 1e-6)

    def mixture_cdf(x):
        return (mix_weight * torch.special.gammainc(a1, x / s1)
                + (1.0 - mix_weight) * torch.special.gammainc(a2, x / s2))

    lo = torch.zeros_like(u_pos)
    hi = torch.full_like(u_pos, x_max)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_low = mixture_cdf(mid) < u_pos
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    x = 0.5 * (lo + hi)

    return torch.where(dry_mask, torch.zeros_like(x), x)


def pit_to_normal_score(r, frac_zero, mix_weight, shape1, scale1, shape2, scale2, rng):
    """
    Randomized probability-integral-transform of an observed value `r`
    through its own forecast's zero-inflated two-component Gamma-mixture
    CDF, mapped to a standard-normal score.  Used only by the fitting step
    (numpy/scipy, CPU) -- the forward-CDF counterpart of invert_zig_mixture.

    Exact zeros land in the CDF's point mass [0, frac_zero); the point
    mass is not invertible one-to-one, so a uniform draw within it (rather
    than a fixed placement) avoids introducing spurious ties/bias in the
    resulting rank correlation estimate.

    Parameters
    ----------
    r, frac_zero, mix_weight, shape1, scale1, shape2, scale2 : np.ndarray, same shape
    rng : np.random.Generator

    Returns
    -------
    z : np.ndarray, same shape -- standard-normal score
    """
    from scipy.special import gammainc
    from scipy.stats import norm

    a1 = np.maximum(shape1, EPS)
    s1 = np.maximum(scale1, EPS)
    a2 = np.maximum(shape2, EPS)
    s2 = np.maximum(scale2, EPS)

    is_zero = (r <= 0.0)
    u = np.empty_like(r, dtype=np.float64)
    u[is_zero] = rng.uniform(0.0, np.maximum(frac_zero[is_zero], EPS))
    pos = ~is_zero
    mix_cdf = (mix_weight[pos] * gammainc(a1[pos], r[pos] / s1[pos])
               + (1.0 - mix_weight[pos]) * gammainc(a2[pos], r[pos] / s2[pos]))
    u[pos] = frac_zero[pos] + (1.0 - frac_zero[pos]) * mix_cdf
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def load_copula_grid(npz_path):
    """Load a fit_conditional_copula.py output file into a dict of numpy
    arrays ready for lookup_rho_multilag()."""
    d = np.load(npz_path)
    return {
        'grid_edges':  [d['grid_edge0'], d['grid_edge1'], d['grid_edge2']],
        'rho_grid':    d['rho_grid'],     # (n_lags, n1, n2, n3)
        'neff_grid':   d['neff_grid'],    # (n_lags, n1, n2, n3)
        'pooled_rho':  d['pooled_rho'],   # (n_lags,)
        'cov_mean':    d['cov_mean'],
        'cov_std':     d['cov_std'],
    }


def lookup_rho_multilag(c1, c2, c3, grid, min_n_eff=50):
    """
    Nearest-grid-node lookup of the fitted empirical lag correlations,
    one per lag, with fallback to the nearest node that has adequate local
    support (n_eff >= min_n_eff) at that lag, or -- if no node anywhere
    has adequate support at that lag -- the pooled/unconditional
    correlation for that lag.

    c1, c2, c3 : arrays, any common shape (e.g. a full (ny, nx) domain
                 field, or a flat per-pixel batch)
    grid       : dict from load_copula_grid()

    Returns
    -------
    rho_lags : array, shape c1.shape + (n_lags,), always finite.
    """
    shape = c1.shape
    cov = np.stack([c1.ravel(), c2.ravel(), c3.ravel()], axis=1)   # (n, 3)
    cov_z = (cov - grid['cov_mean']) / grid['cov_std']

    edges = grid['grid_edges']
    mesh = np.meshgrid(*edges, indexing='ij')
    node_coords = np.stack([m.ravel() for m in mesh], axis=1)       # (n_nodes, 3)

    # distance from every query point to every node (n_nodes is small, so a
    # dense (n_query, n_nodes) distance matrix is cheap, shared across lags)
    dist2 = ((cov_z[:, None, :] - node_coords[None, :, :]) ** 2).sum(axis=2)   # (n, n_nodes)

    n_lags = grid['rho_grid'].shape[0]
    out = np.empty((cov.shape[0], n_lags), dtype=np.float32)
    for ilag in range(n_lags):
        node_rho  = grid['rho_grid'][ilag].ravel()
        node_neff = grid['neff_grid'][ilag].ravel()
        supported = node_neff >= min_n_eff

        if not supported.any():
            out[:, ilag] = grid['pooled_rho'][ilag]
            continue

        d2 = dist2.copy()
        d2[:, ~supported] = np.inf
        nearest = d2.argmin(axis=1)
        out[:, ilag] = node_rho[nearest]

    return out.reshape(shape + (n_lags,))
