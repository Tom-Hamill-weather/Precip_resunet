"""
fit_conditional_copula.py <clead>

Fits the conditional Gaussian-copula dependence structure as an empirical
multi-lag Toeplitz correlation matrix -- rho(lag, c) for lag=1..5,
conditioned on the 3 forecast covariates in copula_common.py -- using the
day 1-7 training window (all 12 months of 2025, 00Z/12Z) -- disjoint from
the day 10-end test window used for evaluation in
reliability_6hourly_mlp_3panel.py.

Method
------
For each sampled (date, grid point), the 6 observed hourly MRMS values are
mapped to standard-normal scores via a randomized probability-integral
transform through that hour's own forecast CDF (pit_to_normal_score). If
the resulting z-scores were jointly Gaussian, the Pearson correlation
between hours lag apart directly estimates rho(lag) -- no rank/tie
handling needed, since the randomized PIT already removed the point-mass
ties at zero.

rho(lag, c) is estimated independently for each lag=1..5 at each node of a
grid spanning the observed covariate range, as a kernel-weighted (Gaussian
kernel, Euclidean distance in standardized covariate space) average
Pearson correlation, pooling all (6-lag) adjacent-by-lag hour pairs per
sample as replicate observations (assumes stationarity of the dependence
structure across the 6-hour window, conditional on the covariates). An
AR(1)-decay diagnostic run (2026-07-31, lead48h) showed AR(1) badly
underestimates persistence beyond lag 1 -- e.g. empirical lag-5 rho=0.055
vs. an AR(1)-implied 0.003 -- so lags are fit directly rather than via any
assumed ARMA order. Nodes lacking adequate local support at a given lag
fall back, at lookup time (lookup_rho_multilag() in copula_common.py), to
the pooled/unconditional correlation for that lag.

Points with c_min_p0 > DEGENERATE_P0_THRESHOLD are excluded (see
copula_common.py): dependence is irrelevant there, so they only dilute
the fit.

Output: {OUTPUT_DIR}/copula_rho_grid_lead{clead}h.npz
    grid_edges  : list of 3 arrays, node coordinates along each covariate
    rho_grid    : array, shape (MAX_LAG, n1, n2, n3)
    neff_grid   : array, same shape -- effective sample size at each
                  (lag, node) (low values mean the fit there is unreliable)
    pooled_rho  : array, shape (MAX_LAG,) -- unconditional fallback rho
                  per lag, used where no node has adequate support

Tom Hamill, Jul 2026
"""
import os
import sys
import numpy as np
from dateutils import daterange, dateshift
from sample_6hourly_prob_mrms import read_prob_file, read_mrms_file
from copula_common import compute_covariates, is_degenerate, importance_weight, pit_to_normal_score

Q = 0.6                 # MRMS quality threshold (matches rest of pipeline)
NSAMPS_PER_INIT = 2000  # matches sample_6hourly_prob_mrms.py
KERNEL_BANDWIDTH = 1.0  # in standardized-covariate units
GRID_SIZE = (4, 4, 3)   # nodes along (c_min_p0, c_max_p0, c_mean_amt)
MAX_LAG = 5             # lags 1..5 span the full 6-hour window

TRAIN_MONTHS = [
    (1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
    (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31),
]


def detect_paths():
    for base in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.isdir(base):
            return os.path.join(base, 'probs'), os.path.join(base, 'MRMS'), os.path.join(base, 'copula')
    raise RuntimeError("Cannot locate resnet_data directory.")


def build_train_datelist():
    dates = []
    for mm, ndays in TRAIN_MONTHS:
        for dd in range(1, 8):
            dates.append(f'2025{mm:02d}{dd:02d}00')
            dates.append(f'2025{mm:02d}{dd:02d}12')
    return dates


def read_hourly_stack(probs_dir, mrms_dir, cyyyymmddhh, clead):
    """Read the 6 hourly forecast + MRMS fields for this init/lead, without
    summing.  Returns None if anything is missing."""
    lead_offsets = list(range(-5, 1))
    lead_times = [clead + o for o in lead_offsets]

    keys = ['fraction_zero', 'mixture_weight', 'gamma_shape1',
            'gamma_scale1', 'gamma_shape2', 'gamma_scale2']
    stacks = {k: [] for k in keys}
    for lt in lead_times:
        r = read_prob_file(probs_dir, cyyyymmddhh, lt)
        if r is None:
            return None
        for k in keys:
            stacks[k].append(r[k])
    for k in keys:
        stacks[k] = np.stack(stacks[k], axis=0)   # (6, ny, nx)

    precip_list, qual_list = [], []
    for offset in lead_offsets:
        vt = dateshift(cyyyymmddhh, clead + offset)
        r = read_mrms_file(mrms_dir, vt)
        if r is None:
            return None
        precip, quality, lats, lons = r
        precip_list.append(precip)
        qual_list.append(quality)
    precip_6 = np.stack(precip_list, axis=0)   # (6, ny, nx)
    qual_6   = np.stack(qual_list,   axis=0)

    return stacks, precip_6, qual_6


def sample_points(mean_qual, c_min_p0, rng, n_target=NSAMPS_PER_INIT):
    """Importance-weighted point sample, excluding degenerate/low-quality points."""
    ny, nx = mean_qual.shape
    good = (mean_qual > Q) & (~is_degenerate(c_min_p0))
    weight = np.where(good, importance_weight(c_min_p0), 0.0)
    total = weight.sum()
    if total <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    flat_w = (weight / total).ravel()
    n_valid = int((flat_w > 0).sum())
    n = min(n_target, n_valid)
    chosen_flat = rng.choice(ny * nx, size=n, replace=False, p=flat_w)
    return np.divmod(chosen_flat, nx)


def weighted_pearson_grid(x_pairs, y_pairs, cov_pairs, grid_pts, bandwidth, min_support=50):
    """
    Kernel-weighted (Gaussian kernel, standardized-covariate Euclidean
    distance) Pearson correlation of (x_pairs, y_pairs) at each node in
    grid_pts, using cov_pairs as each pair's covariate location.

    Returns (rho_flat, neff_flat), same length as grid_pts; rho_flat is
    NaN at nodes with effective support (sum of kernel weights) below
    min_support.
    """
    rho_flat = np.full(len(grid_pts), np.nan)
    neff_flat = np.zeros(len(grid_pts))

    for inode, g in enumerate(grid_pts):
        dist2 = ((cov_pairs - g) ** 2).sum(axis=1)
        w = np.exp(-0.5 * dist2 / bandwidth ** 2)
        sw = w.sum()
        neff_flat[inode] = sw
        if sw < min_support:   # too little local support to trust the estimate
            continue
        mx = (w * x_pairs).sum() / sw
        my = (w * y_pairs).sum() / sw
        cov_xy = (w * (x_pairs - mx) * (y_pairs - my)).sum() / sw
        var_x  = (w * (x_pairs - mx) ** 2).sum() / sw
        var_y  = (w * (y_pairs - my) ** 2).sum() / sw
        rho_flat[inode] = cov_xy / np.sqrt(var_x * var_y + 1e-12)

    return rho_flat, neff_flat


def main():
    if len(sys.argv) != 2:
        print('Usage: python fit_conditional_copula.py <clead>')
        sys.exit(1)
    clead = int(sys.argv[1])

    probs_dir, mrms_dir, out_dir = detect_paths()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    date_list = build_train_datelist()
    print(f'Fitting conditional copula for lead {clead}h over {len(date_list)} init times')

    c1_all, c2_all, c3_all, z_all = [], [], [], []

    for idate, cdate in enumerate(date_list):
        result = read_hourly_stack(probs_dir, mrms_dir, cdate, clead)
        if result is None:
            print(f'{idate+1:4d}/{len(date_list)}  init={cdate}  missing, skipping')
            continue
        stacks, precip_6, qual_6 = result

        mean_qual = qual_6.mean(axis=0)
        c1, c2, c3 = compute_covariates(stacks['fraction_zero'], stacks['mixture_weight'],
                                         stacks['gamma_shape1'], stacks['gamma_scale1'],
                                         stacks['gamma_shape2'], stacks['gamma_scale2'])

        chosen_i, chosen_j = sample_points(mean_qual, c1, rng)
        n = len(chosen_i)
        if n == 0:
            print(f'{idate+1:4d}/{len(date_list)}  init={cdate}  no informative points')
            continue

        z = np.empty((6, n), dtype=np.float64)
        for k in range(6):
            z[k] = pit_to_normal_score(
                precip_6[k, chosen_i, chosen_j],
                stacks['fraction_zero'][k, chosen_i, chosen_j],
                stacks['mixture_weight'][k, chosen_i, chosen_j],
                stacks['gamma_shape1'][k, chosen_i, chosen_j],
                stacks['gamma_scale1'][k, chosen_i, chosen_j],
                stacks['gamma_shape2'][k, chosen_i, chosen_j],
                stacks['gamma_scale2'][k, chosen_i, chosen_j],
                rng)

        c1_all.append(c1[chosen_i, chosen_j])
        c2_all.append(c2[chosen_i, chosen_j])
        c3_all.append(c3[chosen_i, chosen_j])
        z_all.append(z.T)   # (n, 6)

        print(f'{idate+1:4d}/{len(date_list)}  init={cdate}  sampled {n} informative points')

    c1_all = np.concatenate(c1_all)
    c2_all = np.concatenate(c2_all)
    c3_all = np.concatenate(c3_all)
    z_all  = np.concatenate(z_all, axis=0)   # (N, 6)
    print(f'\nTotal informative samples: {len(c1_all):,}')

    # ------------------------------------------------------------------
    # Standardize covariates, build grid, kernel-weighted rho estimation
    # ------------------------------------------------------------------
    cov = np.stack([c1_all, c2_all, c3_all], axis=1)   # (N, 3)
    cov_mean = cov.mean(axis=0)
    cov_std  = cov.std(axis=0)
    cov_std[cov_std < 1e-8] = 1.0
    cov_z = (cov - cov_mean) / cov_std

    grid_edges = [np.linspace(cov_z[:, d].min(), cov_z[:, d].max(), GRID_SIZE[d])
                  for d in range(3)]
    mesh = np.meshgrid(*grid_edges, indexing='ij')
    grid_pts = np.stack([m.ravel() for m in mesh], axis=1)   # (n_nodes, 3)

    # ------------------------------------------------------------------
    # Empirical multi-lag Toeplitz fit: rather than assuming any ARMA
    # order, fit each lag's correlation (1..MAX_LAG) directly, both by
    # covariate node (kernel-weighted, same grid/bandwidth as before) and
    # pooled/unconditional (used as the lookup fallback wherever a node
    # lacks support -- see lookup_rho_multilag()).  Confirmed by the AR(1)
    # diagnostic run on 2026-07-31 (lead48h): AR(1) badly underestimates
    # persistence beyond lag 1 (e.g. empirical lag-5 rho=0.055 vs an AR(1)-
    # predicted 0.003), so lags are fit independently rather than imposed
    # by a decay formula.
    # ------------------------------------------------------------------
    rho_grid_all  = np.full((MAX_LAG,) + GRID_SIZE, np.nan)
    neff_grid_all = np.zeros((MAX_LAG,) + GRID_SIZE)
    pooled_rho    = np.zeros(MAX_LAG)

    print(f'{"lag":>4} {"n_pairs":>10} {"pooled_rho":>10} {"nodes_ok":>12}')
    for lag in range(1, MAX_LAG + 1):
        npairs = 6 - lag
        x_l = z_all[:, :npairs].reshape(-1)
        y_l = z_all[:, lag:].reshape(-1)
        cov_l = np.tile(cov_z, (npairs, 1))

        pooled_rho[lag - 1] = np.corrcoef(x_l, y_l)[0, 1]

        rho_l_flat, neff_l_flat = weighted_pearson_grid(x_l, y_l, cov_l, grid_pts,
                                                          KERNEL_BANDWIDTH)
        rho_grid_all[lag - 1]  = rho_l_flat.reshape(GRID_SIZE)
        neff_grid_all[lag - 1] = neff_l_flat.reshape(GRID_SIZE)
        n_ok = int(np.isfinite(rho_grid_all[lag - 1]).sum())
        print(f'{lag:4d} {len(x_l):10d} {pooled_rho[lag - 1]:10.4f} '
              f'{n_ok:5d}/{rho_grid_all[lag - 1].size}')

    print('\n=== reference only: empirical decay vs. what AR(1) (fit from lag-1 alone)'
          ' would have predicted ===')
    rho1_pooled = pooled_rho[0]
    print(f'{"lag":>4} {"pooled_emp":>10} {"AR1_pred":>10} {"emp-AR1":>10}')
    for lag in range(1, MAX_LAG + 1):
        pred = rho1_pooled ** lag
        print(f'{lag:4d} {pooled_rho[lag - 1]:10.4f} {pred:10.4f} '
              f'{pooled_rho[lag - 1] - pred:10.4f}')

    out_fname = os.path.join(out_dir, f'copula_rho_grid_lead{clead}h.npz')
    np.savez(out_fname,
             grid_edge0=grid_edges[0], grid_edge1=grid_edges[1], grid_edge2=grid_edges[2],
             rho_grid=rho_grid_all, neff_grid=neff_grid_all, pooled_rho=pooled_rho,
             cov_mean=cov_mean, cov_std=cov_std)
    print(f'\nSaved: {out_fname}  (empirical multi-lag Toeplitz fit, lags 1-{MAX_LAG})')


if __name__ == '__main__':
    main()
