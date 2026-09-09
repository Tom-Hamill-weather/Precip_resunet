"""
generate_6h_conditional_copula_control.py <clead> [N_members]

Full-domain six-hourly exceedance-probability control using the fitted
conditional Gaussian-copula empirical multi-lag Toeplitz dependence (see
fit_conditional_copula.py and copula_common.py), evaluated over the day
10-end test window (all 12 months of 2025) -- the same test window and
same per-init/lead input files as generate_6h_independence_control.py, so
all three (MLP, naive independence control, conditional-copula control)
are directly comparable in reliability_6hourly_mlp_3panel.py.

For each grid point: if c_min_p0 > DEGENERATE_P0_THRESHOLD (no hour has a
meaningful chance of precip), dependence is irrelevant, so all 5 lag
correlations are set to 0 (-> independence). Otherwise: look up rho(lag, c)
for lag=1..5 from the fitted grid, build the per-pixel 6x6 Toeplitz
correlation matrix and PSD-clip it (an empirical multi-lag fit isn't
guaranteed PSD the way an AR(1)/AR(2)-derived one is), draw an N-member
Gaussian-copula ensemble, invert each hour's own Gamma-mixture CDF at the
correlated uniforms, sum, and estimate exceedance frequency -- same final
step as the independence control, just with correlated instead of
independent hourly draws.

Output: one compact netCDF per (init time, clead), matching the
independence-control format:
    probs_control/{cyyyymmddhh}_{clead}_copula_ensemble_probs.nc

Tom Hamill, Jul 2026
"""
import os
import sys
import numpy as np
import torch

from generate_6h_independence_control import (
    get_paths, read_prob_params_6h, PARAM_VARS, THRESHOLDS,
    VARNAME_BY_THRESH, write_control_file, build_test_datelist,
)
from copula_common import (
    compute_covariates, is_degenerate, load_copula_grid, lookup_rho_multilag,
    build_toeplitz_from_lag_corrs, psd_clip_correlation,
    sample_correlated_uniforms_from_corr, invert_zig_mixture, EPS,
)

SAMPLE_BATCH = 262144   # matches the independence control's batch; measured
                         # ~14GB free headroom at this size with 50 members


def build_copula_ensemble(params_6h, ny, nx, N, device, grid):
    npix = ny * nx
    flat = {k: params_6h[k].reshape(6, npix) for k in PARAM_VARS}

    c1, c2, c3 = compute_covariates(
        params_6h['fraction_zero'], params_6h['mixture_weight'],
        params_6h['gamma_shape1'], params_6h['gamma_scale1'],
        params_6h['gamma_shape2'], params_6h['gamma_scale2'])
    c1f, c2f, c3f = c1.reshape(npix), c2.reshape(npix), c3.reshape(npix)
    degenerate = is_degenerate(c1f)
    rho_lags_field = lookup_rho_multilag(c1f, c2f, c3f, grid).astype(np.float32)  # (npix, 5)
    rho_lags_field[degenerate] = 0.0

    prob_fields = {t: np.empty(npix, dtype=np.float32) for t in THRESHOLDS}

    for start in range(0, npix, SAMPLE_BATCH):
        end = min(start + SAMPLE_BATCH, npix)
        b = end - start

        frac_zero_h  = torch.tensor(flat['fraction_zero'][:, start:end],  device=device)
        mix_weight_h = torch.tensor(flat['mixture_weight'][:, start:end], device=device)
        shape1_h     = torch.clamp(torch.tensor(flat['gamma_shape1'][:, start:end], device=device), min=EPS)
        scale1_h     = torch.clamp(torch.tensor(flat['gamma_scale1'][:, start:end], device=device), min=EPS)
        shape2_h     = torch.clamp(torch.tensor(flat['gamma_shape2'][:, start:end], device=device), min=EPS)
        scale2_h     = torch.clamp(torch.tensor(flat['gamma_scale2'][:, start:end], device=device), min=EPS)
        rho_lags_b   = torch.tensor(rho_lags_field[start:end], device=device)   # (b, 5)

        corr = build_toeplitz_from_lag_corrs(rho_lags_b, n=6)   # (b, 6, 6)
        corr = psd_clip_correlation(corr)
        u = sample_correlated_uniforms_from_corr(corr, N, device)   # (b, 6, N) -- rho=0 -> independent

        sixhr_sum = torch.zeros((b, N), dtype=torch.float32, device=device)
        for h in range(6):
            x_h = invert_zig_mixture(
                u[:, h, :],
                frac_zero_h[h].unsqueeze(1).expand(b, N),
                mix_weight_h[h].unsqueeze(1).expand(b, N),
                shape1_h[h].unsqueeze(1).expand(b, N),
                scale1_h[h].unsqueeze(1).expand(b, N),
                shape2_h[h].unsqueeze(1).expand(b, N),
                scale2_h[h].unsqueeze(1).expand(b, N))
            sixhr_sum += x_h

        for t in THRESHOLDS:
            prob = (sixhr_sum >= t).float().mean(dim=1)
            prob_fields[t][start:end] = prob.cpu().numpy()

    return {t: prob_fields[t].reshape(ny, nx) for t in THRESHOLDS}


def main():
    if len(sys.argv) not in (2, 3):
        print('Usage: python generate_6h_conditional_copula_control.py <clead> [N_members]')
        sys.exit(1)

    clead = int(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) == 3 else 50
    if clead < 6:
        print('ERROR: clead must be >= 6')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'generate_6h_conditional_copula_control.py  clead={clead}h  N={N}  device={device}')

    probs_dir, control_dir = get_paths()
    os.makedirs(control_dir, exist_ok=True)

    copula_dir = os.path.join(os.path.dirname(control_dir), 'copula')
    grid_path = os.path.join(copula_dir, f'copula_rho_grid_lead{clead}h.npz')
    grid = load_copula_grid(grid_path)
    print(f'Loaded copula grid: {grid_path}')

    cyyyymmddhh_list = build_test_datelist()
    ndates = len(cyyyymmddhh_list)

    nwritten = 0
    nskipped_existing = 0
    nskipped_missing = 0

    for idate, cdate in enumerate(cyyyymmddhh_list):
        out_fname = os.path.join(
            control_dir, f'{cdate}_{clead}_copula_ensemble_probs.nc')

        if os.path.exists(out_fname):
            nskipped_existing += 1
            print(f'{idate+1:4d}/{ndates}  init={cdate}  already exists, skipping')
            continue

        params_6h, lat, lon = read_prob_params_6h(probs_dir, cdate, clead)
        if params_6h is None:
            nskipped_missing += 1
            print(f'{idate+1:4d}/{ndates}  init={cdate}  hourly params missing, skipping')
            continue

        torch.manual_seed((int(cdate) * 100 + clead) % (2**31 - 1))

        ny, nx = lat.shape
        prob_fields = build_copula_ensemble(params_6h, ny, nx, N, device, grid)
        write_control_file(out_fname, prob_fields, lat, lon, N)
        nwritten += 1
        print(f'{idate+1:4d}/{ndates}  init={cdate}  wrote {out_fname}')

    print(f'\nDone.  wrote={nwritten}  skipped_existing={nskipped_existing}  '
          f'skipped_missing={nskipped_missing}  total={ndates}')


if __name__ == '__main__':
    main()
