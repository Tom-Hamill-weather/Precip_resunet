"""
generate_6h_independence_control.py — independence-assumption ensemble control
for six-hourly reliability diagrams.

Usage:
    python generate_6h_independence_control.py <clead> [N_members]

    clead     : integer lead time (hours) for the END of the 6-h window.
    N_members : ensemble size (default 50).

For each init time and each of the 6 hourly zero-inflated two-component
Gamma-mixture forecasts feeding that window, draws an N-member Monte Carlo
ensemble per grid point and sums the six hourly members together under an
assumption of temporal independence.  Six-hourly exceedance probability at
0.25, 2.5, and 10.0 mm is then estimated as ensemble relative frequency.

This is a naive "what if we ignored hour-to-hour correlation and just
summed independent draws" control, meant to be compared against the
six-hourly MLP (which learns the true aggregate distribution) in
reliability_6hourly_mlp_3panel.py.

Output: one compact netCDF per (init time, clead) written to
probs_control/{cyyyymmddhh}_{clead}_indep_ensemble_probs.nc
Existing output files are skipped, so the job is resumable.

Tom Hamill, Jul 2026
"""

import os
import sys
import numpy as np
from dateutils import daterange
from netCDF4 import Dataset
import torch

from reliability_6hourly_mlp_3panel import build_test_datelist

np.set_printoptions(precision=3, suppress=True)

# =========================================================================
# Environment detection (mirrors reliability_6hourly_mlp_3panel.py)
# =========================================================================

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            print(f'Detected AWS environment ({path})')
            return 'aws', path
    print('Detected laptop environment')
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()


def get_paths():
    if ENVIRONMENT == 'aws':
        base = AWS_BASE_PATH
    else:
        base = os.path.expanduser('~/python/resnet_data')
    return (
        os.path.join(base, 'probs'),
        os.path.join(base, 'probs_control'),
    )

# =========================================================================
# Read 6 consecutive hourly gamma-mixture parameter files
# (identical to reliability_6hourly_mlp_3panel.py::read_prob_params_6h)
# =========================================================================

PARAM_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]


def read_prob_params_6h(probs_dir, cyyyymmddhh, clead):
    lead_times = list(range(clead - 5, clead + 1))
    stacks = {k: [] for k in PARAM_VARS}
    lat = lon = None

    for lt in lead_times:
        fname = os.path.join(probs_dir,
                             f'{cyyyymmddhh}_{lt}_probs_gamma_mixture.nc')
        if not os.path.exists(fname):
            return None, None, None
        try:
            with Dataset(fname, 'r') as ds:
                for k in PARAM_VARS:
                    arr = ds.variables[k][:].data.astype(np.float32)
                    stacks[k].append(arr)
                if lat is None:
                    lat = ds.variables['lat'][:].data.astype(np.float32)
                    lon = ds.variables['lon'][:].data.astype(np.float32)
        except Exception as exc:
            print(f'  WARNING: cannot read {fname}: {exc}')
            return None, None, None

    for k in PARAM_VARS:
        stacks[k] = np.stack(stacks[k], axis=0)

    return stacks, lat, lon

# =========================================================================
# Monte Carlo independence-assumption ensemble
# =========================================================================

THRESHOLDS = [0.25, 2.5, 10.0]
SAMPLE_BATCH = 262144
EPS = 1e-7


def build_independence_ensemble(params_6h, ny, nx, N, device):
    """
    Draw an N-member ensemble of 6-h accumulations per grid point by
    summing, hour by hour, an independent zero-inflated two-component
    Gamma-mixture draw.  Returns exceedance-probability fields (one per
    threshold), each shape (ny, nx).
    """
    npix = ny * nx
    flat = {k: params_6h[k].reshape(6, npix) for k in PARAM_VARS}

    prob_fields = {t: np.empty(npix, dtype=np.float32) for t in THRESHOLDS}

    for start in range(0, npix, SAMPLE_BATCH):
        end = min(start + SAMPLE_BATCH, npix)
        b = end - start

        sixhr_sum = torch.zeros((b, N), dtype=torch.float32, device=device)

        for h in range(6):
            frac_zero  = torch.tensor(flat['fraction_zero'][h, start:end], device=device)
            mix_weight = torch.tensor(flat['mixture_weight'][h, start:end], device=device)
            shape1     = torch.clamp(torch.tensor(flat['gamma_shape1'][h, start:end], device=device), min=EPS)
            scale1     = torch.clamp(torch.tensor(flat['gamma_scale1'][h, start:end], device=device), min=EPS)
            shape2     = torch.clamp(torch.tensor(flat['gamma_shape2'][h, start:end], device=device), min=EPS)
            scale2     = torch.clamp(torch.tensor(flat['gamma_scale2'][h, start:end], device=device), min=EPS)

            gamma1 = torch.distributions.Gamma(
                shape1.unsqueeze(1).expand(b, N),
                1.0 / scale1.unsqueeze(1).expand(b, N)).sample()
            gamma2 = torch.distributions.Gamma(
                shape2.unsqueeze(1).expand(b, N),
                1.0 / scale2.unsqueeze(1).expand(b, N)).sample()

            use_comp1 = torch.rand((b, N), device=device) < mix_weight.unsqueeze(1)
            hourly_val = torch.where(use_comp1, gamma1, gamma2)

            is_zero = torch.rand((b, N), device=device) < frac_zero.unsqueeze(1)
            hourly_val = torch.where(is_zero, torch.zeros_like(hourly_val), hourly_val)

            sixhr_sum += hourly_val

        for t in THRESHOLDS:
            prob = (sixhr_sum >= t).float().mean(dim=1)
            prob_fields[t][start:end] = prob.cpu().numpy()

    return {t: prob_fields[t].reshape(ny, nx) for t in THRESHOLDS}

# =========================================================================
# netCDF output
# =========================================================================

VARNAME_BY_THRESH = {0.25: 'prob_0p25mm', 2.5: 'prob_2p5mm', 10.0: 'prob_10mm'}


def write_control_file(out_fname, prob_fields, lat, lon, N):
    ny, nx = lat.shape
    with Dataset(out_fname, 'w', format='NETCDF4') as ds:
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)
        ds.N_members = N

        v = ds.createVariable('lat', 'f4', ('y', 'x'), zlib=True)
        v[:] = lat
        v = ds.createVariable('lon', 'f4', ('y', 'x'), zlib=True)
        v[:] = lon

        for t, arr in prob_fields.items():
            v = ds.createVariable(VARNAME_BY_THRESH[t], 'f4', ('y', 'x'), zlib=True)
            v[:] = arr

# =========================================================================
# Out-of-sample test date list imported from reliability_6hourly_mlp_3panel
# (single source of truth -- was previously a stale hardcoded duplicate
# here with TEST_DAY_START=10 and 00Z/12Z-only cycles, which had drifted
# out of sync with that module's day1-9 train / day12-end test window
# and its 4-cycles/day extension).
# =========================================================================

# =========================================================================
# Main
# =========================================================================

def main():
    if len(sys.argv) not in (2, 3):
        print('Usage: python generate_6h_independence_control.py <clead> [N_members]')
        sys.exit(1)

    clead = int(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) == 3 else 50
    if clead < 6:
        print('ERROR: clead must be >= 6')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'generate_6h_independence_control.py  clead={clead}h  N={N}  device={device}')

    probs_dir, control_dir = get_paths()
    os.makedirs(control_dir, exist_ok=True)

    cyyyymmddhh_list = build_test_datelist()
    ndates = len(cyyyymmddhh_list)

    nwritten = 0
    nskipped_existing = 0
    nskipped_missing = 0

    for idate, cdate in enumerate(cyyyymmddhh_list):
        out_fname = os.path.join(
            control_dir, f'{cdate}_{clead}_indep_ensemble_probs.nc')

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
        prob_fields = build_independence_ensemble(params_6h, ny, nx, N, device)
        write_control_file(out_fname, prob_fields, lat, lon, N)
        nwritten += 1
        print(f'{idate+1:4d}/{ndates}  init={cdate}  wrote {out_fname}')

    print(f'\nDone.  wrote={nwritten}  skipped_existing={nskipped_existing}  '
          f'skipped_missing={nskipped_missing}  total={ndates}')


if __name__ == '__main__':
    main()
