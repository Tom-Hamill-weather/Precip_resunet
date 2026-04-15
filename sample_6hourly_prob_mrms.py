"""
python sample_6hourly_prob_mrms.py YYYYMMDDHH_start YYYYMMDDHH_end clead

Build a training dataset by importance-sampling 6-hourly gamma-mixture
probabilistic forecasts and matched MRMS verification data.

For each initial-condition date (12-hour stride between start and end),
six consecutive hourly probability forecast files are read (lead times
clead-5 through clead), along with the corresponding six MRMS verification
files.  Grid points are randomly sampled with probability proportional to
the likelihood of non-zero precipitation (importance sampling), restricted
to points with sufficiently high MRMS quality.

Feature data: fraction_zero, mixture_weight, gamma_shape1/2, gamma_scale1/2
              for each of the 6 lead times  →  shape (nsamples, 6) per variable
Target data:  6-hourly MRMS precipitation sum (mm)
Quality:      mean MRMS quality over the 6 verification hours

Output is written as one netCDF per calendar month of the initial condition:
    {OUTPUT_DIR}/prob_MRMS_samples_{YYYYMM}_lead{clead}h.nc

Tom Hamill, Apr 2026
"""

import sys
import os
import numpy as np
from netCDF4 import Dataset
from dateutils import daterange, dateshift, splitdate
from datetime import datetime

# =========================================================================
# Adjustable parameters
# =========================================================================

aconst = 0.05   # floor for sampling weights: psamp = aconst + (1-aconst)*pnonzero
nsamps = 2000   # target number of samples per init time
Q      = 0.6    # minimum mean MRMS quality to include a grid point

# =========================================================================
# Paths  (auto-detected: G5 GPU instance takes priority)
# =========================================================================

def detect_paths():
    """Return (probs_dir, mrms_dir, output_dir) for current host."""
    for base in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.isdir(base):
            return (
                os.path.join(base, 'probs'),
                os.path.join(base, 'MRMS'),
                os.path.join(base, 'prob_samples'),
            )
    raise RuntimeError("Cannot locate resnet_data directory. "
                       "Expected /data/resnet_data or /data2/resnet_data.")

# =========================================================================
# I/O helpers
# =========================================================================

def read_prob_file(probs_dir, yyyymmddhh, lead_time):
    """
    Read one gamma-mixture probability netCDF file.

    Returns dict of 2-D numpy arrays (ny, nx) for each of the six
    distribution parameters, or None if the file is missing/unreadable.
    """
    fname = os.path.join(probs_dir,
                         f'{yyyymmddhh}_{lead_time}_probs_gamma_mixture.nc')
    if not os.path.exists(fname):
        print(f'  WARNING: prob file not found: {fname}')
        return None
    try:
        with Dataset(fname, 'r') as ds:
            # fraction_zero and mixture_weight are stored as int16 with
            # scale_factor=0.0001; netCDF4 auto-applies the scale on read.
            frac_zero     = ds['fraction_zero'][:].data.astype(np.float32)
            mix_weight    = ds['mixture_weight'][:].data.astype(np.float32)
            gamma_shape1  = ds['gamma_shape1'][:].data.astype(np.float32)
            gamma_scale1  = ds['gamma_scale1'][:].data.astype(np.float32)
            gamma_shape2  = ds['gamma_shape2'][:].data.astype(np.float32)
            gamma_scale2  = ds['gamma_scale2'][:].data.astype(np.float32)
        return {
            'fraction_zero': frac_zero,
            'mixture_weight': mix_weight,
            'gamma_shape1': gamma_shape1,
            'gamma_scale1': gamma_scale1,
            'gamma_shape2': gamma_shape2,
            'gamma_scale2': gamma_scale2,
        }
    except Exception as exc:
        print(f'  WARNING: could not read prob file {fname}: {exc}')
        return None


def read_mrms_file(mrms_dir, yyyymmddhh):
    """
    Read one MRMS 1-h precipitation + quality netCDF file.

    Returns (precipitation, data_quality, lats, lons) as numpy arrays,
    or None if the file is missing/unreadable.
    lats and lons are only populated on the first call (caller may pass
    them as None to signal "not yet read").
    """
    fname = os.path.join(mrms_dir,
                         f'MRMS_1h_pamt_and_data_qual_{yyyymmddhh}.nc')
    if not os.path.exists(fname):
        print(f'  WARNING: MRMS file not found: {fname}')
        return None
    try:
        with Dataset(fname, 'r') as ds:
            precip  = ds['precipitation'][:].data.astype(np.float32)
            quality = ds['data_quality'][:].data.astype(np.float32)
            lats    = ds['lats'][:].data.astype(np.float32)
            lons    = ds['lons'][:].data.astype(np.float32)
        return precip, quality, lats, lons
    except Exception as exc:
        print(f'  WARNING: could not read MRMS file {fname}: {exc}')
        return None

# =========================================================================
# Output writer
# =========================================================================

def write_monthly_netcdf(output_dir, month_key, clead, data):
    """
    Write one month's accumulated samples to netCDF.

    Parameters
    ----------
    output_dir  : str
    month_key   : str  YYYYMM
    clead       : int  lead time in hours
    data        : dict  lists of per-date sample arrays
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir,
                         f'prob_MRMS_samples_{month_key}_lead{clead}h.nc')

    # Stack accumulated lists into arrays
    frac_zero    = np.concatenate(data['fraction_zero'],  axis=0)  # (N, 6)
    mix_weight   = np.concatenate(data['mixture_weight'], axis=0)
    gshape1      = np.concatenate(data['gamma_shape1'],   axis=0)
    gscale1      = np.concatenate(data['gamma_scale1'],   axis=0)
    gshape2      = np.concatenate(data['gamma_shape2'],   axis=0)
    gscale2      = np.concatenate(data['gamma_scale2'],   axis=0)
    precip_6h    = np.concatenate(data['target_precip_6h'],  axis=0)  # (N,)
    mean_quality = np.concatenate(data['mean_quality'],       axis=0)
    sample_lat   = np.concatenate(data['sample_lat'],         axis=0)
    sample_lon   = np.concatenate(data['sample_lon'],         axis=0)

    nsamples, nlead_times = frac_zero.shape

    print(f'Writing {nsamples} samples to {fname}')
    with Dataset(fname, 'w', format='NETCDF4') as nc:
        nc.createDimension('nsamples',    nsamples)
        nc.createDimension('nlead_times', nlead_times)

        def _write2d(name, arr, units, long_name):
            v = nc.createVariable(name, 'f4', ('nsamples', 'nlead_times'),
                                  zlib=True, complevel=4)
            v.units = units
            v.long_name = long_name
            v[:] = arr

        def _write1d(name, arr, units, long_name):
            v = nc.createVariable(name, 'f4', ('nsamples',),
                                  zlib=True, complevel=4)
            v.units = units
            v.long_name = long_name
            v[:] = arr

        _write2d('fraction_zero',  frac_zero,
                 '1', 'Prob of zero precip for each lead time (clead-5 to clead)')
        _write2d('mixture_weight', mix_weight,
                 '1', 'Gamma component-1 mixture weight for each lead time')
        _write2d('gamma_shape1',   gshape1,
                 '1', 'Gamma component-1 shape (alpha1) for each lead time')
        _write2d('gamma_scale1',   gscale1,
                 'mm', 'Gamma component-1 scale (theta1) for each lead time')
        _write2d('gamma_shape2',   gshape2,
                 '1', 'Gamma component-2 shape (alpha2) for each lead time')
        _write2d('gamma_scale2',   gscale2,
                 'mm', 'Gamma component-2 scale (theta2) for each lead time')

        _write1d('target_precip_6h', precip_6h,
                 'mm', '6-hourly MRMS precipitation sum')
        _write1d('mean_quality',     mean_quality,
                 '1', 'Mean MRMS data quality over the 6 verification hours')
        _write1d('sample_lat',       sample_lat,
                 'degrees_north', 'Latitude of sampled grid point')
        _write1d('sample_lon',       sample_lon,
                 'degrees_east',  'Longitude of sampled grid point')

        # Global attributes
        nc.clead         = int(clead)
        nc.aconst        = float(aconst)
        nc.nsamps_target = int(nsamps)
        nc.Q_threshold   = float(Q)
        nc.month         = month_key
        nc.lead_times    = f'clead-5 through clead ({clead-5} to {clead} h)'
        nc.history       = (f'Created {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} '
                            f'by sample_6hourly_prob_mrms.py')
        nc.description   = ('Importance-sampled 6-hourly gamma mixture probabilistic '
                            'forecasts and matched MRMS verification for ML training.')

    print(f'  Done.')

# =========================================================================
# Main
# =========================================================================

def main():
    if len(sys.argv) != 4:
        print('Usage: python sample_6hourly_prob_mrms.py '
              'YYYYMMDDHH_start YYYYMMDDHH_end clead')
        sys.exit(1)

    yyyymmddhh_start = sys.argv[1]
    yyyymmddhh_end   = sys.argv[2]
    clead            = int(sys.argv[3])

    if clead < 5:
        raise ValueError(f'clead must be >= 5 to allow 6 consecutive lead times '
                         f'(got clead={clead})')

    probs_dir, mrms_dir, output_dir = detect_paths()
    print(f'Probs dir:  {probs_dir}')
    print(f'MRMS dir:   {mrms_dir}')
    print(f'Output dir: {output_dir}')
    print(f'Lead time:  {clead} h')
    print(f'Date range: {yyyymmddhh_start} to {yyyymmddhh_end} (12-h stride)')
    print(f'Parameters: aconst={aconst}, nsamps={nsamps}, Q={Q}')
    print()

    # Six lead times per init time: (clead-5) ... clead
    lead_offsets = list(range(-5, 1))   # [-5, -4, -3, -2, -1, 0]

    # Accumulate samples keyed by calendar month of the init time
    monthly_data = {}

    datelist = daterange(yyyymmddhh_start, yyyymmddhh_end, 12)
    lats_grid = None   # read once from the first successful MRMS file
    lons_grid = None

    for yyyymmddhh in datelist:
        yyyy, mm, dd, hh = splitdate(yyyymmddhh)
        month_key = f'{yyyy:04d}{mm:02d}'
        print(f'--- Processing {yyyymmddhh} ---')

        # ------------------------------------------------------------------
        # 1. Read 6 probability forecast files
        # ------------------------------------------------------------------
        lead_times = [clead + offset for offset in lead_offsets]  # clead-5 .. clead

        prob_params = {k: [] for k in ('fraction_zero', 'mixture_weight',
                                        'gamma_shape1', 'gamma_scale1',
                                        'gamma_shape2', 'gamma_scale2')}
        all_prob_ok = True
        for lt in lead_times:
            result = read_prob_file(probs_dir, yyyymmddhh, lt)
            if result is None:
                print(f'  Skipping {yyyymmddhh}: missing prob file for lead {lt}')
                all_prob_ok = False
                break
            for k in prob_params:
                prob_params[k].append(result[k])

        if not all_prob_ok:
            continue

        # Stack → shape (6, ny, nx), ordered clead-5 to clead
        frac_zero_6  = np.stack(prob_params['fraction_zero'],  axis=0)
        mix_weight_6 = np.stack(prob_params['mixture_weight'], axis=0)
        gshape1_6    = np.stack(prob_params['gamma_shape1'],   axis=0)
        gscale1_6    = np.stack(prob_params['gamma_scale1'],   axis=0)
        gshape2_6    = np.stack(prob_params['gamma_shape2'],   axis=0)
        gscale2_6    = np.stack(prob_params['gamma_scale2'],   axis=0)

        ny, nx = frac_zero_6.shape[1], frac_zero_6.shape[2]

        # ------------------------------------------------------------------
        # 2. Read 6 MRMS verification files
        # ------------------------------------------------------------------
        # Verification times: shift the init time by (clead+offset) for each slot
        verif_times = [dateshift(yyyymmddhh, clead + offset)
                       for offset in lead_offsets]   # VV-5 .. VV

        precip_list  = []
        quality_list = []
        all_mrms_ok  = True

        for vt in verif_times:
            result = read_mrms_file(mrms_dir, vt)
            if result is None:
                print(f'  Skipping {yyyymmddhh}: missing MRMS file for {vt}')
                all_mrms_ok = False
                break
            precip, quality, lats, lons = result
            precip_list.append(precip)
            quality_list.append(quality)
            if lats_grid is None:
                lats_grid = lats
                lons_grid = lons

        if not all_mrms_ok:
            continue

        # Stack → shape (6, ny, nx)
        precip_stack  = np.stack(precip_list,  axis=0)
        quality_stack = np.stack(quality_list, axis=0)

        # ------------------------------------------------------------------
        # 3. Compute target and mean quality
        # ------------------------------------------------------------------
        precip_6h  = precip_stack.sum(axis=0)    # (ny, nx)  6-h precip sum
        mean_qual  = quality_stack.mean(axis=0)  # (ny, nx)

        # ------------------------------------------------------------------
        # 4. Importance sampling weights
        # ------------------------------------------------------------------
        # Use the hour with the *lowest* fraction_zero (= most likely wet)
        min_frac_zero = frac_zero_6.min(axis=0)          # (ny, nx)
        pnonzero      = 1.0 - min_frac_zero               # max pnonzero across 6 hrs
        psamp         = aconst + (1.0 - aconst) * pnonzero  # ∈ [aconst, 1]

        # ------------------------------------------------------------------
        # 5. Quality mask — exclude low-quality points
        # ------------------------------------------------------------------
        bad_quality = (mean_qual < Q)
        psamp[bad_quality] = 0.0

        flat_psamp   = psamp.ravel()
        total_weight = flat_psamp.sum()

        if total_weight <= 0:
            print(f'  WARNING: no valid grid points for {yyyymmddhh}. Skipping.')
            continue

        prob_norm = flat_psamp / total_weight
        n_valid   = int((flat_psamp > 0).sum())

        # ------------------------------------------------------------------
        # 6. Weighted random sampling without replacement
        # ------------------------------------------------------------------
        actual_n = min(nsamps, n_valid)
        if actual_n < nsamps:
            print(f'  WARNING: only {n_valid} valid points; sampling {actual_n}')

        chosen_flat = np.random.choice(ny * nx, size=actual_n,
                                       replace=False, p=prob_norm)
        chosen_i, chosen_j = np.divmod(chosen_flat, nx)

        # ------------------------------------------------------------------
        # 7. Extract samples
        # ------------------------------------------------------------------
        # Feature arrays: (actual_n, 6) — lead-time axis ordered clead-5..clead
        def _extract(arr6):
            """arr6 shape (6, ny, nx) → (actual_n, 6)"""
            return arr6[:, chosen_i, chosen_j].T.astype(np.float32)

        s_frac_zero  = _extract(frac_zero_6)
        s_mix_weight = _extract(mix_weight_6)
        s_gshape1    = _extract(gshape1_6)
        s_gscale1    = _extract(gscale1_6)
        s_gshape2    = _extract(gshape2_6)
        s_gscale2    = _extract(gscale2_6)

        s_precip_6h    = precip_6h[chosen_i, chosen_j].astype(np.float32)
        s_mean_quality = mean_qual[chosen_i, chosen_j].astype(np.float32)
        s_lat          = lats_grid[chosen_i, chosen_j].astype(np.float32)
        s_lon          = lons_grid[chosen_i, chosen_j].astype(np.float32)

        print(f'  Sampled {actual_n} points; '
              f'mean precip={s_precip_6h.mean():.2f} mm, '
              f'min quality={s_mean_quality.min():.3f}')

        # ------------------------------------------------------------------
        # 8. Accumulate into monthly dict
        # ------------------------------------------------------------------
        if month_key not in monthly_data:
            monthly_data[month_key] = {
                'fraction_zero':  [],
                'mixture_weight': [],
                'gamma_shape1':   [],
                'gamma_scale1':   [],
                'gamma_shape2':   [],
                'gamma_scale2':   [],
                'target_precip_6h': [],
                'mean_quality':     [],
                'sample_lat':       [],
                'sample_lon':       [],
            }

        d = monthly_data[month_key]
        d['fraction_zero'].append(s_frac_zero)
        d['mixture_weight'].append(s_mix_weight)
        d['gamma_shape1'].append(s_gshape1)
        d['gamma_scale1'].append(s_gscale1)
        d['gamma_shape2'].append(s_gshape2)
        d['gamma_scale2'].append(s_gscale2)
        d['target_precip_6h'].append(s_precip_6h)
        d['mean_quality'].append(s_mean_quality)
        d['sample_lat'].append(s_lat)
        d['sample_lon'].append(s_lon)

    # ======================================================================
    # 9. Write one netCDF file per calendar month
    # ======================================================================
    if not monthly_data:
        print('No data accumulated — check that input files exist.')
        return

    for month_key, data in sorted(monthly_data.items()):
        write_monthly_netcdf(output_dir, month_key, clead, data)


if __name__ == '__main__':
    main()
