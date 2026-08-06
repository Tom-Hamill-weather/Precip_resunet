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
import math
import numpy as np
from netCDF4 import Dataset
from dateutils import daterange, dateshift, splitdate, dayofyear
from datetime import datetime

# =========================================================================
# Adjustable parameters
# =========================================================================

aconst = 0.01   # floor for sampling weights: psamp = aconst + (1-aconst)*pnonzero
nsamps = 2500   # target number of samples per init time, POST spatial-thinning
                # (was 8000, 2026-08-04; reduced 2026-08-05 -- diagnosed as the
                # cause of training-set redundancy: 8000 samples/date from only
                # ~336 unique synoptic snapshots let the MLP overfit within
                # epoch 1 at every lead. See BLOCK_SIZE/MAX_PER_BLOCK below for
                # the complementary spatial-thinning fix.)
Q      = 0.6    # minimum mean MRMS quality to include a grid point

# Spatial-thinning (2026-08-05): weighted sampling alone can draw many
# points from the same storm system, which are highly spatially correlated
# and add little independent information despite counting as distinct
# samples. Oversample, then greedily accept in descending-weight order with
# a cap on how many points may land in the same spatial block.
BLOCK_SIZE        = 15    # grid cells per block (~15 * 0.043 deg ~ 72 km,
                          # comparable to the 60-km terrain-smoothing scale)
MAX_PER_BLOCK     = 2     # max accepted samples per block per date
OVERSAMPLE_FACTOR = 6.0   # candidate pool size relative to nsamps, pre-thinning
                          # (2.5x left ~40% of dates well short of the nsamps
                          # target on quiet days -- 6x closes most of that gap
                          # at negligible extra cost, still I/O-bound overall)

# Extreme-precipitation boost (2026-08-04): psamp additionally rewards
# pixels with a high *expected* 6-h total (not just nonzero probability),
# so heavy-precip forecasts are oversampled regardless of terrain --
# terrain itself enters the MLP as a separate local_std feature, not via
# sampling.
#   amt_score = expected_6h / (expected_6h + SAT_MM)   (saturates toward 1)
#   psamp     = aconst + (1-aconst)*pnonzero + BETA*amt_score
BETA   = 8.0    # weight on the extreme-amount boost relative to pnonzero
SAT_MM = 20.0   # expected_6h (mm) at which amt_score = 0.5
# BETA/SAT_MM doubled 2026-08-04 (were 4.0/10.0) after Tom asked for more
# emphasis on heavier cases. Verified via population-weighted analysis
# against 22 representative dates (lead24h): (4,10)->(8,20) raises the
# post-resample fraction exceeding 10mm from 0.147 to 0.164 (+11% relative)
# and exceeding 25mm from 0.031 to 0.036 (+16% relative), vs. domain
# climatology of ~0.020 and ~0.004 respectively -- a further deliberate
# step beyond the already-large oversampling factor, without going as far
# as the most aggressive candidate tested (BETA=12, SAT_MM=15, which gave
# similar gains with a less interpretable parameter combination).

TERRAIN_MASK_NC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'terrain_roughness_mask_graf.nc')


def load_local_std():
    """Static (ny, nx) local terrain-roughness field (60-km-smoothed local
    std of elevation), same grid as GRAF/MRMS. See terrain_roughness_graf.py."""
    with Dataset(TERRAIN_MASK_NC, 'r') as ds:
        return np.asarray(ds.variables['local_std'][:], dtype=np.float32)


def julian_features(cyyyymmddhh):
    """Cyclic day-of-year encoding: cos/sin(2*pi*julian_day/365)."""
    yyyy, mm, dd, hh = splitdate(cyyyymmddhh)
    doy = dayofyear(yyyy, mm, dd)
    angle = 2.0 * math.pi * doy / 365.0
    return math.cos(angle), math.sin(angle)

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
    cyyyymm = yyyymmddhh[0:6]
    fname = os.path.join(mrms_dir, cyyyymm,
                         f'MRMS_1h_pamt_and_data_qual_{yyyymmddhh}.nc')
    if not os.path.exists(fname):
        print(f'  WARNING: MRMS file not found: {fname}')
        return None
    try:
        with Dataset(fname, 'r') as ds:
            precip  = np.ma.filled(ds['precipitation'][:], 0.0).astype(np.float32)
            quality = np.ma.filled(ds['data_quality'][:],  0.0).astype(np.float32)
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
    sample_date  = np.concatenate(data['sample_date'],        axis=0)
    sample_cos_doy = np.concatenate(data['sample_cos_doy'],    axis=0)
    sample_sin_doy = np.concatenate(data['sample_sin_doy'],    axis=0)
    sample_local_std = np.concatenate(data['sample_local_std'], axis=0)

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

        def _write1d_int(name, arr, units, long_name):
            v = nc.createVariable(name, 'i4', ('nsamples',),
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
        _write1d_int('sample_date', sample_date,
                 '1', 'Initial condition date/time, YYYYMMDDHH')
        _write1d('sample_cos_doy', sample_cos_doy,
                 '1', 'cos(2*pi*julian_day/365) of the init date')
        _write1d('sample_sin_doy', sample_sin_doy,
                 '1', 'sin(2*pi*julian_day/365) of the init date')
        _write1d('sample_local_std', sample_local_std,
                 'm', 'Local terrain-roughness (60-km-smoothed local std of '
                      'elevation) at the sampled grid point')

        # Global attributes
        nc.clead         = int(clead)
        nc.aconst        = float(aconst)
        nc.beta          = float(BETA)
        nc.sat_mm        = float(SAT_MM)
        nc.nsamps_target = int(nsamps)
        nc.block_size    = int(BLOCK_SIZE)
        nc.max_per_block = int(MAX_PER_BLOCK)
        nc.oversample_factor = float(OVERSAMPLE_FACTOR)
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
    print(f'Date range: {yyyymmddhh_start} to {yyyymmddhh_end} (6-h stride, all 4 cycles)')
    print(f'Parameters: aconst={aconst}, BETA={BETA}, SAT_MM={SAT_MM}, '
          f'nsamps={nsamps}, Q={Q}')
    print()

    local_std = load_local_std()

    # Six lead times per init time: (clead-5) ... clead
    lead_offsets = list(range(-5, 1))   # [-5, -4, -3, -2, -1, 0]

    # Accumulate samples keyed by calendar month of the init time
    monthly_data = {}

    # 6-h stride (2026-08-04): pick up all 4 GRAF cycles (00/06/12/18Z),
    # not just 00/12Z, so training uses more independent synoptic snapshots
    # per calendar day rather than just denser pixel sampling of the same
    # two cycles.
    datelist = daterange(yyyymmddhh_start, yyyymmddhh_end, 6)
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

        # Extreme-amount boost: forecast's own expected 6-h total, from the
        # six hourly zero-inflated 2-component Gamma-mixture means.
        mean_hour   = ((1.0 - frac_zero_6)
                       * (mix_weight_6 * gshape1_6 * gscale1_6
                          + (1.0 - mix_weight_6) * gshape2_6 * gscale2_6))
        expected_6h = mean_hour.sum(axis=0)               # (ny, nx)
        amt_score   = expected_6h / (expected_6h + SAT_MM)

        psamp = aconst + (1.0 - aconst) * pnonzero + BETA * amt_score

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
        # 6. Weighted random sampling without replacement, then spatial
        #    block-cap thinning (see BLOCK_SIZE/MAX_PER_BLOCK above)
        # ------------------------------------------------------------------
        oversample_n = min(int(nsamps * OVERSAMPLE_FACTOR), n_valid)
        cand_flat = np.random.choice(ny * nx, size=oversample_n,
                                     replace=False, p=prob_norm)
        cand_i, cand_j = np.divmod(cand_flat, nx)
        cand_weight = flat_psamp[cand_flat]

        n_block_j = (nx + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_id  = (cand_i // BLOCK_SIZE) * n_block_j + (cand_j // BLOCK_SIZE)

        # Accept candidates in descending-weight order, capping how many
        # land in the same block, until nsamps accepted or pool exhausted.
        order = np.argsort(-cand_weight)
        block_counts = {}
        accepted = []
        for pos in order:
            b = block_id[pos]
            c = block_counts.get(b, 0)
            if c < MAX_PER_BLOCK:
                accepted.append(pos)
                block_counts[b] = c + 1
                if len(accepted) >= nsamps:
                    break

        accepted = np.asarray(accepted, dtype=np.int64)
        chosen_i, chosen_j = cand_i[accepted], cand_j[accepted]
        actual_n = len(accepted)
        if actual_n < nsamps:
            print(f'  WARNING: block-cap thinning yielded only '
                  f'{actual_n}/{nsamps} samples ({n_valid} valid points)')

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
        s_local_std    = local_std[chosen_i, chosen_j].astype(np.float32)
        s_date         = np.full(actual_n, int(yyyymmddhh), dtype=np.int32)
        cos_doy, sin_doy = julian_features(yyyymmddhh)
        s_cos_doy      = np.full(actual_n, cos_doy, dtype=np.float32)
        s_sin_doy      = np.full(actual_n, sin_doy, dtype=np.float32)

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
                'sample_date':      [],
                'sample_cos_doy':   [],
                'sample_sin_doy':   [],
                'sample_local_std': [],
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
        d['sample_date'].append(s_date)
        d['sample_cos_doy'].append(s_cos_doy)
        d['sample_sin_doy'].append(s_sin_doy)
        d['sample_local_std'].append(s_local_std)

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
