"""
python save_graf_texture_features.py YYYYMMDDHH_start YYYYMMDDHH_end clead

Precompute 7x7-window spatial texture statistics (per hour) and cross-hour
temporal persistence statistics (per consecutive-hour pair), plus the raw
GRAF 6-h precipitation total, from the raw GRAF hourly APCP fields -- for
use as additional 6-hourly MLP input features (see train_6hourly_mlp.py).

Computed from the RAW GRAF field (not the fitted gamma-mixture parameters),
so the MLP gets information about spatial texture and cross-hour coherence
that the per-pixel marginal gamma-mixture fit doesn't carry -- motivated by
a reliability-diagram underconfidence bias in the 6-hourly MLP not present
in the 1-hour ResUNet (same loss, richer raw-field CNN inputs), hypothesized
to be an information bottleneck in the MLP's compressed feature set.

Output: one netCDF per (init date, clead), covering all 6 hours needed for
that lead's 6-hour window:
    {output_dir}/{yyyymmddhh}_{clead}_graf_texture_features.nc

Tom Hamill, Aug 2026
"""

import sys
import os
import numpy as np
from scipy import ndimage
from netCDF4 import Dataset
from dateutils import daterange, dateshift

WINDOW_SIZE   = 15     # pixels, spatial/temporal texture window (widened from
                       # 7 -- 2026-08: 7x7 (~28km at 4km resolution) showed no
                       # calibration benefit; 15x15 (~60km) matches this repo's
                       # existing terrain-smoothing/spatial-thinning scale
                       # (BLOCK_SIZE=15 in sample_6hourly_prob_mrms.py), closer
                       # to synoptic coherence scale than pixel-level texture)
WET_THRESH_MM = 0.1    # mm, "measurably wet" threshold for wet-area-fraction/Jaccard
EPS_MEAN      = 1e-4   # mm, denominator floor for ratios
EPS_STD       = 1e-4   # mm, denominator floor for z-scoring
EPS_COUNT     = 1e-4   # pixel-count floor for Jaccard union

# =========================================================================
# Paths
# =========================================================================

def detect_paths():
    """Return (graf_dir_new, graf_dir_old, output_dir) for current host."""
    for base in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.isdir(base):
            return (
                os.path.join(base, 'GRAF', 'hdo-graf_conus') + os.sep,
                os.path.join(base, 'GRAF', 'hdo-graflr_conus') + os.sep,
                os.path.join(base, 'graf_texture'),
            )
    raise RuntimeError("Cannot locate resnet_data directory. "
                       "Expected /data/resnet_data or /data2/resnet_data.")

# =========================================================================
# Raw GRAF GRIB2 reading (adapted, not imported, from make_plots_6hourly_mlp.py
# -- that module parses sys.argv at import time and is unsafe to import)
# =========================================================================

def read_gribdata(gribfilename, endStep):
    import pygrib
    if not os.path.exists(gribfilename):
        return -1, None
    try:
        fcstfile = pygrib.open(gribfilename)
        grb = fcstfile.select(endStep=endStep)[0]
        precipitation = np.where(grb.values < 0., 0., grb.values)
        fcstfile.close()
        return 0, precipitation
    except Exception as e:
        print(f'  Error reading {gribfilename}: {e}')
        return -1, None


def graf_grib_path(lead, cyyyymmddhh, graf_dir_new, graf_dir_old):
    """Return the file path for the 1-h APCP GRAF grib at the given lead."""
    cyyyymmdd = cyyyymmddhh[0:8]
    chh       = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, lead)
    cyyyymmdd_fcst   = cyyyymmddhh_fcst[0:8]
    chh_fcst         = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) >= 2024040100:
        input_directory = graf_dir_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = graf_dir_old
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = (prefix + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.'
                  + cyyyymmdd + 'T' + chh + '0000Z.PT' + str(lead)
                  + 'H.CONUS@4km.APCP.SFC.grb2')
    return input_directory + input_file


def GRAF_6h_hourly_read(clead, cyyyymmddhh, graf_dir_new, graf_dir_old):
    """
    Read the 6 individual 1-h GRAF APCP fields (leads clead-5..clead).
    Returns (istat, list_of_6_precip_fields) -- istat=-1 if any file is
    missing. Unlike GRAF_6h_precip_read (make_plots_6hourly_mlp.py), keeps
    the hourly fields separate rather than summing them, since the texture/
    persistence statistics need per-hour spatial fields, not just the total.
    """
    lead_times = list(range(clead - 5, clead + 1))
    fields = []
    for lt in lead_times:
        fpath = graf_grib_path(lt, cyyyymmddhh, graf_dir_new, graf_dir_old)
        istat, precip = read_gribdata(fpath, lt)
        if istat != 0:
            return -1, None
        fields.append(np.where(precip > 200., 200., precip).astype(np.float32))
    return 0, fields

# =========================================================================
# Windowed spatial statistics (per hour)
# =========================================================================

def windowed_spatial_stats(precip_h, size=WINDOW_SIZE):
    """
    7x7-window spatial texture statistics for one hour's raw GRAF field.
    Returns (wet_area_fraction, peak_to_mean_ratio, coeff_variation,
    mean_h, std_h) -- mean_h/std_h are returned too since
    windowed_temporal_stats() reuses them for z-scoring, avoiding
    recomputation.
    """
    mean_h   = ndimage.uniform_filter(precip_h, size=size, mode='reflect')
    meansq_h = ndimage.uniform_filter(precip_h**2, size=size, mode='reflect')
    var_h    = np.clip(meansq_h - mean_h**2, 0.0, None)
    std_h    = np.sqrt(var_h)

    wet_h = (precip_h > WET_THRESH_MM).astype(np.float32)
    wet_area_fraction = ndimage.uniform_filter(wet_h, size=size, mode='reflect')

    max_h = ndimage.maximum_filter(precip_h, size=size, mode='reflect')
    peak_to_mean_ratio = max_h / np.maximum(mean_h, EPS_MEAN)
    coeff_variation    = std_h / np.maximum(mean_h, EPS_MEAN)

    return wet_area_fraction, peak_to_mean_ratio, coeff_variation, mean_h, std_h

# =========================================================================
# Windowed temporal statistics (per consecutive-hour pair)
# =========================================================================

def windowed_temporal_stats(precip_h1, mean_h1, std_h1,
                            precip_h2, mean_h2, std_h2, size=WINDOW_SIZE):
    """
    7x7-window cross-hour persistence statistics for one consecutive pair.
    Returns (wetdry_jaccard, zscore_pattern_corr).

    wetdry_jaccard: binary wet/dry overlap of the two hours' windows --
    structural, doesn't touch amount magnitude, so it's not confounded by
    precip intensity regime the way a raw-value correlation would be.

    zscore_pattern_corr: each hour's window is first z-scored by its own
    local mean/std (removing both the amount level and the amount spread),
    then the two z-scored fields' windowed correlation is computed -- this
    compares relative spatial pattern shape, not absolute amounts, for the
    same reason.
    """
    wet1 = precip_h1 > WET_THRESH_MM
    wet2 = precip_h2 > WET_THRESH_MM
    npix = float(size * size)
    inter = ndimage.uniform_filter((wet1 & wet2).astype(np.float32), size, mode='reflect') * npix
    union = ndimage.uniform_filter((wet1 | wet2).astype(np.float32), size, mode='reflect') * npix
    wetdry_jaccard = inter / np.maximum(union, EPS_COUNT)

    z1 = (precip_h1 - mean_h1) / np.maximum(std_h1, EPS_STD)
    z2 = (precip_h2 - mean_h2) / np.maximum(std_h2, EPS_STD)
    mz1 = ndimage.uniform_filter(z1, size, mode='reflect')
    mz2 = ndimage.uniform_filter(z2, size, mode='reflect')
    cov  = ndimage.uniform_filter(z1 * z2, size, mode='reflect') - mz1 * mz2
    var1 = np.maximum(ndimage.uniform_filter(z1**2, size, mode='reflect') - mz1**2, EPS_STD)
    var2 = np.maximum(ndimage.uniform_filter(z2**2, size, mode='reflect') - mz2**2, EPS_STD)
    zscore_pattern_corr = np.clip(cov / np.sqrt(var1 * var2), -1.0, 1.0)

    return wetdry_jaccard.astype(np.float32), zscore_pattern_corr.astype(np.float32)

# =========================================================================
# NetCDF output
# =========================================================================

def write_texture_netcdf(output_dir, cyyyymmddhh, clead,
                         wet_area_fraction_6, peak_to_mean_6, coeff_var_6,
                         jaccard_5, pattern_corr_5, precip_6h_total):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f'{cyyyymmddhh}_{clead}_graf_texture_features.nc')

    ny, nx = precip_6h_total.shape
    with Dataset(fname, 'w', format='NETCDF4') as nc:
        nc.createDimension('ny', ny)
        nc.createDimension('nx', nx)
        nc.createDimension('nlead_times', 6)
        nc.createDimension('npairs', 5)

        def _write3d(name, arr, dimname, units, long_name, lsd=3):
            v = nc.createVariable(name, 'f4', (dimname, 'ny', 'nx'),
                                  zlib=True, complevel=4,
                                  least_significant_digit=lsd)
            v.units = units
            v.long_name = long_name
            v[:] = arr

        _write3d('wet_area_fraction', wet_area_fraction_6, 'nlead_times',
                 '1', f'Fraction of {WINDOW_SIZE}x{WINDOW_SIZE} window > {WET_THRESH_MM}mm, per hour')
        _write3d('peak_to_mean_ratio', peak_to_mean_6, 'nlead_times',
                 '1', f'Max/mean of {WINDOW_SIZE}x{WINDOW_SIZE} window, per hour')
        _write3d('coeff_variation', coeff_var_6, 'nlead_times',
                 '1', f'Std/mean of {WINDOW_SIZE}x{WINDOW_SIZE} window, per hour')
        _write3d('wetdry_jaccard', jaccard_5, 'npairs',
                 '1', f'Wet/dry ({WET_THRESH_MM}mm) Jaccard overlap between consecutive hours')
        _write3d('zscore_pattern_corr', pattern_corr_5, 'npairs',
                 '1', 'Windowed correlation of per-hour z-scored patterns, consecutive hours')

        v = nc.createVariable('precip_6h_total', 'f4', ('ny', 'nx'),
                              zlib=True, complevel=4, least_significant_digit=2)
        v.units = 'mm'
        v.long_name = 'Raw GRAF 6-h precipitation total (sum of 6 hourly fields)'
        v[:] = precip_6h_total

        nc.clead            = int(clead)
        nc.window_size       = WINDOW_SIZE
        nc.wet_threshold_mm  = WET_THRESH_MM
        nc.lead_times        = str(list(range(clead - 5, clead + 1)))
        nc.history           = f'Created by save_graf_texture_features.py for init time {cyyyymmddhh}'
        nc.description       = ('7x7-window spatial texture (per hour) and cross-hour '
                                'temporal persistence (per consecutive pair) statistics, '
                                'computed from the raw GRAF field, for 6-hourly MLP input '
                                'feature engineering.')

    print(f'  Wrote {fname}')

# =========================================================================
# Main
# =========================================================================

def main():
    if len(sys.argv) != 4:
        print('Usage: python save_graf_texture_features.py '
              'YYYYMMDDHH_start YYYYMMDDHH_end clead')
        sys.exit(1)

    yyyymmddhh_start = sys.argv[1]
    yyyymmddhh_end   = sys.argv[2]
    clead            = int(sys.argv[3])

    if clead < 5:
        raise ValueError(f'clead must be >= 5 to allow 6 consecutive lead times '
                         f'(got clead={clead})')

    graf_dir_new, graf_dir_old, output_dir = detect_paths()
    os.makedirs(output_dir, exist_ok=True)
    print(f'GRAF dir (new): {graf_dir_new}')
    print(f'GRAF dir (old): {graf_dir_old}')
    print(f'Output dir:     {output_dir}')
    print(f'Lead time:      {clead} h')
    print(f'Date range:     {yyyymmddhh_start} to {yyyymmddhh_end} (6-h stride, all 4 cycles)')

    datelist = daterange(yyyymmddhh_start, yyyymmddhh_end, 6)
    nprocessed = 0
    nskipped_exists = 0
    nskipped_missing = 0

    for cyyyymmddhh in datelist:
        out_fname = os.path.join(output_dir, f'{cyyyymmddhh}_{clead}_graf_texture_features.nc')
        if os.path.exists(out_fname):
            nskipped_exists += 1
            continue

        istat, fields = GRAF_6h_hourly_read(clead, cyyyymmddhh, graf_dir_new, graf_dir_old)
        if istat != 0:
            print(f'  Skipping {cyyyymmddhh}: missing GRAF file(s) for clead={clead}')
            nskipped_missing += 1
            continue

        # Per-hour spatial stats (also keep mean_h/std_h for temporal reuse)
        wet_area_fraction_list, peak_to_mean_list, coeff_var_list = [], [], []
        means, stds = [], []
        for precip_h in fields:
            waf, p2m, cv, mean_h, std_h = windowed_spatial_stats(precip_h)
            wet_area_fraction_list.append(waf)
            peak_to_mean_list.append(p2m)
            coeff_var_list.append(cv)
            means.append(mean_h)
            stds.append(std_h)

        # Per-consecutive-pair temporal stats
        jaccard_list, pattern_corr_list = [], []
        for i in range(5):
            jac, pcorr = windowed_temporal_stats(
                fields[i], means[i], stds[i], fields[i + 1], means[i + 1], stds[i + 1])
            jaccard_list.append(jac)
            pattern_corr_list.append(pcorr)

        precip_6h_total = np.sum(np.stack(fields, axis=0), axis=0).astype(np.float32)

        write_texture_netcdf(
            output_dir, cyyyymmddhh, clead,
            np.stack(wet_area_fraction_list, axis=0),
            np.stack(peak_to_mean_list, axis=0),
            np.stack(coeff_var_list, axis=0),
            np.stack(jaccard_list, axis=0),
            np.stack(pattern_corr_list, axis=0),
            precip_6h_total)
        nprocessed += 1

    print(f'\nDone: {nprocessed} processed, {nskipped_exists} already existed, '
          f'{nskipped_missing} missing GRAF data.')


if __name__ == '__main__':
    main()
