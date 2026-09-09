"""
python blob_size_distribution.py

Diagnostic requested by Tom (2026-09-01 session): compare the spatial-
organization of precipitation features between 2025 and 2026, holding
time-of-year fixed (Jan/Apr/Jun, matching the reliability comparison
already done), to test whether GRAF's raw high-confidence heavy-precip
bin collapse (see reliability_resunet_mixture.py 2025h1 vs 2026h1 runs)
reflects a shift toward more scattered / less organized (MCS-like)
precipitation structure rather than a change in per-pixel intensity.

Method: threshold each field, label 8-connected contiguous blobs, and
compare blob-size (pixel count) distributions across years -- separately
for GRAF raw forecast precip (mm) and MRMS observed precip (mm), at each
of 5 thresholds, at 24h lead.

Blobs touching the array boundary, or (for MRMS) touching a bad-quality
pixel, have their true extent unknown (truncated), so they are excluded
from the size statistic -- standard practice in object-based precip
verification (e.g. MODE). A blob is excluded if any of its pixels lies
on the domain edge, or (MRMS only) within 1 pixel of a quality<=0.5 pixel.
"""
import os
import numpy as np
import pygrib
from netCDF4 import Dataset
from scipy import ndimage
import pickle as cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dateutils import dateshift, daterange

GRAFDATADIR = '/data/resnet_data/GRAF/hdo-graf_conus/'
MRMSDATADIR = '/data/resnet_data/MRMS/'
CLEAD = '24'
THRESHOLDS = [0.25, 1.0, 2.5, 5.0, 10.0]
CONNECTIVITY = np.ones((3, 3))  # 8-connected

OUTDIR = '/data/resnet_data/relia'
os.makedirs(OUTDIR, exist_ok=True)
CACHE_FILE = os.path.join(OUTDIR, 'blob_size_distribution.cPick')


def date_list_for(year):
    jan = daterange(f'{year}010100', f'{year}013118', 6)
    apr = daterange(f'{year}040100', f'{year}043018', 6)
    jun = daterange(f'{year}060100', f'{year}063018', 6)
    return jan + apr + jun


def read_graf_precip(cyyyymmddhh, clead):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]
    infile = (GRAFDATADIR + cyyyymmdd + '/' + chh + '/' +
              'grid.hdo-graf_conus.' + cyyyymmdd_fcst + 'T' + chh_fcst +
              '0000Z.' + cyyyymmdd + 'T' + chh + '0000Z.PT' + clead +
              'H.CONUS@4km.APCP.SFC.grb2')
    if not os.path.exists(infile):
        return None
    try:
        g = pygrib.open(infile)
        grb = g.select(endStep=il)[0]
        vals = np.asarray(grb.values)
        g.close()
    except Exception as e:
        print(f'  Error reading {infile}: {e}')
        return None
    return np.where(vals > 75., 75., vals)


def read_mrms(cyyyymmddhh_verif):
    """Returns (precip, quality, invalid_mask). precip/quality/data_quality
    are netCDF masked arrays (~37% of the CONUS-grid domain lies outside
    MRMS/radar coverage and reads back masked, with the fill value
    ~9.97e36 for both variables if naively converted with np.asarray --
    that fill value passes a naive `quality > 0.5` check, so the mask
    must be combined explicitly with the quality/obs-range check rather
    than relying on quality alone.)"""
    infile = (MRMSDATADIR + cyyyymmddhh_verif[0:6] + '/MRMS_1h_pamt_and_data_qual_' +
              cyyyymmddhh_verif + '.nc')
    if not os.path.exists(infile):
        return None, None, None
    nc = Dataset(infile, 'r')
    precip_ma = nc.variables['precipitation'][:, :]
    quality_ma = nc.variables['data_quality'][:, :]
    nc.close()
    mask = np.ma.getmaskarray(precip_ma) | np.ma.getmaskarray(quality_ma)
    precip = np.ma.filled(precip_ma, -1.0)
    quality = np.ma.filled(quality_ma, -1.0)
    obs_range_ok = (precip >= 0.0) & (precip <= 200.0)
    invalid_mask = mask | (quality <= 0.5) | ~obs_range_ok
    return precip, quality, invalid_mask


def blob_sizes(exceed_mask, invalid_mask=None):
    """Label 8-connected blobs in exceed_mask, return sizes of blobs that
    do not touch the array boundary or (if invalid_mask given) a bad pixel."""
    labeled, nlab = ndimage.label(exceed_mask, structure=CONNECTIVITY)
    if nlab == 0:
        return np.array([], dtype=int)

    bad_ids = set()
    bad_ids.update(np.unique(labeled[0, :]))
    bad_ids.update(np.unique(labeled[-1, :]))
    bad_ids.update(np.unique(labeled[:, 0]))
    bad_ids.update(np.unique(labeled[:, -1]))
    bad_ids.discard(0)

    if invalid_mask is not None and invalid_mask.any():
        dilated_invalid = ndimage.binary_dilation(invalid_mask)
        bad_ids.update(np.unique(labeled[dilated_invalid]))
        bad_ids.discard(0)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    for bid in bad_ids:
        sizes[bid] = 0
    return sizes[sizes > 0]


def process_year(year):
    dates = date_list_for(year)
    results = {thresh: {'GRAF': [], 'MRMS': []} for thresh in THRESHOLDS}
    ngood = 0
    for idate, date in enumerate(dates):
        validity_date = dateshift(date, int(CLEAD))
        graf = read_graf_precip(date, CLEAD)
        mrms, quality, invalid_mrms = read_mrms(validity_date)
        status_g = 'ok' if graf is not None else 'missing'
        status_m = 'ok' if mrms is not None else 'missing'
        if idate % 40 == 0:
            print(f'{year} {idate:4d}/{len(dates)}  init={date}  GRAF={status_g}  MRMS={status_m}')
        if graf is None or mrms is None:
            continue
        ngood += 1
        # Restrict the MRMS exceedance field itself to good-quality,
        # in-range, unmasked pixels -- a bad/missing pixel never counts as
        # "exceeding", rather than being checked only after labeling.
        quality_good = ~invalid_mrms
        for thresh in THRESHOLDS:
            g_sizes = blob_sizes(graf >= thresh)
            m_sizes = blob_sizes((mrms >= thresh) & quality_good, invalid_mask=invalid_mrms)
            results[thresh]['GRAF'].append(g_sizes)
            results[thresh]['MRMS'].append(m_sizes)
    print(f'{year}: {ngood}/{len(dates)} dates with both fields present')
    for thresh in THRESHOLDS:
        for src in ('GRAF', 'MRMS'):
            arrs = results[thresh][src]
            results[thresh][src] = (np.concatenate(arrs) if arrs
                                     else np.array([], dtype=int))
    return results, ngood


def ccdf(sizes):
    """Return (x, y) for a log-log complementary CDF: y = P(size >= x)."""
    if sizes.size == 0:
        return np.array([]), np.array([])
    s = np.sort(sizes)[::-1]
    n = s.size
    y = np.arange(1, n + 1) / float(n)
    # Deduplicate for a cleaner plot (keep last occurrence of each value,
    # i.e. the smallest ccdf value at that size).
    uniq_s, idx = np.unique(s[::-1], return_index=True)
    y_at_uniq = y[::-1][idx]
    return uniq_s[::-1], y_at_uniq[::-1]


def main():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            all_results = cPickle.load(f)
        print(f'Loaded cached results from {CACHE_FILE}')
    else:
        all_results = {}
        for year in (2025, 2026):
            all_results[year], ngood = process_year(year)
        with open(CACHE_FILE, 'wb') as f:
            cPickle.dump(all_results, f)
        print(f'Saved results to {CACHE_FILE}')

    for thresh in THRESHOLDS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, src in zip(axes, ('GRAF', 'MRMS')):
            for year, color in ((2025, 'RoyalBlue'), (2026, 'Red')):
                sizes = all_results[year][thresh][src]
                x, y = ccdf(sizes)
                if x.size == 0:
                    continue
                ax.loglog(x, y, '-', color=color, linewidth=1.5,
                          label=f'{year} (n={sizes.size})')
            ax.set_xlabel('Blob size (pixels)')
            ax.set_ylabel('P(size >= x)')
            ax.set_title(f'{src}, threshold={thresh}mm')
            ax.legend(fontsize=9)
            ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        outfile = f'blob_ccdf_{str(thresh).replace(".", "p")}mm_lead{CLEAD}h.png'
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f'Saved {outfile}')


if __name__ == '__main__':
    main()
