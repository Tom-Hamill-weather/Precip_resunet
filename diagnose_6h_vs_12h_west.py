"""
python diagnose_6h_vs_12h_west.py cyyyymmddhh_begin cyyyymmddhh_end

For each initialization time T in the date range, compare two forecasts that
verify at the same valid time (T+6h):

  6h  forecast:  init=T,      lead=6h
  12h forecast:  init=T-6h,   lead=12h  (previous GRAF cycle)

For both, compute per-day mean Brier Score over the western US (lon < -105W)
at the 5mm threshold using the ResUNet gamma-mixture probabilities.

Rank days by (mean_BS_6h - mean_BS_12h), most positive (6h worst) at top.

Usage:
    python diagnose_6h_vs_12h_west.py 2025030100 2025123112
"""

import os, sys
import numpy as np
import _pickle as cPickle
from dateutils import dateshift, daterange
from netCDF4 import Dataset

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            return 'aws', path
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# ---------------------------------------------------------------------------

def read_config_file(config_file):
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.read(config_file)
    d = cfg['DIRECTORIES']
    if 'GRAFprobsdir_conus' in d:
        GRAFprobsdir = d['GRAFprobsdir_conus']
        mrms_dir     = os.path.expanduser(d['mrms_data_directory'])
    else:
        base = d.get('resnet_data_directory', AWS_BASE_PATH or '/data/resnet_data')
        GRAFprobsdir = f'{base}/probs/'
        mrms_dir     = f'{base}/MRMS/'
    return GRAFprobsdir, mrms_dir

# ---------------------------------------------------------------------------

def probability_read(clead, cyyyymmddhh, GRAFprobsdir):
    """Return gamma-mixture 5mm prob array, or None on failure."""
    infile = os.path.join(GRAFprobsdir,
        f'{cyyyymmddhh}_{int(clead)}_probs_gamma_mixture.nc')
    if not os.path.exists(infile):
        return None
    try:
        nc = Dataset(infile, 'r')
        p = nc.variables['gamma_p5mm_prob'][:,:]
        nc.close()
        return p
    except Exception as e:
        print(f'  Warning: could not read {infile}: {e}')
        return None

# ---------------------------------------------------------------------------

def read_MRMS(mrms_dir, cyyyymmddhh_verif):
    infile = os.path.join(mrms_dir, cyyyymmddhh_verif[:6],
        f'MRMS_1h_pamt_and_data_qual_{cyyyymmddhh_verif}.nc')
    if not os.path.exists(infile):
        return None, None
    try:
        nc = Dataset(infile, 'r')
        precip  = nc.variables['precipitation'][:,:]
        quality = nc.variables['data_quality'][:,:]
        nc.close()
        return precip, quality
    except Exception as e:
        print(f'  Warning: could not read MRMS {infile}: {e}')
        return None, None

# ---------------------------------------------------------------------------

def compute_BS_west(prob, obs, quality, west_mask, climo_mask, threshold=5.0):
    """
    Mean Brier Score and sample count over western US pixels passing
    quality and climo masks.  Returns (mean_BS, nsamps, nevents).
    """
    mask = np.logical_and(quality > 0.5, climo_mask)
    mask = np.logical_and(mask, west_mask)

    good_1 = np.where(np.logical_and(mask,
                 np.logical_and(obs >= threshold, obs <= 200.0)))
    good_0 = np.where(np.logical_and(mask,
                 np.logical_and(obs >= 0.0,
                 np.logical_and(obs < threshold, obs <= 200.0))))

    p0 = np.clip(prob[good_0], 0., 1.)
    p1 = np.clip(prob[good_1], 0., 1.)
    BS      = float(np.sum(p0**2) + np.sum((1.0 - p1)**2))
    nsamps  = len(good_0[0]) + len(good_1[0])
    nevents = len(good_1[0])
    mean_BS = BS / nsamps if nsamps > 0 else np.nan
    return mean_BS, nsamps, nevents

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end   = sys.argv[2]

print(f'diagnose_6h_vs_12h_west.py  {cyyyymmddhh_begin} to {cyyyymmddhh_end}')

config_file = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
GRAFprobsdir, mrms_dir = read_config_file(config_file)

# Climatology (needed only for the climo_mask — valid-pixel coverage)
if ENVIRONMENT == 'aws':
    climo_file = os.path.join(AWS_BASE_PATH, 'stage4_climo_on_graf.nc')
else:
    climo_file = os.path.expanduser('~/python/resnet_data/stage4_climo_on_graf.nc')

_nc = Dataset(climo_file, 'r')
climo_prob_arr       = _nc.variables['climo_prob'][:]     # (nthresh,12,24,ny,nx)
climo_thresholds_arr = _nc.variables['threshold'][:]
_nc.close()

climo_tidx = int(np.argmin(np.abs(climo_thresholds_arr - 5.0)))
print(f'Climatology threshold index {climo_tidx} '
      f'({climo_thresholds_arr[climo_tidx]:.2f} mm) used for pixel mask only')

all_dates = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 6)
print(f'Processing {len(all_dates)} init times')

records  = []
west_mask = None

for idate, date in enumerate(all_dates):
    valid_time = dateshift(date,  6)
    prev_init  = dateshift(date, -6)

    p6h  = probability_read('06', date,      GRAFprobsdir)
    p12h = probability_read('12', prev_init, GRAFprobsdir)
    obs, qual = read_MRMS(mrms_dir, valid_time)

    status = (
        ('6h_ok'   if p6h  is not None else '6h_miss')  + '  ' +
        ('12h_ok'  if p12h is not None else '12h_miss') + '  ' +
        ('mrms_ok' if obs  is not None else 'mrms_miss')
    )
    print(f'{idate:4d}  init={date}  valid={valid_time}  {status}')

    if p6h is None or p12h is None or obs is None:
        continue

    ny, nx = obs.shape

    if west_mask is None:
        infile = os.path.join(GRAFprobsdir,
            f'{date}_6_probs_gamma_mixture.nc')
        nc  = Dataset(infile, 'r')
        lon = nc.variables['lon'][:,:]
        lat = nc.variables['lat'][:,:]
        nc.close()
        west_mask = lon < -105.0
        print(f'  west_mask: {west_mask.sum():,} of {ny*nx:,} pixels')

    vmon       = int(valid_time[4:6]) - 1
    vhour      = int(valid_time[8:10])
    climo_2d   = climo_prob_arr[climo_tidx, vmon, vhour]
    climo_mask = np.isfinite(climo_2d)

    mbs6,  ns6,  nev6  = compute_BS_west(p6h,  obs, qual, west_mask, climo_mask)
    mbs12, ns12, nev12 = compute_BS_west(p12h, obs, qual, west_mask, climo_mask)

    if np.isnan(mbs6) or np.isnan(mbs12):
        continue

    records.append({
        'init_6h':   date,
        'init_12h':  prev_init,
        'valid':     valid_time,
        'mbs6':      mbs6,
        'mbs12':     mbs12,
        'dbs':       mbs6 - mbs12,    # positive = 6h worse
        'ns6':       ns6,
        'ns12':      ns12,
        'nevents6':  nev6,
        'nevents12': nev12,
    })

print(f'\nProcessed {len(records)} dates with complete data')

# ---------------------------------------------------------------------------
# Rank by dbs (most positive = 6h worst relative to same-valid-time 12h)
# ---------------------------------------------------------------------------

records.sort(key=lambda r: r['dbs'], reverse=True)

# Require a minimum number of western-US events to reduce noise
min_events = 50
filtered = [r for r in records
            if r['nevents6'] >= min_events and r['nevents12'] >= min_events]
print(f'{len(filtered)} dates with >= {min_events} western US 5mm events in both forecasts')

print()
hdr = (f'{"init_6h":>12}  {"valid":>12}  {"mBS_6h":>8}  '
       f'{"mBS_12h":>8}  {"dBS":>8}  {"nevt6":>6}  {"nevt12":>7}')
print('Top 30 dates where 6h ResUNet is WORST relative to previous-cycle 12h (western US, 5mm):')
print(hdr)
print('-' * 75)
for r in filtered[:30]:
    print(f'{r["init_6h"]:>12}  {r["valid"]:>12}  {r["mbs6"]:>8.5f}  '
          f'{r["mbs12"]:>8.5f}  {r["dbs"]:>8.5f}  '
          f'{r["nevents6"]:>6d}  {r["nevents12"]:>7d}')

print()
print('Top 10 dates where 6h ResUNet is BEST relative to previous-cycle 12h:')
print(hdr)
print('-' * 75)
for r in filtered[-10:][::-1]:
    print(f'{r["init_6h"]:>12}  {r["valid"]:>12}  {r["mbs6"]:>8.5f}  '
          f'{r["mbs12"]:>8.5f}  {r["dbs"]:>8.5f}  '
          f'{r["nevents6"]:>6d}  {r["nevents12"]:>7d}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

if ENVIRONMENT == 'aws':
    out_dir = os.path.join(AWS_BASE_PATH, 'relia')
else:
    out_dir = os.path.expanduser('~/python/resnet_data/relia')
os.makedirs(out_dir, exist_ok=True)

outfile = os.path.join(out_dir,
    f'diag_6h_vs_12h_west_5mm_{cyyyymmddhh_begin}_to_{cyyyymmddhh_end}.cPick')
with open(outfile, 'wb') as f:
    cPickle.dump({'records': records, 'filtered': filtered}, f)
print(f'\nSaved {outfile}')
