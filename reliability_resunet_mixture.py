"""
python reliability_resunet_mixture.py clead

e.g.,

python reliability_resunet_mixture.py 12

    clead = sys.argv[1]

This will compute BS, reliability, freq use for the test of
Attention ResUNet with Gamma mixture model.

"""

import os, sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pygrib
from mpl_toolkits.basemap import Basemap, interp
import _pickle as cPickle
from dateutils import dateshift, daterange
from netCDF4 import Dataset
import scipy.stats as stats
from scipy import ndimage
np.set_printoptions(precision=3, suppress=True)

# --- Auto-detect environment (AWS vs local) ---
def detect_environment():
    """Detect if running on AWS or local laptop."""
    # Check for AWS paths (prioritize /data over /data2)
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path

    # Default to laptop
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    import os
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    # Check if this is laptop config or AWS config
    if "GRAFdatadir_conus" in directory:
        # Laptop config
        GRAFdatadir_conus = directory["GRAFdatadir_conus"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus"]
        GRAF_plot_dir = directory["GRAF_plot_dir"]
        mrms_data_directory = os.path.expanduser(directory["mrms_data_directory"])
    else:
        # AWS config
        GRAFdatadir_conus = directory.get("GRAFdatadir_conus_new")
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"
        mrms_data_directory = f"{base_dir}/MRMS/"

    return GRAFdatadir_conus, GRAFprobsdir_conus, \
        GRAF_plot_dir, mrms_data_directory

# ----------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    import os
    import pygrib
    istat = -1
    fexist_grib = os.path.exists(gribfilename)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep = endStep)[0]
            lats, lons = grb.latlons()
            precipitation = grb.values # constrain max amt like data save.
            precipitation = np.where(precipitation > 75., \
                75., precipitation)
            lon_0 = grb.projparams["lon_0"]
            lat_0 = grb.projparams["lat_0"]
            lat_1 = grb.projparams["lat_1"]
            lat_2 = grb.projparams["lat_2"]
            istat = 0
            fcstfile.close()
        except Exception as e:
            print(f'   Error in read_gribdata reading {gribfilename}: {e}')
            istat = -1
    else:
        print ('grib file does not exist.')
        istat = -1
        precipitation = np.empty((0,0))
        lats = np.empty((0,0))
        lons = np.empty((0,0))
        lon_0=0; lat_0=0; lat_1=0; lat_2=0 # dummy defaults

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2

# ---------------------------------------------------------------

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm= cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    # April 1, 2024 00Z is the dividing line between old and new GRAF naming
    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = prefix +cyyyymmdd_fcst+\
        'T'+chh_fcst+'0000Z.'+cyyyymmdd+'T'+chh+\
        '0000Z.PT'+clead+'H.CONUS@4km.APCP.SFC.grb2'
    infile = input_directory + input_file
    fexist1 = os.path.exists(infile)
    print (infile, fexist1)

    if fexist1 == True:
        istat, precipitation, lats, lons, lon_0, \
            lat_0, lat_1, lat_2 = read_gribdata(infile, il)
        ny, nx = np.shape(lats)
        latmax = np.max(lats); latmin = np.min(lats)
        lonmax = np.max(lons); lonmin = np.min(lons)
        tzoff = lons*12/180.
        verif_local_time = int(chh_fcst) + tzoff
    else:
        print ('  could not find ', infile)
        istat = -1
        ny = 0; nx = 0
        latmin = -99.99; latmax = -99.99
        lonmin = -999.99; lonmax = -999.99
        lon_0 = -999.99; lat_0 = -999.99
        lat_1 = -999.99; lat_2 = -999.99
        precipitation = np.empty((0,0))
        lats = np.empty((0,0), dtype=float)
        lons = np.empty((0,0), dtype=float)
        verif_local_time = np.empty((0,0), dtype=float)

    return istat, precipitation, lats, lons, ny, nx,\
        latmin, latmax, lonmin, lonmax, verif_local_time, \
        lon_0, lat_0, lat_1, lat_2

# ----------------------------------------------------------

def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus, probs_suffix='_probs_gamma_mixture.nc'):
    """Read Gamma mixture model probability files and return as dictionary."""

    infile = GRAFprobsdir_conus + cyyyymmddhh + \
        '_'+ clead + probs_suffix
    fexist = os.path.exists(infile)

    if fexist == True:
        try:
            nc = Dataset(infile,'r')
        except OSError as e:
            istat_prob = -1
            lat = np.empty((0,0), dtype=float)
            lon = np.empty((0,0), dtype=float)
            probs = None
            return istat_prob, probs, lat, lon
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]

        # Store probabilities in nested dictionary for cleaner access
        probs = {
            0.25: {
                'raw': nc.variables['raw_p0p25mm_prob'][:,:],
                'gamma': nc.variables['gamma_p0p25mm_prob'][:,:]
            },
            1.0: {
                'raw': nc.variables['raw_p1mm_prob'][:,:],
                'gamma': nc.variables['gamma_p1mm_prob'][:,:]
            },
            2.5: {
                'raw': nc.variables['raw_p2p5mm_prob'][:,:],
                'gamma': nc.variables['gamma_p2p5mm_prob'][:,:]
            },
            5.0: {
                'raw': nc.variables['raw_p5mm_prob'][:,:],
                'gamma': nc.variables['gamma_p5mm_prob'][:,:]
            },
            10.0: {
                'raw': nc.variables['raw_p10mm_prob'][:,:],
                'gamma': nc.variables['gamma_p10mm_prob'][:,:]
            }
        }
        nc.close()
        istat_prob = 0
    else:
        istat_prob = -1
        lat = np.empty((0,0), dtype=float)
        lon = np.empty((0,0), dtype=float)
        probs = None

    return istat_prob, probs, lat, lon

# -------------------------------------------------------------------------

def format_date_range(cyyyymmddhh_begin, cyyyymmddhh_end):
    """
    Format date range from YYYYMMDDHH strings to readable format.
    Example: '2025030100' and '2025033112' -> '1 Mar - 31 Mar 2025'
    """
    from datetime import datetime

    # Parse begin date
    begin_dt = datetime.strptime(cyyyymmddhh_begin[:8], '%Y%m%d')
    # Parse end date
    end_dt = datetime.strptime(cyyyymmddhh_end[:8], '%Y%m%d')

    # Format dates
    begin_str = begin_dt.strftime('%-d %b')  # '1 Mar'
    end_str = end_dt.strftime('%-d %b %Y')  # '31 Mar 2025'

    return f"{begin_str} - {end_str}"

# -------------------------------------------------------------------------

def read_MRMS(mrms_data_directory, cyyyymmddhh_verif):

    infile = mrms_data_directory + cyyyymmddhh_verif[0:6]+ \
        '/MRMS_1h_pamt_and_data_qual_' +\
        cyyyymmddhh_verif + '.nc'
    fexist = os.path.exists(infile)
    if fexist == True:
        istat = 0
        nc = Dataset(infile, 'r')
        MRMS_precipitation = nc.variables['precipitation'][:,:]
        MRMS_quality = nc.variables['data_quality'][:,:]
        nc.close()
    else:
        istat = -1
        MRMS_precipitation = np.empty((0,0), dtype=float)
        MRMS_quality = np.empty((0,0), dtype=float)
    return istat, MRMS_precipitation, MRMS_quality

# -------------------------------------------------------------------------

def bin_index(prob, ncats):
    """
    Vectorized replacement for the old per-category np.where loop: maps each
    probability to its reliability-diagram bin index. Bin k is centered at
    k/(ncats-1) with half-width 1/(2*(ncats-1)); boundary values fall into
    the upper bin (matches the old '>=pmin, <pmax' convention).

    Uses np.searchsorted against edges computed with the exact same
    expression as the old code's pmax (float(k)/(ncats-1) + 1./(2*(ncats-1))),
    rather than a closed-form floor(prob*(ncats-1)+0.5) reformulation -- the
    latter looked equivalent on paper but is NOT bit-identical at real
    boundary values (verified against real GRAF probability fields,
    2026-07-20: floor() disagreed with the old loop for a small fraction of
    pixels sitting exactly on a bin edge; searchsorted with matching edge
    arithmetic reproduces the old loop's contingency-table counts exactly).
    """
    edges = np.array([float(k) / (ncats - 1) + 1. / (2 * (ncats - 1))
                       for k in range(ncats - 1)])
    return np.searchsorted(edges, prob, side='right')

# --------------------------------------------------------

def accumulate_threshold_stats(prob_raw, prob_gamma, climo_2d, obs, quality,
                                threshold, climo_valid, region_masks, ncats):
    """
    Vectorized replacement for the old compute_contab_BS/compute_BS_climo/
    compute_BS_only trio, consolidated into one pass per (date, threshold).
    The old functions each independently recomputed quality>0.5, the obs
    validity range, and prob**2/(1-prob)**2 from scratch for every one of
    the unstratified/west/top10/bottom90 regions (a >10x redundant-pass
    factor once the terrain-roughness columns were added); here those
    per-pixel quantities are computed once and reused across all regions.
    Verified numerically identical to the old per-region function calls
    (contingency tables and BS sums, including at exact bin-boundary
    probabilities) via synthetic- and real-GRAF-data equivalence tests,
    2026-07-20.

    region_masks: dict of {name: boolean 2-D mask}. 'unstrat' should map to
    None (no extra region restriction beyond quality/obs/climo validity).

    Returns (contab_raw_delta, contab_gamma_delta, region_stats, strat_contab) where
    region_stats[name] = (bs_raw, bs_gamma, bs_climo, nsamps), and strat_contab[name]
    = (contab_raw_delta_region, contab_gamma_delta_region) for name in ('top10',
    'bottom90') -- the full per-bin contingency tables needed for performance
    diagrams (POD/success-ratio curves), which the scalar BS sums in region_stats
    can't provide.
    """
    quality_good = quality > 0.5
    obs_range_ok = np.logical_and(obs >= 0.0, obs <= 200.0)
    is_event = obs >= threshold
    valid = quality_good & obs_range_ok & climo_valid
    valid_event = valid & is_event
    valid_nonevent = valid & ~is_event

    climo_prob = np.clip(climo_2d, 0., 1.)

    bin_raw = bin_index(prob_raw, ncats)
    bin_gamma = bin_index(prob_gamma, ncats)

    contab_raw_delta = np.zeros((ncats, 2), dtype=int)
    contab_gamma_delta = np.zeros((ncats, 2), dtype=int)
    contab_raw_delta[:, 1] = np.bincount(bin_raw[valid_event], minlength=ncats)
    contab_raw_delta[:, 0] = np.bincount(bin_raw[valid_nonevent], minlength=ncats)
    contab_gamma_delta[:, 1] = np.bincount(bin_gamma[valid_event], minlength=ncats)
    contab_gamma_delta[:, 0] = np.bincount(bin_gamma[valid_nonevent], minlength=ncats)

    err_raw = np.where(is_event, (1.0 - prob_raw)**2, prob_raw**2)
    err_gamma = np.where(is_event, (1.0 - prob_gamma)**2, prob_gamma**2)
    err_climo = np.where(is_event, (1.0 - climo_prob)**2, climo_prob**2)

    region_stats = {}
    for name, extra_mask in region_masks.items():
        m = valid if extra_mask is None else np.logical_and(valid, extra_mask)
        nsamps = int(np.count_nonzero(m))
        bs_raw = float(np.sum(err_raw[m]))
        bs_gamma = float(np.sum(err_gamma[m]))
        bs_climo = float(np.sum(err_climo[m]))
        region_stats[name] = (bs_raw, bs_gamma, bs_climo, nsamps)

    # Per-bin contingency tables for the terrain-roughness strata (needed for
    # performance-diagram POD/success-ratio curves; see make_performance_diagram.py)
    strat_contab = {}
    for name in ('top10', 'bottom90'):
        extra_mask = region_masks.get(name)
        if extra_mask is None:
            continue
        ve = valid_event & extra_mask
        vn = valid_nonevent & extra_mask
        c_raw = np.zeros((ncats, 2), dtype=int)
        c_gamma = np.zeros((ncats, 2), dtype=int)
        c_raw[:, 1] = np.bincount(bin_raw[ve], minlength=ncats)
        c_raw[:, 0] = np.bincount(bin_raw[vn], minlength=ncats)
        c_gamma[:, 1] = np.bincount(bin_gamma[ve], minlength=ncats)
        c_gamma[:, 0] = np.bincount(bin_gamma[vn], minlength=ncats)
        strat_contab[name] = (c_raw, c_gamma)

    return contab_raw_delta, contab_gamma_delta, region_stats, strat_contab

# --------------------------------------------------------

def compute_relia(contab, ncats):

    """
    compute reliability and frequency of usage of
    each probability bin.

    """
    frequse = np.zeros((ncats), dtype=float)
    relia = np.zeros((ncats), dtype=float)
    nsamps_total = np.sum(contab)
    for icat in range(ncats):
        frequse[icat] = np.sum(contab[icat,:]) / float(nsamps_total)
        if np.sum(contab[icat,:]) > 5:
            relia[icat] = \
                float(contab[icat,1]) / np.sum(contab[icat,:])
        else:
            relia[icat] = -99.99
    return frequse, relia

# --------------------------------------------------------
# --------------------------------------------------------

clead = sys.argv[1]
# model_tag selects which inference output to score: 'baseline' (the
# per-lead/per-month-retrained model, default, preserves old filenames/
# cache exactly) or 'season' (the season-pooled + FiLM lead-pooled model).
# Kept separate from probs_suffix's default so cache/output filenames
# never collide between the two models when scoring the same date/lead.
model_tag = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
# date_set selects the representative-month sample: '2025' is the original
# one-month-per-season sample used for all prior baseline evaluations;
# '2026h1' is the equivalent sample drawn from the independent Jan-Jun 2026
# data (Jan/Apr/Jun as DJF/MAM/JJA proxies -- no SON proxy exists in H1).
date_set = sys.argv[3] if len(sys.argv) > 3 else '2025'
if model_tag == 'season':
    probs_suffix = '_probs_gamma_mixture_season.nc'
    cache_tag = '_season'
    out_model_name = 'ResUNet_Mixture_Season'
elif model_tag == 'baseline':
    probs_suffix = '_probs_gamma_mixture.nc'
    cache_tag = ''
    out_model_name = 'ResUNet_Mixture'
else:
    raise ValueError(f"Unknown model_tag '{model_tag}', expected 'baseline' or 'season'")
print(f"reliability_resunet_mixture.py lead={clead}h model_tag={model_tag} date_set={date_set}")
cmtit = 'GRAF'
pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
nthresholds = len(pthresholds)
ncats = 11
cmodel = 'GRAF'
cmonths = ['Jan','Feb','Mar','Apr','May','Jun','Jul',\
    'Aug','Sep','Oct','Nov','Dec']
if date_set == '2025':
    mar = daterange('2025030100','2025033118',6)
    jun = daterange('2025060100','2025063018',6)
    sep = daterange('2025090100','2025093018',6)
    dec = daterange('2025120100','2025123118',6)
    cyyyymmddhh_list = mar + jun + sep + dec
elif date_set == '2026h1':
    jan = daterange('2026010100','2026013118',6)
    apr = daterange('2026040100','2026043018',6)
    jun = daterange('2026060100','2026063018',6)
    cyyyymmddhh_list = jan + apr + jun
elif date_set == '2025h1':
    # Same Jan/Apr/Jun months as '2026h1', one year earlier -- lets a
    # comparison hold time-of-year fixed and isolate year-over-year
    # differences in the GRAF-vs-MRMS relationship instead of conflating
    # them with a seasonal difference.
    jan = daterange('2025010100','2025013118',6)
    apr = daterange('2025040100','2025043018',6)
    jun = daterange('2025060100','2025063018',6)
    cyyyymmddhh_list = jan + apr + jun
else:
    raise ValueError(f"Unknown date_set '{date_set}', expected '2025', '2026h1', or '2025h1'")
ndates = len(cyyyymmddhh_list)

# --- read paths to data

# Select config file based on environment
if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

GRAFdatadir_conus, GRAFprobsdir_conus, \
    GRAF_plot_dir, mrms_data_directory = \
    read_config_file(config_file_name, 'DIRECTORIES')

# ---- Output directory for saved reliability data

if ENVIRONMENT == 'aws':
    relia_dir = os.path.join(AWS_BASE_PATH, 'relia')
else:
    relia_dir = os.path.expanduser('~/python/resnet_data/relia')
os.makedirs(relia_dir, exist_ok=True)

# ---- Per-date cache of contingency-table/BS deltas.
# Reading raw GRAF probability + MRMS netCDFs for every date is the slow
# part of this script; the per-date accumulation itself is cheap. Caching
# each date's delta lets a rerun (e.g. to only change downstream plotting,
# or to add a new stratification) skip straight to the cheap accumulation
# step for any date already processed, instead of re-reading everything.
# Only successful ("ok") dates are cached -- checking whether a date's raw
# files are missing is itself cheap, so there's no benefit to caching misses,
# and it avoids permanently hiding a date whose data later becomes available.
daily_cache_dir = os.path.join(relia_dir, 'daily_contab')
os.makedirs(daily_cache_dir, exist_ok=True)

# ---- Read pre-interpolated Stage IV climatology on the GRAF grid

if ENVIRONMENT == 'aws':
    climo_graf_file = os.path.join(AWS_BASE_PATH, 'stage4_climo_on_graf.nc')
else:
    climo_graf_file = os.path.expanduser(
        '~/python/resnet_data/stage4_climo_on_graf.nc')

_nc = Dataset(climo_graf_file, 'r')
# NOTE: climo_prob is (7,12,24,ny,nx) float32 -- ~16 GB if loaded fully into
# memory (confirmed OOM-killing this script on this machine). Keep it as a
# netCDF4 Variable (not materialized with [:]) and slice one (ny,nx) plane
# at a time inside the date loop below; _nc stays open until that loop ends.
climo_prob_arr       = _nc.variables['climo_prob']       # (7,12,24,ny,nx)
climo_thresholds_arr = _nc.variables['threshold'][:]     # mm

# Map pthresholds -> climatology threshold dimension indices
climo_tidx = []
for thresh in pthresholds:
    idx = int(np.argmin(np.abs(climo_thresholds_arr - thresh)))
    if abs(float(climo_thresholds_arr[idx]) - thresh) > 0.01:
        print(f"WARNING: threshold {thresh} mm not found in climatology file")
    climo_tidx.append(idx)

# ---- Declare running-sum accumulators

contab_raw = np.zeros((nthresholds, ncats, 2), dtype=int)
BS_raw = np.zeros((nthresholds), dtype=float)
nsamps_raw = np.zeros((nthresholds), dtype=float)

contab_gamma = np.zeros((nthresholds, ncats, 2), dtype=int)
BS_gamma = np.zeros((nthresholds), dtype=float)
nsamps_gamma = np.zeros((nthresholds), dtype=float)

# Per-bin contingency tables for the terrain-roughness strata (performance diagrams)
contab_raw_top10    = np.zeros((nthresholds, ncats, 2), dtype=int)
contab_gamma_top10  = np.zeros((nthresholds, ncats, 2), dtype=int)
contab_raw_bottom90   = np.zeros((nthresholds, ncats, 2), dtype=int)
contab_gamma_bottom90 = np.zeros((nthresholds, ncats, 2), dtype=int)

BS_climo    = np.zeros(nthresholds, dtype=float)
nsamps_climo = np.zeros(nthresholds, dtype=float)

BS_raw_west    = np.zeros(nthresholds, dtype=float)
BS_gamma_west  = np.zeros(nthresholds, dtype=float)
BS_climo_west  = np.zeros(nthresholds, dtype=float)
nsamps_raw_west   = np.zeros(nthresholds, dtype=float)
nsamps_gamma_west = np.zeros(nthresholds, dtype=float)
nsamps_climo_west = np.zeros(nthresholds, dtype=float)

BS_raw_top10    = np.zeros(nthresholds, dtype=float)
BS_gamma_top10  = np.zeros(nthresholds, dtype=float)
BS_climo_top10  = np.zeros(nthresholds, dtype=float)
nsamps_raw_top10   = np.zeros(nthresholds, dtype=float)
nsamps_gamma_top10 = np.zeros(nthresholds, dtype=float)
nsamps_climo_top10 = np.zeros(nthresholds, dtype=float)

BS_raw_bottom90    = np.zeros(nthresholds, dtype=float)
BS_gamma_bottom90  = np.zeros(nthresholds, dtype=float)
BS_climo_bottom90  = np.zeros(nthresholds, dtype=float)
nsamps_raw_bottom90   = np.zeros(nthresholds, dtype=float)
nsamps_gamma_bottom90 = np.zeros(nthresholds, dtype=float)
nsamps_climo_bottom90 = np.zeros(nthresholds, dtype=float)

# ---- Load terrain-roughness top10/bottom90 masks (same grid/shape as the
# probability and MRMS fields, so no regridding is needed -- see
# GRAF_TERRAIN_BSS_ADAPTATION_GUIDE.md and terrain_roughness_graf.py).
_mask_nc_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'terrain_roughness_mask_graf.nc')
_mask_nc = Dataset(_mask_nc_path, 'r')
top10_mask    = np.asarray(_mask_nc.variables['top10_mask'][:], bool)
bottom90_mask = np.asarray(_mask_nc.variables['bottom90_mask'][:], bool)
_mask_nc.close()

# --- Loop over dates, accumulating contingency table and BS data

lats_save = None
lons_save = None
west_mask = None   # True where lon < -105
ngood = 0

region_accum = {
    'unstrat':  (BS_raw, BS_gamma, BS_climo,
                 nsamps_raw, nsamps_gamma, nsamps_climo),
    'west':     (BS_raw_west, BS_gamma_west, BS_climo_west,
                 nsamps_raw_west, nsamps_gamma_west, nsamps_climo_west),
    'top10':    (BS_raw_top10, BS_gamma_top10, BS_climo_top10,
                 nsamps_raw_top10, nsamps_gamma_top10, nsamps_climo_top10),
    'bottom90': (BS_raw_bottom90, BS_gamma_bottom90, BS_climo_bottom90,
                 nsamps_raw_bottom90, nsamps_gamma_bottom90, nsamps_climo_bottom90),
}

for idate, date in enumerate(cyyyymmddhh_list):
    validity_date = dateshift(date, int(clead))
    cache_file = os.path.join(daily_cache_dir, f'{date}_lead{clead}h{cache_tag}.cPick')

    # ---- Cache hit: skip the expensive read + per-pixel accumulation
    # entirely and just add this date's already-computed deltas.
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f_cache:
            cached = cPickle.load(f_cache)
        if cached.get('pthresholds') == pthresholds and cached.get('ncats') == ncats:
            ngood += 1
            contab_raw   += cached['contab_raw_delta']
            contab_gamma += cached['contab_gamma_delta']
            contab_raw_top10      += cached['contab_raw_top10_delta']
            contab_gamma_top10    += cached['contab_gamma_top10_delta']
            contab_raw_bottom90   += cached['contab_raw_bottom90_delta']
            contab_gamma_bottom90 += cached['contab_gamma_bottom90_delta']
            for name, (bs_raw_arr, bs_gamma_arr, bs_climo_arr,
                       ns_raw_arr, ns_gamma_arr, ns_climo_arr) in region_accum.items():
                c_bs_raw, c_bs_gamma, c_bs_climo, c_ns = cached['region_stats'][name]
                bs_raw_arr    += c_bs_raw
                bs_gamma_arr  += c_bs_gamma
                bs_climo_arr  += c_bs_climo
                ns_raw_arr    += c_ns
                ns_gamma_arr  += c_ns
                ns_climo_arr  += c_ns
            print(f"{idate:4d}  init={date}  lead={clead}h  [cached]")
            continue
        # else: cache was written with a different pthresholds/ncats config
        # -- fall through and recompute/overwrite it below.

    # --- Read previously generated raw and gamma-derived probabilities
    istat_prob, probs, lat, lon = \
        probability_read(clead, date, GRAFprobsdir_conus, probs_suffix)

    # Save reference lat/lon from first successful read
    if lats_save is None and istat_prob == 0:
        lats_save = lat
        lons_save = lon
        west_mask = lon < -105.0

    # ---- Read MRMS hourly accumulated precip and data quality
    istat_MRMS, MRMS_precip, MRMS_quality = \
        read_MRMS(mrms_data_directory, validity_date)

    prob_status = 'ok' if istat_prob == 0 else 'missing'
    mrms_status = 'ok' if istat_MRMS == 0 else 'missing'
    print(f"{idate:4d}  init={date}  lead={clead}h  "
          f"prob={prob_status}  mrms={mrms_status}")

    if istat_MRMS != 0 or istat_prob != 0:
        continue

    ngood += 1
    ny, nx = MRMS_precip.shape

    # ---- Look up pre-interpolated Stage IV climatology for this validity time
    validity_month_idx = int(validity_date[4:6]) - 1   # 0-indexed (0=Jan)
    validity_utc_hour  = int(validity_date[8:10])

    # Stack needed thresholds: shape (ny, nx, nthresholds)
    climo_all = np.stack([
        climo_prob_arr[climo_tidx[i], validity_month_idx, validity_utc_hour]
        for i in range(nthresholds)
    ], axis=-1)

    # ---- Accumulate contingency table and BS for each threshold
    # (vectorized: see accumulate_threshold_stats -- one consolidated pass
    # per threshold instead of 12 separate function calls each redoing the
    # same quality/obs-range/squared-error math per region)
    region_masks = {
        'unstrat':  None,
        'west':     west_mask,
        'top10':    top10_mask,
        'bottom90': bottom90_mask,
    }

    # Per-date collectors, stacked across thresholds, written to this date's
    # cache file below once the threshold loop finishes.
    date_contab_raw            = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_contab_gamma          = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_contab_raw_top10      = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_contab_gamma_top10    = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_contab_raw_bottom90   = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_contab_gamma_bottom90 = np.zeros((nthresholds, ncats, 2), dtype=int)
    date_region_stats = {name: [np.zeros(nthresholds) for _ in range(4)]
                         for name in region_masks}

    for ithresh, thresh in enumerate(pthresholds):

        climo_2d    = climo_all[:, :, ithresh]
        climo_valid = np.isfinite(climo_2d)   # False where Stage IV has no data

        ctab_r_delta, ctab_g_delta, region_stats, strat_contab = accumulate_threshold_stats(
            probs[thresh]['raw'], probs[thresh]['gamma'], climo_2d,
            MRMS_precip, MRMS_quality, thresh, climo_valid, region_masks, ncats)

        contab_raw[ithresh]   += ctab_r_delta
        contab_gamma[ithresh] += ctab_g_delta
        contab_raw_top10[ithresh]      += strat_contab['top10'][0]
        contab_gamma_top10[ithresh]    += strat_contab['top10'][1]
        contab_raw_bottom90[ithresh]   += strat_contab['bottom90'][0]
        contab_gamma_bottom90[ithresh] += strat_contab['bottom90'][1]

        date_contab_raw[ithresh]            = ctab_r_delta
        date_contab_gamma[ithresh]          = ctab_g_delta
        date_contab_raw_top10[ithresh]      = strat_contab['top10'][0]
        date_contab_gamma_top10[ithresh]    = strat_contab['top10'][1]
        date_contab_raw_bottom90[ithresh]   = strat_contab['bottom90'][0]
        date_contab_gamma_bottom90[ithresh] = strat_contab['bottom90'][1]

        for name, (bs_raw_arr, bs_gamma_arr, bs_climo_arr,
                   ns_raw_arr, ns_gamma_arr, ns_climo_arr) in region_accum.items():
            bs_raw, bs_gamma, bs_climo, ns = region_stats[name]
            bs_raw_arr[ithresh]    += bs_raw
            bs_gamma_arr[ithresh]  += bs_gamma
            bs_climo_arr[ithresh]  += bs_climo
            ns_raw_arr[ithresh]    += ns
            ns_gamma_arr[ithresh]  += ns
            ns_climo_arr[ithresh]  += ns

            date_region_stats[name][0][ithresh] = bs_raw
            date_region_stats[name][1][ithresh] = bs_gamma
            date_region_stats[name][2][ithresh] = bs_climo
            date_region_stats[name][3][ithresh] = ns

    cache_dict = {
        'pthresholds': pthresholds,
        'ncats': ncats,
        'contab_raw_delta':            date_contab_raw,
        'contab_gamma_delta':          date_contab_gamma,
        'contab_raw_top10_delta':      date_contab_raw_top10,
        'contab_gamma_top10_delta':    date_contab_gamma_top10,
        'contab_raw_bottom90_delta':   date_contab_raw_bottom90,
        'contab_gamma_bottom90_delta': date_contab_gamma_bottom90,
        'region_stats': {name: tuple(arrs) for name, arrs in date_region_stats.items()},
    }
    with open(cache_file, 'wb') as f_cache:
        cPickle.dump(cache_dict, f_cache)

_nc.close()

# ---- Check that we have usable data

if ngood == 0:
    print("\n ERROR: No dates with complete data found!")
    print(" Check that these paths exist and contain data:")
    print(f"   Probabilities: {GRAFprobsdir_conus}")
    print(f"   MRMS: {mrms_data_directory}")
    sys.exit(1)

print(f"\n Found {ngood} dates with complete data out of {ndates} total dates")

# ---- Allocate per-threshold storage for output file

relia_raw_arr    = np.full((nthresholds, ncats), -99.99)
relia_gamma_arr  = np.full((nthresholds, ncats), -99.99)
frequse_raw_arr  = np.zeros((nthresholds, ncats))
frequse_gamma_arr = np.zeros((nthresholds, ncats))
BSS_raw_arr      = np.full(nthresholds, np.nan)
BSS_gamma_arr    = np.full(nthresholds, np.nan)
BS_raw_arr       = np.full(nthresholds, np.nan)
BS_gamma_arr     = np.full(nthresholds, np.nan)
BS_climo_arr     = np.full(nthresholds, np.nan)
BSS_raw_west_arr   = np.full(nthresholds, np.nan)
BSS_gamma_west_arr = np.full(nthresholds, np.nan)
BS_climo_west_arr  = np.full(nthresholds, np.nan)
BSS_raw_top10_arr   = np.full(nthresholds, np.nan)
BSS_gamma_top10_arr = np.full(nthresholds, np.nan)
BS_climo_top10_arr  = np.full(nthresholds, np.nan)
BSS_raw_bottom90_arr   = np.full(nthresholds, np.nan)
BSS_gamma_bottom90_arr = np.full(nthresholds, np.nan)
BS_climo_bottom90_arr  = np.full(nthresholds, np.nan)

# ---- Compute reliability, frequency of usage, and Brier score per threshold

# Bin centers used as x-axis for reliability diagram
probability = np.arange(ncats) * 100. / float(ncats - 1)

for ithresh, thresh in enumerate(pthresholds):

    print ('Processing threshold = ', thresh)

    frequse_raw, relia_raw = compute_relia(contab_raw[ithresh,:,:], ncats)
    BS_raw[ithresh] = BS_raw[ithresh] / float(nsamps_raw[ithresh])

    frequse_gamma, relia_gamma = compute_relia(\
        contab_gamma[ithresh,:,:], ncats)
    BS_gamma[ithresh] = BS_gamma[ithresh] / \
        float(nsamps_gamma[ithresh])

    BS_climo_mean = BS_climo[ithresh] / float(nsamps_climo[ithresh]) \
        if nsamps_climo[ithresh] > 0 else np.nan
    BSS_raw   = 1.0 - BS_raw[ithresh]   / BS_climo_mean \
        if BS_climo_mean > 0 else np.nan
    BSS_gamma = 1.0 - BS_gamma[ithresh] / BS_climo_mean \
        if BS_climo_mean > 0 else np.nan

    BS_climo_west_mean = BS_climo_west[ithresh] / float(nsamps_climo_west[ithresh]) \
        if nsamps_climo_west[ithresh] > 0 else np.nan
    BSS_raw_west   = 1.0 - (BS_raw_west[ithresh]   / nsamps_raw_west[ithresh])   / BS_climo_west_mean \
        if BS_climo_west_mean > 0 else np.nan
    BSS_gamma_west = 1.0 - (BS_gamma_west[ithresh] / nsamps_gamma_west[ithresh]) / BS_climo_west_mean \
        if BS_climo_west_mean > 0 else np.nan

    BS_climo_top10_mean = BS_climo_top10[ithresh] / float(nsamps_climo_top10[ithresh]) \
        if nsamps_climo_top10[ithresh] > 0 else np.nan
    BSS_raw_top10   = 1.0 - (BS_raw_top10[ithresh]   / nsamps_raw_top10[ithresh])   / BS_climo_top10_mean \
        if BS_climo_top10_mean > 0 else np.nan
    BSS_gamma_top10 = 1.0 - (BS_gamma_top10[ithresh] / nsamps_gamma_top10[ithresh]) / BS_climo_top10_mean \
        if BS_climo_top10_mean > 0 else np.nan

    BS_climo_bottom90_mean = BS_climo_bottom90[ithresh] / float(nsamps_climo_bottom90[ithresh]) \
        if nsamps_climo_bottom90[ithresh] > 0 else np.nan
    BSS_raw_bottom90   = 1.0 - (BS_raw_bottom90[ithresh]   / nsamps_raw_bottom90[ithresh])   / BS_climo_bottom90_mean \
        if BS_climo_bottom90_mean > 0 else np.nan
    BSS_gamma_bottom90 = 1.0 - (BS_gamma_bottom90[ithresh] / nsamps_gamma_bottom90[ithresh]) / BS_climo_bottom90_mean \
        if BS_climo_bottom90_mean > 0 else np.nan

    print(f"  thresh={thresh}mm | CONUS:  BSS_raw={BSS_raw:.2f}  BSS_gamma={BSS_gamma:.2f}  "
          f"BS_climo={BS_climo_mean:.5f}")
    print(f"  thresh={thresh}mm | West:   BSS_raw={BSS_raw_west:.2f}  BSS_gamma={BSS_gamma_west:.2f}  "
          f"BS_climo={BS_climo_west_mean:.5f}")
    print(f"  thresh={thresh}mm | Top10:  BSS_raw={BSS_raw_top10:.2f}  BSS_gamma={BSS_gamma_top10:.2f}  "
          f"BS_climo={BS_climo_top10_mean:.5f}")
    print(f"  thresh={thresh}mm | Bot90:  BSS_raw={BSS_raw_bottom90:.2f}  BSS_gamma={BSS_gamma_bottom90:.2f}  "
          f"BS_climo={BS_climo_bottom90_mean:.5f}")

    relia_raw_arr[ithresh]    = relia_raw
    relia_gamma_arr[ithresh]  = relia_gamma
    frequse_raw_arr[ithresh]  = frequse_raw
    frequse_gamma_arr[ithresh] = frequse_gamma
    BSS_raw_arr[ithresh]      = BSS_raw
    BSS_gamma_arr[ithresh]    = BSS_gamma
    BS_raw_arr[ithresh]       = BS_raw[ithresh]
    BS_gamma_arr[ithresh]     = BS_gamma[ithresh]
    BS_climo_arr[ithresh]     = BS_climo_mean
    BSS_raw_west_arr[ithresh]   = BSS_raw_west
    BSS_gamma_west_arr[ithresh] = BSS_gamma_west
    BS_climo_west_arr[ithresh]  = BS_climo_west_mean
    BSS_raw_top10_arr[ithresh]   = BSS_raw_top10
    BSS_gamma_top10_arr[ithresh] = BSS_gamma_top10
    BS_climo_top10_arr[ithresh]  = BS_climo_top10_mean
    BSS_raw_bottom90_arr[ithresh]   = BSS_raw_bottom90
    BSS_gamma_bottom90_arr[ithresh] = BSS_gamma_bottom90
    BS_climo_bottom90_arr[ithresh]  = BS_climo_bottom90_mean

    cthresh = r'P(obs $\geq$ '+str(thresh) + ' mm)'
    ctthresh = str(thresh)+'mm'

    # ----- Make plots of 6-h reliability and frequency of usage

    cleadb = str(int(clead)-6)
    ctitle = clead+'-h forecast reliability, '+\
        cthresh  #+'\n'+ cyyyymmddhh_list[0] + ' to ' + \
        #cyyyymmddhh_list[-1]
    fig = plt.figure(figsize=(5.,5.))
    a1 = fig.add_axes([.13,.1,.83,.8])
    a1.set_title(ctitle,fontsize=14)

    ## Add date range in upper left corner
    #date_range_str = format_date_range(cyyyymmddhh_begin, cyyyymmddhh_end)
    #a1.text(0.02, 0.98, date_range_str, transform=a1.transAxes,
    #        fontsize=10, verticalalignment='top', horizontalalignment='left',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    for imodel in range(2):
        if imodel == 0:
            a1.plot([0,100],[0,100],'--',color='k')
            a1.set_ylabel('Observed relative frequency (%)',fontsize=12)
            a1.set_xlabel('Forecast probability (%)',fontsize=12)
            a1.set_ylim(-1,101)
            a1.set_xlim(-1,101)
            relia = relia_raw
            f = frequse_raw
            cbss = "%.2f" % BSS_raw if not np.isnan(BSS_raw) else "N/A"
            label = 'Smoothed GRAF raw probability, BSS = ' + cbss
            color='Red'
        elif imodel == 1:
            relia = relia_gamma
            f = frequse_gamma
            cbss = "%.2f" % BSS_gamma if not np.isnan(BSS_gamma) else "N/A"
            label = 'Attention ResUNet, BSS = ' + cbss
            color='RoyalBlue'

        relia_ma = ma.masked_where(relia < -99., relia)
        a1.plot(probability, 100.*relia_ma, 'o-',\
            color=color,linewidth=2,label=label)

        # --- Frequency of usage inset diagram

        if imodel == 0:
            a2 = fig.add_axes([.26,.63,.34,.18])
            a2.bar(probability-1.5,f[:],width=1.5,bottom=1e-5,\
                log=True,color=color,edgecolor='None',align='center')
            a2.set_xlim(-5,105)
            a2.set_ylim(1e-5,1.)
            a2.set_title('Frequency of usage',fontsize=9)
            a2.set_xlabel('Forecast probability',fontsize=7)
            a2.set_ylabel('Forecast frequency',fontsize=7)
            a2.hlines([1e-4,0.001,.01,.1],0,100,linestyles='dashed',colors='gray',lw=0.5)
        elif imodel == 1:
            a2.bar(probability, f[:], width=1.5, bottom=1e-5,\
                log=True,color=color,edgecolor='None',align='center')

    a1.legend(loc=4, fontsize='small')
    plot_title = f'Relia_GRAF_{out_model_name}_MRMS_' + \
        cyyyymmddhh_list[0] + '_to_' + cyyyymmddhh_list[-1] + '_' + \
        ctthresh + '_' + clead + 'h.png'
    print ('  Saving plot to file = ',plot_title)
    plt.savefig(plot_title, dpi=300)

# ---- Save reliability data to cPick file

out_dict = {
    'pthresholds':    pthresholds,
    'probability':    probability,
    'ngood':          ngood,
    'relia_raw':      relia_raw_arr,
    'relia_gamma':    relia_gamma_arr,
    'frequse_raw':    frequse_raw_arr,
    'frequse_gamma':  frequse_gamma_arr,
    'BSS_raw':        BSS_raw_arr,
    'BSS_gamma':      BSS_gamma_arr,
    'BS_raw':         BS_raw_arr,
    'BS_gamma':       BS_gamma_arr,
    'BS_climo':       BS_climo_arr,
    'BSS_raw_west':   BSS_raw_west_arr,
    'BSS_gamma_west': BSS_gamma_west_arr,
    'BS_climo_west':  BS_climo_west_arr,
    'BSS_raw_top10':      BSS_raw_top10_arr,
    'BSS_gamma_top10':    BSS_gamma_top10_arr,
    'BS_climo_top10':     BS_climo_top10_arr,
    'BSS_raw_bottom90':   BSS_raw_bottom90_arr,
    'BSS_gamma_bottom90': BSS_gamma_bottom90_arr,
    'BS_climo_bottom90':  BS_climo_bottom90_arr,
    'contab_raw':     contab_raw,
    'contab_gamma':   contab_gamma,
    'contab_raw_top10':      contab_raw_top10,
    'contab_gamma_top10':    contab_gamma_top10,
    'contab_raw_bottom90':   contab_raw_bottom90,
    'contab_gamma_bottom90': contab_gamma_bottom90,
    'nsamps_raw':     nsamps_raw,
    'nsamps_gamma':   nsamps_gamma,
    'nsamps_climo':   nsamps_climo,
}
relia_outfile = os.path.join(relia_dir,
    f'relia_GRAF_{out_model_name}_q0.5_{cyyyymmddhh_list[0]}_to_'
    f'{cyyyymmddhh_list[-1]}_lead{clead}h.cPick')
with open(relia_outfile, 'wb') as f_out:
    cPickle.dump(out_dict, f_out)
print(f'Saved reliability data to {relia_outfile}')
