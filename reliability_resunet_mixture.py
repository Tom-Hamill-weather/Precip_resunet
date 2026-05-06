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

def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus):
    """Read Gamma mixture model probability files and return as dictionary."""

    infile = GRAFprobsdir_conus + cyyyymmddhh + \
        '_'+ clead + '_probs_gamma_mixture.nc'
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

def compute_contab_BS(ny, nx, prob, obs, quality, ncats, threshold,
                       climo_mask=None):
    """
    Compute contingency table and Brier Score for one case day.
    Operates on full 2D arrays; handles quality masking internally.
    Call once per case day per threshold; accumulate returned values
    into running totals outside.
    climo_mask: optional bool 2-D array (True = covered by climatology).
    When provided, pixels outside the climatology domain are excluded.
    """

    contab = np.zeros((ncats, 2), dtype=int)

    # Assign binary_obs: 1=event, 0=non-event, -1=masked
    binary_obs = -1 * np.ones((ny, nx), dtype=int)

    base_cond = quality > 0.5
    if climo_mask is not None:
        base_cond = np.logical_and(base_cond, climo_mask)

    a = np.where(np.logical_and(base_cond,
        np.logical_and(obs >= threshold, obs <= 200.0)))
    binary_obs[a] = 1

    a = np.where(np.logical_and(base_cond,
        np.logical_and(obs >= 0.0,
        np.logical_and(obs < threshold, obs <= 200.0))))
    binary_obs[a] = 0

    # Accumulate contingency table counts per probability bin
    for icat in range(ncats):
        pmin = np.max([0.0, float(icat) / (ncats-1) - 1./(2*(ncats-1))])
        pmax = np.min([1.0, float(icat) / (ncats-1) + 1./(2*(ncats-1))])
        in_bin = np.logical_and(prob >= pmin,
            prob <= pmax if icat == ncats-1 else prob < pmax)

        a = np.where(np.logical_and(in_bin, binary_obs == 1))
        contab[icat, 1] += len(a[0])

        a = np.where(np.logical_and(in_bin, binary_obs == 0))
        contab[icat, 0] += len(a[0])

    # Brier Score over quality-masked pixels
    good_0 = np.where(binary_obs == 0)
    good_1 = np.where(binary_obs == 1)
    BS = float(np.sum(prob[good_0]**2) + np.sum((1.0 - prob[good_1])**2))
    nsamps = len(good_0[0]) + len(good_1[0])

    return contab, BS, nsamps

# --------------------------------------------------------

def compute_BS_climo(ny, nx, climo_prob_2d, obs, quality, threshold, climo_mask):
    """
    Brier Score contribution for the climatological forecast.
    Only counts pixels where quality > 0.5 AND climo_mask is True,
    so the sample set is identical to compute_contab_BS with climo_mask.
    """
    base = np.logical_and(quality > 0.5, climo_mask)

    good_1 = np.where(np.logical_and(base,
        np.logical_and(obs >= threshold, obs <= 200.0)))
    good_0 = np.where(np.logical_and(base,
        np.logical_and(obs >= 0.0,
        np.logical_and(obs < threshold, obs <= 200.0))))

    p0 = np.clip(climo_prob_2d[good_0], 0., 1.)
    p1 = np.clip(climo_prob_2d[good_1], 0., 1.)

    BS = float(np.sum(p0**2) + np.sum((1.0 - p1)**2))
    nsamps = len(good_0[0]) + len(good_1[0])
    return BS, nsamps

# --------------------------------------------------------

def compute_BS_only(prob, obs, quality, threshold, mask):
    """
    Brier Score and sample count under an arbitrary boolean mask.
    Pass the combined mask (e.g. climo_valid & west_mask) as 'mask'.
    """
    base = np.logical_and(quality > 0.5, mask)
    good_1 = np.where(np.logical_and(base,
        np.logical_and(obs >= threshold, obs <= 200.0)))
    good_0 = np.where(np.logical_and(base,
        np.logical_and(obs >= 0.0,
        np.logical_and(obs < threshold, obs <= 200.0))))
    p0 = np.clip(prob[good_0], 0., 1.)
    p1 = np.clip(prob[good_1], 0., 1.)
    BS = float(np.sum(p0**2) + np.sum((1.0 - p1)**2))
    nsamps = len(good_0[0]) + len(good_1[0])
    return BS, nsamps

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
print(f"reliability_resunet_mixture.py lead={clead}h")
cmtit = 'GRAF'
pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
nthresholds = len(pthresholds)
ncats = 11
cmodel = 'GRAF'
cmonths = ['Jan','Feb','Mar','Apr','May','Jun','Jul',\
    'Aug','Sep','Oct','Nov','Dec']
mar = daterange('2025030100','2025033118',6)
jun = daterange('2025060100','2025063018',6)
sep = daterange('2025090100','2025093018',6)
dec = daterange('2025120100','2025123118',6)

cyyyymmddhh_list = mar + jun + sep + dec
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

# ---- Read pre-interpolated Stage IV climatology on the GRAF grid

if ENVIRONMENT == 'aws':
    climo_graf_file = os.path.join(AWS_BASE_PATH, 'stage4_climo_on_graf.nc')
else:
    climo_graf_file = os.path.expanduser(
        '~/python/resnet_data/stage4_climo_on_graf.nc')

_nc = Dataset(climo_graf_file, 'r')
climo_prob_arr       = _nc.variables['climo_prob'][:]    # (7,12,24,ny,nx)
climo_thresholds_arr = _nc.variables['threshold'][:]     # mm
_nc.close()

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

BS_climo    = np.zeros(nthresholds, dtype=float)
nsamps_climo = np.zeros(nthresholds, dtype=float)

BS_raw_west    = np.zeros(nthresholds, dtype=float)
BS_gamma_west  = np.zeros(nthresholds, dtype=float)
BS_climo_west  = np.zeros(nthresholds, dtype=float)
nsamps_raw_west   = np.zeros(nthresholds, dtype=float)
nsamps_gamma_west = np.zeros(nthresholds, dtype=float)
nsamps_climo_west = np.zeros(nthresholds, dtype=float)

# --- Loop over dates, accumulating contingency table and BS data

lats_save = None
lons_save = None
west_mask = None   # True where lon < -105
ngood = 0

for idate, date in enumerate(cyyyymmddhh_list):
    validity_date = dateshift(date, int(clead))

    # --- Read previously generated raw and gamma-derived probabilities
    istat_prob, probs, lat, lon = \
        probability_read(clead, date, GRAFprobsdir_conus)

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
    for ithresh, thresh in enumerate(pthresholds):

        climo_2d    = climo_all[:, :, ithresh]
        climo_valid = np.isfinite(climo_2d)   # False where Stage IV has no data

        ctab, bs, ns = compute_contab_BS(ny, nx,
            probs[thresh]['raw'], MRMS_precip, MRMS_quality, ncats, thresh,
            climo_mask=climo_valid)
        contab_raw[ithresh] += ctab
        BS_raw[ithresh] += bs
        nsamps_raw[ithresh] += ns

        ctab, bs, ns = compute_contab_BS(ny, nx,
            probs[thresh]['gamma'], MRMS_precip, MRMS_quality, ncats, thresh,
            climo_mask=climo_valid)
        contab_gamma[ithresh] += ctab
        BS_gamma[ithresh] += bs
        nsamps_gamma[ithresh] += ns

        bs_c, ns_c = compute_BS_climo(ny, nx, climo_2d,
            MRMS_precip, MRMS_quality, thresh, climo_valid)
        BS_climo[ithresh]    += bs_c
        nsamps_climo[ithresh] += ns_c

        # ---- West-of-105W subset (for BSS only, not reliability diagrams)
        west_climo_mask = np.logical_and(climo_valid, west_mask)

        bs_rw, ns_rw = compute_BS_only(probs[thresh]['raw'],
            MRMS_precip, MRMS_quality, thresh, west_climo_mask)
        BS_raw_west[ithresh]    += bs_rw
        nsamps_raw_west[ithresh] += ns_rw

        bs_gw, ns_gw = compute_BS_only(probs[thresh]['gamma'],
            MRMS_precip, MRMS_quality, thresh, west_climo_mask)
        BS_gamma_west[ithresh]    += bs_gw
        nsamps_gamma_west[ithresh] += ns_gw

        bs_cw, ns_cw = compute_BS_climo(ny, nx, climo_2d,
            MRMS_precip, MRMS_quality, thresh, west_climo_mask)
        BS_climo_west[ithresh]    += bs_cw
        nsamps_climo_west[ithresh] += ns_cw

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

    print(f"  thresh={thresh}mm | CONUS:  BSS_raw={BSS_raw:.2f}  BSS_gamma={BSS_gamma:.2f}  "
          f"BS_climo={BS_climo_mean:.5f}")
    print(f"  thresh={thresh}mm | West:   BSS_raw={BSS_raw_west:.2f}  BSS_gamma={BSS_gamma_west:.2f}  "
          f"BS_climo={BS_climo_west_mean:.5f}")

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
    plot_title = 'Relia_GRAF_ResUNet_Mixture_MRMS_' + \
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
    'contab_raw':     contab_raw,
    'contab_gamma':   contab_gamma,
    'nsamps_raw':     nsamps_raw,
    'nsamps_gamma':   nsamps_gamma,
    'nsamps_climo':   nsamps_climo,
}
relia_outfile = os.path.join(relia_dir,
    f'relia_GRAF_ResUNet_Mixture_q0.5_{cyyyymmddhh_list[0]}_to_'
    f'{cyyyymmddhh_list[-1]}_lead{clead}h.cPick')
with open(relia_outfile, 'wb') as f_out:
    cPickle.dump(out_dict, f_out)
print(f'Saved reliability data to {relia_outfile}')
