"""
python reliability_resunet_mixture.py cyyyymmddhh_begin cyyyymmddhh_end clead

e.g.,

python reliability_resunet_mixture.py 2025120100 2025123112 12

    cyyyymmddhh_begin = sys.argv[1]
    cyyyymmddhh_end = sys.argv[2]
    clead = sys.argv[3]

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
    print(f'INFO: {config_file}')
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

    print(f"  GRAF data path: {GRAFdatadir_conus}")
    print(f"  Probs path: {GRAFprobsdir_conus}")
    print(f"  Plot path: {GRAF_plot_dir}")
    print(f"  MRMS path: {mrms_data_directory}")

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
        nc = Dataset(infile,'r')
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
        print (infile)
        print ('no such file exists.')
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
    print (infile)
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

def compute_contab_BS(nstns, prob, obs, ncats, threshold, coslat):
    """
    VECTORIZED VERSION: Compute contingency table and Brier Score.
    Much faster than looping through categories.
    """

    contab = np.zeros((ncats, 2), dtype=float)
    fmean = np.zeros((ncats), dtype=float)

    # Convert observations to binary
    binary_obs = (obs >= threshold).astype(int)

    # Define probability bin edges
    # For ncats=11, bins are: [0, 0.05, 0.15, ..., 0.95, 1.0]
    bin_edges = np.linspace(0, 1, ncats)
    bin_width = 1.0 / (ncats - 1)
    bin_edges = bin_edges - bin_width / 2.0
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    # Assign each probability to a bin (vectorized!)
    # Returns indices 0 to ncats-1
    bins = np.digitize(prob, bin_edges) - 1
    bins = np.clip(bins, 0, ncats - 1)

    # Compute contingency table using vectorized operations
    for icat in range(ncats):
        mask = (bins == icat)
        if np.any(mask):
            # Compute mean forecast probability for this bin
            fmean[icat] = np.mean(prob[mask])

            # Count events and non-events weighted by coslat
            event_mask = mask & (binary_obs == 1)
            non_event_mask = mask & (binary_obs == 0)

            contab[icat, 1] = np.sum(coslat[event_mask])
            contab[icat, 0] = np.sum(coslat[non_event_mask])

    # Compute Brier Score (vectorized)
    BS_terms = coslat * ((prob - binary_obs) ** 2)
    BS = np.sum(BS_terms)
    nsamps = np.sum(coslat)

    return contab, BS, nsamps, fmean

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
        if np.sum(contab[icat,:]) > 100:
            relia[icat] = \
                float(contab[icat,1]) / np.sum(contab[icat,:])
            frequse[icat] = \
                np.sum(contab[icat,:]) / float(nsamps_total)
        else:
            relia[icat] = -99.99
            frequse[icat] = -99.99
    return frequse, relia

# --------------------------------------------------------
# --------------------------------------------------------

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]
clead = sys.argv[3]
print (cyyyymmddhh_begin, cyyyymmddhh_end, clead)
cmtit = 'GRAF'
pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
nthresholds = len(pthresholds)
ncats = 11
cmodel = 'GRAF'
cmonths = ['Jan','Feb','Mar','Apr','May','Jun','Jul',\
    'Aug','Sep','Oct','Nov','Dec']
cyyyymmddhh_list = daterange(cyyyymmddhh_begin, \
    cyyyymmddhh_end, 12)
ndates = len(cyyyymmddhh_list)

# --- read paths to data

# Select config file based on environment
if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

print(f"Using config file: {config_file_name}")
GRAFdatadir_conus, GRAFprobsdir_conus, \
    GRAF_plot_dir, mrms_data_directory = \
    read_config_file(config_file_name, 'DIRECTORIES')

# ---- Declare arrays

contab_raw = np.zeros((nthresholds, ncats,2), dtype=int)
frequse_raw = np.zeros((nthresholds, ncats), dtype=np.float64)
fmean_raw = np.zeros((ncats), dtype=np.float64)
relia_raw = -99.99*np.ones((nthresholds, ncats), dtype=np.float64)
BS_raw = np.zeros((nthresholds), dtype=float)
nsamps_raw = np.zeros((nthresholds), dtype=int)

contab_gamma = np.zeros((nthresholds, ncats,2), dtype=int)
frequse_gamma = np.zeros((nthresholds, ncats), dtype=np.float64)
fmean_gamma = np.zeros((ncats), dtype=np.float64)
relia_gamma = -99.99*np.ones((nthresholds, ncats), dtype=np.float64)
BS_gamma = np.zeros((nthresholds), dtype=float)
nsamps_gamma = np.zeros((nthresholds), dtype=int)

# --- loop over dates using dynamic lists (no pre-allocation)

# Initialize lists to store only valid data
raw_probs = {thresh: [] for thresh in pthresholds}
gamma_probs = {thresh: [] for thresh in pthresholds}
mrms_precip_list = []
mrms_quality_list = []
coslat_list = []
lats_list = []
lons_list = []

lats_save = None
lons_save = None
coslat_save = None

for idate, date in enumerate(cyyyymmddhh_list):
    print ('-------- idate, date = ', idate, date)
    validity_date = dateshift(date, int(clead))

    # --- Read previously generated raw and gamma-derived probabilities
    istat_prob, probs, lat, lon = \
        probability_read(clead, date, GRAFprobsdir_conus)

    # Save reference lat/lon from first successful read
    if lats_save is None and istat_prob == 0:
        lats_save = lat
        lons_save = lon
        coslat_save = np.cos(lat * 3.1415926 / 180.)

    # ---- Read MRMS hourly accumulated precip and data quality
    istat_MRMS, MRMS_precip, MRMS_quality = \
        read_MRMS(mrms_data_directory, validity_date)

    # ---- If all files available, append to lists
    print ('istat_MRMS, istat_prob = ', istat_MRMS, istat_prob)
    if istat_MRMS == 0 and istat_prob == 0:
        # Append probability data for each threshold
        for thresh in pthresholds:
            raw_probs[thresh].append(probs[thresh]['raw'])
            gamma_probs[thresh].append(probs[thresh]['gamma'])

        # Append MRMS data
        mrms_precip_list.append(MRMS_precip)
        mrms_quality_list.append(MRMS_quality)

        # Append lat/lon/coslat
        lats_list.append(lat)
        lons_list.append(lon)
        coslat = np.cos(lat * 3.1415926 / 180.)
        coslat_list.append(coslat)

# --- Convert lists to arrays (only valid dates included)

ngood = len(mrms_precip_list)

if ngood == 0:
    print("\n ERROR: No dates with complete data found!")
    print(" Check that these paths exist and contain data:")
    print(f"   Probabilities: {GRAFprobsdir_conus}")
    print(f"   MRMS: {mrms_data_directory}")
    sys.exit(1)

print(f"\n Found {ngood} dates with complete data out of {ndates} total dates")

# Convert lists to numpy arrays - much faster than pre-allocating and deleting
lats_all = np.array(lats_list)
lons_all = np.array(lons_list)
coslat_all = np.array(coslat_list)
MRMS_precip_all = np.array(mrms_precip_list)
MRMS_data_quality_all = np.array(mrms_quality_list)

# Convert probability dictionaries to arrays
raw_ensemble_probs = {thresh: np.array(raw_probs[thresh]) for thresh in pthresholds}
gamma_ensemble_probs = {thresh: np.array(gamma_probs[thresh]) for thresh in pthresholds}

ndates, ny, nx = np.shape(MRMS_precip_all)
print(f"Array shape: ({ndates}, {ny}, {nx})")

# ---- Process this threshold

for ithresh, thresh in enumerate(pthresholds):

    print ('Processing threshold = ', thresh)

    # MEMORY OPTIMIZATION: Apply quality filter BEFORE flattening
    # Create boolean mask on 3D arrays (much more memory efficient)
    print ('  Creating quality mask...')
    quality_mask = (MRMS_precip_all >= 0.0) & \
                   (MRMS_data_quality_all >= 0.5) & \
                   (MRMS_precip_all < 100.0)

    # Extract only valid points (no need to flatten everything first!)
    print ('  Extracting valid points...')
    observations = MRMS_precip_all[quality_mask]
    prob_forecast_raw = raw_ensemble_probs[thresh][quality_mask]
    prob_forecast_gamma = gamma_ensemble_probs[thresh][quality_mask]
    coslat_flat = coslat_all[quality_mask]

    nobs = len(observations)
    print (f'  Valid observations: {nobs:,} (reduced from {MRMS_precip_all.size:,})')

    # Note: We don't need lats/lons for the contingency table computation

    # --- contingency tables for raw

    print ('  Computing contingency table for Raw')
    contab_raw[ithresh,:,:], BS_raw[ithresh], \
        nsamps_raw[ithresh], fmean_raw = \
        compute_contab_BS(nobs, prob_forecast_raw, \
        observations, ncats, thresh, coslat_flat)
    print ('fmean raw: ', fmean_raw)

    print ('  Computing contingency table for Gamma mixture')
    contab_gamma[ithresh,:,:], BS_gamma[ithresh], \
        nsamps_gamma[ithresh], fmean_gamma = \
        compute_contab_BS(nobs, prob_forecast_gamma, \
        observations, ncats, thresh, coslat_flat)
    print ('fmean gamma mixture: ', fmean_gamma)


    cthresh = r'P(obs $\geq$ '+str(thresh) + ' mm)'
    ctthresh = str(thresh)+'mm'

    # --- Calculate frequency of use and reliability for raw

    frequse_raw, relia_raw = compute_relia(contab_raw[ithresh,:,:], ncats)
    BS_raw[ithresh] = BS_raw[ithresh] / float(nsamps_raw[ithresh])

    # --- Calculate frequency of use and reliability for Gamma mixture

    frequse_gamma, relia_gamma = compute_relia(\
        contab_gamma[ithresh,:,:], ncats)
    BS_gamma[ithresh] = BS_gamma[ithresh] / \
        float(nsamps_gamma[ithresh])

    # ----- Make plots of 6-h reliability and frequency of usage

    probability = np.arange(11) * 100. / np.real(10.)
    cleadb = str(int(clead)-6)
    ctitle = clead+'-h forecast reliability, '+\
        cthresh  #+'\n'+ cyyyymmddhh_list[0] + ' to ' + \
        #cyyyymmddhh_list[-1]
    fig = plt.figure(figsize=(5.,5.))
    a1 = fig.add_axes([.13,.1,.83,.8])
    a1.set_title(ctitle,fontsize=14)

    # Add date range in upper left corner
    date_range_str = format_date_range(cyyyymmddhh_begin, cyyyymmddhh_end)
    a1.text(0.02, 0.98, date_range_str, transform=a1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    for imodel in range(2):
        if imodel == 0:
            a1.plot([0,100],[0,100],'--',color='k')
            a1.set_ylabel('Observed relative frequency (%)',fontsize=12)
            a1.set_xlabel('Forecast probability (%)',fontsize=12)
            a1.set_ylim(-1,101)
            a1.set_xlim(-1,101)
            relia = relia_raw
            prob_adjusted = fmean_raw
            f = frequse_raw
            cbs = "%0.5f"%(BS_raw[ithresh])
            label='Smoothed GRAF raw probability, BS = '+cbs
            color='Red'
        elif imodel == 1:
            relia = relia_gamma
            f = frequse_gamma
            prob_adjusted = fmean_gamma
            cbs = "%0.5f"%(BS_gamma[ithresh])
            label='Fitted Gamma mixture probability, BS = '+cbs
            color='RoyalBlue'

        relia_ma = ma.masked_where(f < 1.e-4, relia)
        prob_adjusted_ma = ma.masked_where(f < 1.e-4, prob_adjusted)
        a1.plot(100.*prob_adjusted_ma, 100.*relia_ma, 'o-',\
            color=color,linewidth=2,label=label)

        # --- Frequency of usage inset diagram

        if imodel == 0:
            a2 = fig.add_axes([.26,.63,.34,.18])
            a2.bar(probability-1.5,f[:],width=1.5,bottom=0.0001,\
                log=True,color=color,edgecolor='None',align='center')
            a2.set_xlim(-5,105)
            a2.set_ylim(0.0001,1.)
            a2.set_title('Frequency of usage',fontsize=9)
            a2.set_xlabel('Forecast probability',fontsize=7)
            a2.set_ylabel('Forecast frequency',fontsize=7)
            a2.hlines([0.001,.01,.1],0,100,linestyles='dashed',colors='gray',lw=0.5)
        elif imodel == 1:
            a2.bar(probability, f[:], width=1.5, bottom=0.0001,\
                log=True,color=color,edgecolor='None',align='center')

    a1.legend(loc=4, fontsize='small')
    plot_title = 'Relia_GRAF_ResUNet_Mixture_MRMS_' + \
        cyyyymmddhh_list[0] + '_to_' + cyyyymmddhh_list[-1] + '_' + \
        ctthresh + '_' + clead + 'h.png'
    print ('  Saving plot to file = ',plot_title)
    plt.savefig(plot_title, dpi=300)

