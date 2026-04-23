"""
python relia_GRAF_raw_multisigma.py cyyyymmddhh_begin cyyyymmddhh_end clead

e.g.,

python relia_GRAF_raw_multisigma.py 202512010100 202512311200 12

    cyyyymmddhh_begin = sys.argv[1]
    cyyyymmddhh_end = sys.argv[2]
    clead = sys.argv[3]

This will compute BS, reliability for GRAF raw Gaussian-convolved probabilities.

"""

import os, sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import _pickle as cPickle
from dateutils import dateshift, daterange
from netCDF4 import Dataset
import scipy.stats as stats

# ------------------------------------------------------------

def read_config_file(config_file, directory_object_name):

    """ read appropriate information from the config file
        and return
    """

    from configparser import ConfigParser

    # ---- Read config.ini file

    config_object = ConfigParser()
    config_object.read(config_file)

    print(f'INFO: config_file = {config_file}')

    # ---- Get the information from dictionary structure

    directory = config_object[directory_object_name]
    MRMS_directory = directory["MRMS_directory"]
    probs_directory = directory["probs_directory"]

    return MRMS_directory, probs_directory

# ============================================================

def read_probs(filename):

    """
    Read multi-sigma probability forecasts from netCDF.
    Returns empty arrays if file does not exist.
    """

    fexist = os.path.exists(filename)
    print ('trying to read ', filename, fexist)
    if fexist == True:
        nc = Dataset(filename,'r')
        lats = nc.variables['lats'][:,:]
        lons = nc.variables['lons'][:,:]
        p0p25mm_raw = nc.variables['p0p25mm_raw_forecast'][:,:,:].filled(-99.99)
        p1mm_raw = nc.variables['p1mm_raw_forecast'][:,:,:].filled(-99.99)
        p5mm_raw = nc.variables['p5mm_raw_forecast'][:,:,:].filled(-99.99)
        p10mm_raw = nc.variables['p10mm_raw_forecast'][:,:,:].filled(-99.99)
        sigmas = nc.variables['sigmas'][:].filled(-99.99)
        ny, nx = np.shape(lats)
        nsigmas = len(sigmas)
        nc.close()
    else:
        print ('this file does not exist.')
        p0p25mm_raw = np.empty((0,0,0), dtype=float)
        p1mm_raw = np.empty((0,0,0), dtype=float)
        p5mm_raw = np.empty((0,0,0), dtype=float)
        p10mm_raw = np.empty((0,0,0), dtype=float)
        lats = np.empty((0,0), dtype=float)
        lons = np.empty((0,0), dtype=float)
        sigmas = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0]
        nsigmas = len(sigmas)
        ny, nx = 0, 0

    return ny, nx, lats, lons, p0p25mm_raw, p1mm_raw, \
        p5mm_raw, p10mm_raw, sigmas, nsigmas

# ============================================================

def read_MRMS(MRMS_directory, date):
    infile = MRMS_directory + date[0:6] + '/' + \
        'MRMS_1h_pamt_and_data_qual_'+date+'.nc'
    fexist = os.path.exists(infile)
    if fexist == True:
        nc = Dataset(infile, 'r')
        quality = nc.variables['data_quality'][:,:].filled(0.0)
        observations = nc.variables['precipitation'][:,:].filled(-1.0)
        istat = 0
        nc.close()

        # ---- QC diagnostics
        ntotal = quality.size
        nbad_quality = np.sum(quality <= 0.5)
        nneg_precip = np.sum(observations < 0.0)
        nhigh_precip = np.sum(observations > 200.0)
        ngood = np.sum(quality > 0.5)
        print(f'  MRMS QC: quality range [{quality.min():.3f}, {quality.max():.3f}], '
              f'{nbad_quality}/{ntotal} pixels filtered (quality<=0.5)')
        print(f'  MRMS QC: precip range [{observations.min():.3f}, {observations.max():.3f}] mm, '
              f'{nneg_precip} pixels <0, {nhigh_precip} pixels >200mm')
        print(f'  MRMS QC: {ngood} good-quality pixels retained ({100.*ngood/ntotal:.1f}%)')
    else:
        istat = -1
        quality = np.empty((0,0),dtype=float)
        observations = np.empty((0,0), dtype=float)

    return observations, quality, istat

# ============================================================

def compute_contab_BS(ny, nx, prob, obs, quality, contab, ncats,
        threshold, verbose=False):

    """ For this case day, compute the contingency table elements
        and Brier Score, using 2D gridded arrays. """

    # ---- Convert observation to binary.  Use full np.where tuple
    #      to correctly index 2D arrays.

    binary_obs = -1*np.ones((ny, nx), dtype=int)

    a = np.where(np.logical_and(quality > 0.5,
        np.logical_and(obs >= threshold, obs <= 200.0)))
    binary_obs[a] = 1

    a = np.where(np.logical_and(quality > 0.5,
        np.logical_and(obs >= 0.0, np.logical_and(obs < threshold, obs <= 200.0))))
    binary_obs[a] = 0

    if verbose:
        nevents = np.sum(binary_obs == 1)
        nnonevents = np.sum(binary_obs == 0)
        nfiltered = np.sum(binary_obs == -1)
        base_rate = nevents / float(nevents + nnonevents) if (nevents + nnonevents) > 0 else -99.
        print(f'    thresh={threshold:.3f}mm: events={nevents}, non-events={nnonevents}, '
              f'filtered={nfiltered}, base_rate={base_rate:.3f}')

    # ---- Contingency table: count hits/non-hits per probability category.

    for icat in range(ncats):
        pmin = np.max([0.0, float(icat) / (ncats-1) - 1./(2*(ncats-1))])
        pmax = np.min([1.0, float(icat) / (ncats-1) + 1./(2*(ncats-1))])

        in_bin = np.logical_and(prob >= pmin,
            prob <= pmax if icat == ncats-1 else prob < pmax)

        a = np.where(np.logical_and(in_bin, binary_obs == 1))
        if len(a[0]) > 0:
            contab[icat,1] = contab[icat,1] + len(a[0])

        a = np.where(np.logical_and(in_bin, binary_obs == 0))
        if len(a[0]) > 0:
            contab[icat,0] = contab[icat,0] + len(a[0])

    # ---- Brier Score (vectorized over 2D grid).

    good_0 = np.where(binary_obs == 0)
    good_1 = np.where(binary_obs == 1)
    BS = float(np.sum(prob[good_0]**2) + np.sum((1.0 - prob[good_1])**2))
    nsamps = len(good_0[0]) + len(good_1[0])

    return contab, BS, nsamps

# --------------------------------------------------------

def compute_relia(contab, ncats, frequse, relia):

    """
    compute reliability and frequency of usage of
    each probability bin.

    """
    nsamps_total = np.sum(contab)
    for icat in range(ncats):
        frequse[icat] = np.sum(contab[icat,:]) / float(nsamps_total)
        if np.sum(contab[icat,:]) > 5:
            relia[icat] = float(contab[icat,1]) / np.sum(contab[icat,:])
        else:
            relia[icat] = -99.99
    return frequse, relia

# --------------------------------------------------------
# --------------------------------------------------------

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]
clead = sys.argv[3]
cleadb = str(int(clead)-1)
print (cyyyymmddhh_begin, cyyyymmddhh_end, clead)
cmtit = 'GRAF'
pthresholds = [0.25, 1.0, 5.0, 10.0]
nthresholds = len(pthresholds)
ncats = 11

# --- sigma list defined here so nsigmas is known before array allocation.

sigmas = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0]
nsigmas = len(sigmas)

cmonths = ['Jan','Feb','Mar','Apr','May','Jun','Jul',\
    'Aug','Sep','Oct','Nov','Dec']
cyyyymmddhh_list = daterange(cyyyymmddhh_begin, \
    cyyyymmddhh_end, 24)

# --- config file read for directory names.

directory_object_name = 'DIRECTORIES'
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_hdo.ini')
MRMS_directory, probs_directory = \
    read_config_file(config_file, directory_object_name)

# --- Pre-allocate accumulated arrays (nsigmas now known).

contab_raw_sigmas = np.zeros((nthresholds,ncats,2,nsigmas), dtype=int)
frequse_raw_sigmas = np.zeros((nthresholds,ncats,nsigmas), dtype=np.float64)
relia_raw_sigmas = -99.99*np.ones((nthresholds, ncats,nsigmas), dtype=np.float64)
BS_raw_sigmas = np.zeros((nthresholds, nsigmas), dtype=float)
nsamps_sigmas = np.zeros((nthresholds, nsigmas), dtype=float)

# --- Cache file: if it exists, load accumulated arrays and skip the date loop.

relia_save_dir = '/data/resnet_data/relia'
cache_file = os.path.join(relia_save_dir,
    f'relia_GRAF_raw_{cyyyymmddhh_begin}_to_{cyyyymmddhh_end}_lead{clead}h.cPick')

cache_loaded = False
if os.path.exists(cache_file):
    print(f'Loading cached accumulated data from {cache_file}')
    with open(cache_file, 'rb') as f:
        cache = cPickle.load(f)
    contab_raw_sigmas = cache['contab_raw_sigmas']
    BS_raw_sigmas = cache['BS_raw_sigmas']
    nsamps_sigmas = cache['nsamps_sigmas']
    if np.sum(nsamps_sigmas) > 0:
        cache_loaded = True
        print('Cache loaded; skipping date loop.')
    else:
        print('Cache file exists but contains no data; recomputing.')

if not cache_loaded:

    # --- loop over dates

    for idate, date in enumerate(cyyyymmddhh_list):
        print ('idate, date = ', idate, date)

        # --- read GRAF from netCDF f/o file.

        if int(date) > 2024040512:
            cmodel_in = 'graf_conus'
        else:
            cmodel_in = 'graflr_conus'

        input_directory = probs_directory + \
            cmodel_in + '/' + date[0:6] + '/'
        input_file = input_directory + cmodel_in + \
            '_1h_probs_multisigma_IC' + \
            date + '_lead' + clead + 'h.nc'

        ny, nx, lats, lons, p0p25mm_raw, p1mm_raw, p5mm_raw, \
            p10mm_raw, sigmas, nsigmas = read_probs(input_file)

        if ny == 0:
            print ('skipping date ', date, ' (prob file missing)')
            continue

        # --- read MRMS data.

        date_forecast = dateshift(date, int(clead))
        observations, data_quality, istat_mrms = read_MRMS(MRMS_directory, date_forecast)

        if istat_mrms != 0:
            print ('skipping date ', date, ' (MRMS file missing)')
            continue

        # --- process each threshold and sigma

        for ithresh, thresh in enumerate(pthresholds):

            for isigma in range(nsigmas):

                # ---- Declare per-date arrays

                contab_raw = np.zeros((ncats,2), dtype=int)

                # --- Populate the data depending on the event threshold.

                if ithresh == 0:
                    raw_prob = p0p25mm_raw[isigma, :,:]
                elif ithresh == 1:
                    raw_prob = p1mm_raw[isigma, :,:]
                    if isigma == 3:
                        p50, p90, p95, p99 = np.percentile(raw_prob, [50, 90, 95, 99])
                        print(f'    GRAF p(>=1mm) sigma[3] percentiles: '
                              f'50th={p50:.4f}, 90th={p90:.4f}, '
                              f'95th={p95:.4f}, 99th={p99:.4f}')
                elif ithresh == 2:
                    raw_prob = p5mm_raw[isigma, :,:]
                elif ithresh == 3:
                    raw_prob = p10mm_raw[isigma, :,:]

                # --- Compute contingency tables and Brier Score

                contab_raw, BS_raw, nsamps_raw = compute_contab_BS(ny, nx,
                    raw_prob, observations, data_quality, contab_raw, ncats, thresh,
                    verbose=(isigma == 3))

                contab_raw_sigmas[ithresh,:,:,isigma] = \
                    contab_raw_sigmas[ithresh,:,:,isigma] + contab_raw[:,:]
                BS_raw_sigmas[ithresh,isigma] = \
                    BS_raw_sigmas[ithresh,isigma] + BS_raw
                nsamps_sigmas[ithresh,isigma] = \
                    nsamps_sigmas[ithresh,isigma] + nsamps_raw

    # --- Save accumulated arrays to cache for future re-runs.

    os.makedirs(relia_save_dir, exist_ok=True)
    print(f'Saving accumulated data to cache: {cache_file}')
    with open(cache_file, 'wb') as f:
        cPickle.dump({'contab_raw_sigmas': contab_raw_sigmas,
                      'BS_raw_sigmas': BS_raw_sigmas,
                      'nsamps_sigmas': nsamps_sigmas}, f)

# --- Calculate frequency of use and reliability from accumulated contab arrays.

for ithresh in range(nthresholds):
    for isigma in range(nsigmas):
        frequse_raw = np.zeros((ncats), dtype=np.float64)
        relia_raw = -99.99*np.ones((ncats), dtype=np.float64)
        contab_use = contab_raw_sigmas[ithresh,:,:,isigma]
        frequse_raw, relia_raw = compute_relia(contab_use,
            ncats, frequse_raw, relia_raw)
        frequse_raw_sigmas[ithresh,:,isigma] = frequse_raw[:]
        relia_raw_sigmas[ithresh,:,isigma] = relia_raw[:]
        if nsamps_sigmas[ithresh, isigma] > 0:
            BS_raw_sigmas[ithresh, isigma] = \
                BS_raw_sigmas[ithresh, isigma] / nsamps_sigmas[ithresh, isigma]

# ----- Make plots of reliability and frequency of usage

colors = ['Red','RoyalBlue','LimeGreen','Violet','Gray','DarkCyan',
          'GoldenRod','Orange','Pink','Indigo','Black','DarkGray']

for ithresh, thresh in enumerate(pthresholds):

    ctthresh = str(thresh)
    ctitle = r''+cleadb+'-'+clead+' h '+cmtit+' reliability, '+\
        r'precip $\geq$ '+ctthresh+r' mm,'+'\n'+\
        cyyyymmddhh_begin+' to '+cyyyymmddhh_end

    print ('making plots for threshold = ', thresh)
    probability = np.arange(11) * 100. / np.real(10.)

    fig = plt.figure(figsize=(5.,5.3))
    a1 = fig.add_axes([.13,.1,.83,.78])
    a1.set_title(ctitle, fontsize=13)
    a1.plot([0,100],[0,100],'--',color='k')
    a1.set_ylabel('Observed Relative Frequency (%)', fontsize=12)
    a1.set_xlabel('Forecast Probability (%)', fontsize=12)
    a1.set_ylim(-1,101)
    a1.set_xlim(-1,101)

    # ---- Select subset of sigmas to plot depending on lead time.

    if int(clead) <= 24:
        isigma_start, isigma_end = 0, nsigmas - 2
    else:
        isigma_start, isigma_end = 2, nsigmas

    a2 = None
    for plot_idx, isigma in enumerate(range(isigma_start, isigma_end)):
        sigma = sigmas[isigma]
        BS = BS_raw_sigmas[ithresh, isigma]
        relia = relia_raw_sigmas[ithresh, :, isigma]
        f = frequse_raw_sigmas[ithresh, :, isigma]
        cbs = "%0.4f"%(BS)
        csigma = "%d"%sigma
        label = r'$\sigma$ = '+csigma+', BS = '+cbs
        color = colors[plot_idx % len(colors)]

        relia_ma = ma.masked_where(f < 1.e-4, relia)
        a1.plot(probability, 100.*relia_ma, 'o-',
            color=color, linewidth=2, label=label)

        # --- Frequency of usage inset diagram

        locn = probability + float((plot_idx - 3))
        if plot_idx == 0:
            a2 = fig.add_axes([.26,.64,.32,.2])
            a2.set_xlim(-4,104)
            a2.set_ylim(0.0001,1.)
            a2.set_title('Frequency of usage', fontsize=9)
            a2.set_xlabel('Forecast probability', fontsize=7)
            a2.set_ylabel('Forecast frequency', fontsize=7)
            a2.hlines([0.001,.01,.1],0,100,linestyles='dashed',colors='gray',lw=0.5)
        a2.bar(locn, f, width=1., bottom=0.0001,
            log=True, color=color, edgecolor='None',
            align='center')

    a1.legend(loc=4, fontsize='xx-small')
    plot_title = 'relia_GRAF_'+cyyyymmddhh_begin+'_to_'+\
        cyyyymmddhh_end+'_'+ctthresh+'mm_'+clead+'h.png'

    print ('saving plot to file = ', plot_title)
    plt.savefig(plot_title, dpi=300)
    plt.close(fig)

