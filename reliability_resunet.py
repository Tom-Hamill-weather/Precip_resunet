"""
python reliability_resunet.py cyyyymmddhh_begin cyyyymmddhh_end clead

e.g., 

python reliability_resunet.py 2025120100 2025123112 12

    cyyyymmddhh_begin = sys.argv[1]
    cyyyymmddhh_end = sys.argv[2]
    clead = sys.argv[3]

This will compute BS, reliability, freq use for the test of 
Attention ResUNet.  

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

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    import os
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]
    GRAFdatadir_conus_laptop = directory["GRAFdatadir_conus_laptop"]
    GRAFprobsdir_conus_laptop = directory["GRAFprobsdir_conus_laptop"]
    GRAF_plot_dir = directory["GRAF_plot_dir"]
    mrms_data_directory = os.path.expanduser(directory["mrms_data_directory"])
    return GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop, \
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

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm= cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) > 2024040512:
        input_directory =  GRAFdatadir_conus_laptop
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_laptop
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

def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus_laptop):

    # Updated to read Gamma model probability files
    infile = GRAFprobsdir_conus_laptop + cyyyymmddhh + \
        '_'+ clead + '_probs_gamma.nc'
    fexist = os.path.exists(infile)
    if fexist == True:
        nc = Dataset(infile,'r')
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        raw_p0p25mm_prob = nc.variables['raw_p0p25mm_prob'][:,:]
        gamma_p0p25mm_prob = nc.variables['gamma_p0p25mm_prob'][:,:]
        raw_p1mm_prob = nc.variables['raw_p1mm_prob'][:,:]
        gamma_p1mm_prob = nc.variables['gamma_p1mm_prob'][:,:]
        raw_p2p5mm_prob = nc.variables['raw_p2p5mm_prob'][:,:]
        gamma_p2p5mm_prob = nc.variables['gamma_p2p5mm_prob'][:,:]
        raw_p5mm_prob = nc.variables['raw_p5mm_prob'][:,:]
        gamma_p5mm_prob = nc.variables['gamma_p5mm_prob'][:,:]
        raw_p10mm_prob = nc.variables['raw_p10mm_prob'][:,:]
        gamma_p10mm_prob = nc.variables['gamma_p10mm_prob'][:,:]
        #raw_p25mm_prob = nc.variables['raw_p25mm_prob'][:,:]
        #gamma_p25mm_prob = nc.variables['gamma_p25mm_prob'][:,:]
        #print ('max raw Gamma 0p25mm = ', \
        #    np.max(raw_p0p25mm_prob), np.max(gamma_p0p25mm_prob))
        #print ('max raw Gamma 1mm = ', \
        #    np.max(raw_p1mm_prob), np.max(gamma_p1mm_prob))
        #print ('max raw Gamma 2p5mm = ', \
        #    np.max(raw_p2p5mm_prob), np.max(gamma_p2p5mm_prob))
        #print ('max raw Gamma 5mm = ', \
        #    np.max(raw_p5mm_prob), np.max(gamma_p5mm_prob))
        #print ('max raw Gamma 10mm = ', \
        #    np.max(raw_p10mm_prob), np.max(gamma_p10mm_prob))
        #print ('max raw Gamma 25mm = ', \
        #    np.max(raw_p25mm_prob), np.max(gamma_p25mm_prob))
        nc.close()
        istat_prob = 0
    else:
        print (infile)
        print ('no such file exists.')
        istat_prob = -1
        lat = np.empty((0,0), dtype=float)
        lon = np.empty((0,0), dtype=float)
        raw_p0p25mm_prob = np.empty((0,0), dtype=float)
        gamma_p0p25mm_prob = np.empty((0,0), dtype=float)
        raw_p1mm_prob = np.empty((0,0), dtype=float)
        gamma_p1mm_prob = np.empty((0,0), dtype=float)
        raw_p2p5mm_prob = np.empty((0,0), dtype=float)
        gamma_p2p5mm_prob = np.empty((0,0), dtype=float)
        raw_p5mm_prob = np.empty((0,0), dtype=float)
        gamma_p5mm_prob = np.empty((0,0), dtype=float)
        raw_p10mm_prob = np.empty((0,0), dtype=float)
        gamma_p10mm_prob = np.empty((0,0), dtype=float)
        #raw_p25mm_prob = np.empty((0,0), dtype=float)
        #gamma_p25mm_prob = np.empty((0,0), dtype=float)
        
    return istat_prob, raw_p0p25mm_prob, gamma_p0p25mm_prob, raw_p1mm_prob, \
        gamma_p1mm_prob, raw_p2p5mm_prob, gamma_p2p5mm_prob, raw_p5mm_prob, \
        gamma_p5mm_prob, raw_p10mm_prob, gamma_p10mm_prob, lat, lon

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

    """ for this case day, compute the contingency table
        elements that are later used to calculate
        reliability and frequency of use. """

    # ---- convert observation to binary.  Only use obs
    #      whose verification time align with the forecast time.

    #print ('np.shape(coslat) = ', np.shape(coslat))
    ones = np.ones((nstns), dtype=int)
    zeros = np.zeros((nstns), dtype=int)
    contab = np.zeros((ncats, 2), dtype=float)
    fmean = np.zeros((ncats), dtype=float)

    #print ('nstns, nsamps = ', nstns, nsamps)
    binary_obs = np.where(obs < threshold, zeros, ones)

    # ---- for this category (range of forecast probabilities)
    #      find all the cases with probabilities in this range,
    #      and then populate the contingency table based on
    #      whether the observation was above the threshold or not.

    for icat in range(ncats):

        # ---- saved for 11 categories given 31 ens mbrs.

        pmin = np.max([0.0, float(icat)/(ncats-1) - 1./(2*(ncats-1))])
        pmax = np.min([1.0, float(icat)/(ncats-1) + 1./(2*(ncats-1))])
        
        a = np.where(np.logical_and(prob >= pmin, prob < pmax))[0]
        fmean[icat] = np.mean(prob[a])
        binary_fcst = np.where(np.logical_and ( \
            prob >= pmin, prob < pmax), ones, zeros)
        a = np.where(np.logical_and (binary_fcst == 1, binary_obs == 1))[0]
        if len(a) > 0:
            contab[icat,1] = contab[icat,1] + np.sum(coslat[a])
        a = np.where(np.logical_and (binary_fcst == 1, binary_obs == 0))[0]
        if len(a) > 0:
            contab[icat,0] = contab[icat,0] + np.sum(coslat[a])

    # ---- now compute Brier Score.

    nsamps = 0
    a = np.where(binary_obs == 0)[0]
    if len(a) > 0:
        BS = np.sum(coslat[a]*(prob[a])**2)
        nsamps = nsamps + np.sum(coslat[a])
    a = np.where(binary_obs == 1)[0]
    if len(a) > 0:
        BS = BS + np.sum(coslat[a]*(1.-prob[a])**2)
        nsamps = nsamps + np.sum(coslat[a])
    
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

config_file_name = 'config_laptop.ini'
GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop, \
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

# --- loop over dates

inits = False
good_dates = np.zeros((ndates), dtype=int)
for idate, date in enumerate(cyyyymmddhh_list):
    print ('-------- idate, date = ', idate, date)
    validity_date = dateshift(date, int(clead))
    
    # --- Read the raw GRAF forecast in 
    #     /Users/tom.hamill@weather.com/python/resnet_data/GRAF/YYYYMMDD/HH
    #     and return the GRAF hourly precipitation forecast amount and
    #     lat/lon.  Compute an array of cosine of latitude too (call this coslat)
    
    istat_GRAF, precipitation_GRAF, lats_GRAF, lons_GRAF, \
        ny_GRAF, nx_GRAF, latmin, latmax, lonmin, lonmax, \
        verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
        GRAF_precip_read(clead, date, GRAFdatadir_conus_laptop)
    coslat = np.cos(lats_GRAF*3.1415926/180.)
    
    if istat_GRAF == 0: 
        coslat_save = np.copy(coslat)
        lons_save = np.copy(lons_GRAF)
        lats_save = np.copy(lats_GRAF)
        
    if idate == 0: # declare arrays, set to missing
    
        lats_all =  \
            -999.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        lons_all =  \
            -999.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        raw_ensemble_p0p25mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        raw_ensemble_p1mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        raw_ensemble_p2p5mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float) 
        raw_ensemble_p5mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        raw_ensemble_p10mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float) 
        raw_ensemble_p25mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        
        gamma_p0p25mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        gamma_p1mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float) 
        gamma_p2p5mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        gamma_p5mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        gamma_p10mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float) 
        gamma_p25mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
            
        coslat_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        
        MRMS_precip_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p0p25mm_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p1mm_all = \
            -99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p2p5mm_all = \
            -99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p5mm_all = \
            -99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p10mm_all = \
            -99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_binary_p25mm_all = \
            -99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
        MRMS_data_quality_all = \
            -99.99*np.ones((ndates, ny_GRAF, nx_GRAF), dtype=float)
            
        BS_raw = np.zeros((nthresholds), dtype=float)
        BS_gamma = np.zeros((nthresholds), dtype=float)
        nsamps_raw = np.zeros((nthresholds), dtype=float)
        nsamps_gamma = np.zeros((nthresholds), dtype=float)
    
    # --- Read previously generated raw and gamma-derived probabilities
    
    istat_prob, raw_p0p25mm_prob, gamma_p0p25mm_prob, raw_p1mm_prob, \
        gamma_p1mm_prob, raw_p2p5mm_prob, gamma_p2p5mm_prob, raw_p5mm_prob, \
        gamma_p5mm_prob, raw_p10mm_prob, gamma_p10mm_prob, lat, lon = \
        probability_read(clead, date, GRAFprobsdir_conus_laptop)
        
    # ---- Read MRMS hourly accumulated precip and data quality
    #      as surrogate for observed.
    
    istat_MRMS, MRMS_precip, MRMS_quality = \
        read_MRMS(mrms_data_directory, validity_date)
            
    # ---- If all files available, overwrite missing -99.99 with 
    #      real data.
    
    print ('istat_MRMS, istat_GRAF, istat_prob = ', istat_MRMS, istat_GRAF, istat_prob)
    if istat_MRMS == 0 and istat_GRAF == 0 and istat_prob == 0:
        good_dates[idate] = 1
        
        lats_all[idate,:,:] = lats_save[:,:]
        lons_all[idate,:,:] = lons_save[:,:]
        
        raw_ensemble_p0p25mm_all[idate,:,:] = raw_p0p25mm_prob[:,:]
        raw_ensemble_p1mm_all[idate,:,:] = raw_p1mm_prob[:,:]
        raw_ensemble_p2p5mm_all[idate,:,:] = raw_p2p5mm_prob[:,:]
        raw_ensemble_p5mm_all[idate,:,:] = raw_p5mm_prob[:,:]
        raw_ensemble_p10mm_all[idate,:,:] = raw_p10mm_prob[:,:]
        #raw_ensemble_p25mm_all[idate,:,:] = raw_p25mm_prob[:,:]

        gamma_p0p25mm_all[idate,:,:] = gamma_p0p25mm_prob[:,:]
        gamma_p1mm_all[idate,:,:] = gamma_p1mm_prob[:,:]
        gamma_p2p5mm_all[idate,:,:] = gamma_p2p5mm_prob[:,:]
        gamma_p5mm_all[idate,:,:] = gamma_p5mm_prob[:,:]
        gamma_p10mm_all[idate,:,:] = gamma_p10mm_prob[:,:]
        #gamma_p25mm_all[idate,:,:] = gamma_p25mm_prob[:,:]

        MRMS_precip_all[idate,:,:] = MRMS_precip[:,:]
        MRMS_data_quality_all[idate,:,:] = MRMS_quality[:,:]
        
        MRMS_binary_p0p25mm = np.where(MRMS_precip >= 0.25, 1, 0)
        MRMS_binary_p1mm = np.where(MRMS_precip >= 1.0, 1, 0)
        MRMS_binary_p2p5mm = np.where(MRMS_precip >= 2.5, 1, 0)
        MRMS_binary_p5mm = np.where(MRMS_precip >= 5.0, 1, 0)
        MRMS_binary_p10mm = np.where(MRMS_precip >= 10.0, 1, 0)
        #MRMS_binary_p25mm = np.where(MRMS_precip >= 25.0, 1, 0)

        MRMS_binary_p0p25mm_all[idate,:,:] = MRMS_binary_p0p25mm[:,:]
        MRMS_binary_p1mm_all[idate,:,:] = MRMS_binary_p1mm[:,:]
        MRMS_binary_p2p5mm_all[idate,:,:] = MRMS_binary_p2p5mm[:,:]
        MRMS_binary_p5mm_all[idate,:,:] = MRMS_binary_p5mm[:,:]
        MRMS_binary_p10mm_all[idate,:,:] = MRMS_binary_p10mm[:,:]
        #MRMS_binary_p25mm_all[idate,:,:] = MRMS_binary_p25mm[:,:]

        coslat_all[idate,:,:] = coslat[:,:]
        a = np.where(MRMS_quality > 0.5)[0]

# --- Compact, getting rid of dates without all the data.

nbad = len(good_dates) - np.sum(good_dates)
date_indices = range(ndates)
ones = np.ones((ndates), dtype=int)
date_indices_to_delete = []
for idate in range(ndates):
    if good_dates[idate] == 0: date_indices_to_delete.append(idate)
nbad = len(date_indices_to_delete)
if nbad > 0:
    
    lats_all = np.delete(lats_all, date_indices_to_delete, axis=0)
    lons_all = np.delete(lons_all, date_indices_to_delete, axis=0)
    coslat_all = np.delete(coslat_all, date_indices_to_delete, axis=0)
        
    raw_ensemble_p0p25mm_all = np.delete(raw_ensemble_p0p25mm_all, \
        date_indices_to_delete, axis=0)
    raw_ensemble_p1mm_all = np.delete(raw_ensemble_p1mm_all, \
        date_indices_to_delete, axis=0)
    raw_ensemble_p2p5mm_all = np.delete(raw_ensemble_p2p5mm_all, \
        date_indices_to_delete, axis=0)
    raw_ensemble_p5mm_all = np.delete(raw_ensemble_p5mm_all, \
        date_indices_to_delete, axis=0)
    raw_ensemble_p10mm_all = np.delete(raw_ensemble_p10mm_all, \
        date_indices_to_delete, axis=0)
    #raw_ensemble_p25mm_all = np.delete(raw_ensemble_p25mm_all, \
    #    date_indices_to_delete, axis=0)
        
    gamma_p0p25mm_all = np.delete(gamma_p0p25mm_all, \
        date_indices_to_delete, axis=0)
    gamma_p1mm_all = np.delete(gamma_p1mm_all, \
        date_indices_to_delete, axis=0)
    gamma_p2p5mm_all = np.delete(gamma_p2p5mm_all, \
        date_indices_to_delete, axis=0)
    gamma_p5mm_all = np.delete(gamma_p5mm_all, \
        date_indices_to_delete, axis=0)
    gamma_p10mm_all = np.delete(gamma_p10mm_all, \
        date_indices_to_delete, axis=0)
    #gamma_p25mm_all = np.delete(gamma_p25mm_all, \
    #    date_indices_to_delete, axis=0)
        
    MRMS_precip_all = np.delete(MRMS_precip_all, \
        date_indices_to_delete, axis=0)
    MRMS_binary_p0p25mm_all = np.delete(MRMS_binary_p0p25mm_all, \
        date_indices_to_delete, axis=0)
    MRMS_binary_p1mm_all = np.delete(MRMS_binary_p1mm_all, \
        date_indices_to_delete, axis=0)
    MRMS_binary_p2p5mm_all = np.delete(MRMS_binary_p2p5mm_all, \
        date_indices_to_delete, axis=0)
    MRMS_binary_p5mm_all = np.delete(MRMS_binary_p5mm_all, \
        date_indices_to_delete, axis=0)
    MRMS_binary_p10mm_all = np.delete(MRMS_binary_p10mm_all, \
        date_indices_to_delete, axis=0)
    #MRMS_binary_p25mm_all = np.delete(MRMS_binary_p25mm_all, \
    #    date_indices_to_delete, axis=0)
    MRMS_data_quality_all = np.delete(MRMS_data_quality_all, \
        date_indices_to_delete, axis=0)
    print ('date_indices_to_delete = ',date_indices_to_delete)
    print (np.shape(coslat_all))
    ndates, ny, nx = np.shape(MRMS_precip_all)
    #print ('ndates after elimination of dates with incomplete: ', ndates)

# ---- Process this threshold

for ithresh, thresh in enumerate(pthresholds):
    
    print ('Processing threshold = ', thresh)
    lats = lats_all.flatten()
    lons = lons_all.flatten()
    MRMS_pre = MRMS_precip_all.flatten()
    MRMS_dq = MRMS_data_quality_all.flatten()
    coslat_flat = coslat_all.flatten()
    if ithresh == 0:
        prob_forecast_raw = raw_ensemble_p0p25mm_all.flatten()
        prob_forecast_gamma = gamma_p0p25mm_all.flatten()
    elif ithresh == 1:
        prob_forecast_raw = raw_ensemble_p1mm_all.flatten()
        prob_forecast_gamma = gamma_p1mm_all.flatten()
    elif ithresh == 2:
        prob_forecast_raw = raw_ensemble_p2p5mm_all.flatten()
        prob_forecast_gamma = gamma_p2p5mm_all.flatten()
    elif ithresh == 3:
        prob_forecast_raw = raw_ensemble_p5mm_all.flatten()
        prob_forecast_gamma = gamma_p5mm_all.flatten()
    elif ithresh == 4:
        prob_forecast_raw = raw_ensemble_p10mm_all.flatten()
        prob_forecast_gamma = gamma_p10mm_all.flatten()

    # ---- thin the data to where MRMS observations are 
    #      >= 0.0 and data quality >= 0.5.  Obs >= 0 should cover
    #      extant forecasts
    
    print ('  Thinning to points with data quality > 0.5')
    a = np.where(np.logical_and(np.logical_and(\
        MRMS_pre >= 0.0, MRMS_dq >= 0.5), MRMS_pre < 100.))[0]
    nobs = len(a)
    lats = lats[a]
    lons = lons[a]
    observations = MRMS_pre[a]
    prob_forecast_raw = prob_forecast_raw[a]
    prob_forecast_gamma = prob_forecast_gamma[a]
    coslat_flat = coslat_flat[a]
    
    # --- contingency tables for raw

    print ('  Computing contingency table for Raw')
    contab_raw[ithresh,:,:], BS_raw[ithresh], \
        nsamps_raw[ithresh], fmean_raw = \
        compute_contab_BS(nobs, prob_forecast_raw, \
        observations, ncats, thresh, coslat_flat)
    print ('fmean raw: ', fmean_raw)
        
    print ('  Computing contingency table for Gamma')
    contab_gamma[ithresh,:,:], BS_gamma[ithresh], \
        nsamps_gamma[ithresh], fmean_gamma = \
        compute_contab_BS(nobs, prob_forecast_gamma, \
        observations, ncats, thresh, coslat_flat)
    print ('fmean gamma: ', fmean_gamma)
        
    
    cthresh = r'P(obs $\geq$ '+str(thresh) + ' mm)'
    ctthresh = str(thresh)+'mm'

    # --- Calculate frequency of use and reliability for raw

    frequse_raw, relia_raw = compute_relia(contab_raw[ithresh,:,:], ncats)
    BS_raw[ithresh] = BS_raw[ithresh] / float(nsamps_raw[ithresh])
    
    # --- Calculate frequency of use and reliability for Gamma

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
            label='Fitted Gamma probability, BS = '+cbs
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
    plot_title = 'Relia_GRAF_ResUNet_MRMS_' + \
        cyyyymmddhh_list[0] + '_to_' + cyyyymmddhh_list[-1] + '_' + \
        ctthresh + '_' + clead + 'h.png'
    print ('  Saving plot to file = ',plot_title)
    plt.savefig(plot_title, dpi=300)
    
