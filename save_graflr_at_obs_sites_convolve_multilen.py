"""
python save_graflr_at_obs_sites_convolve_multilen.py cyyyymmddhh clead

where 

cyyyymmddhh is the initial time.
clead is the lead time in hours.

"""
import numpy as np
import numpy.ma as ma
import pygrib
from dateutils import daterange, dateshift
import _pickle as cPickle
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, stringtochar
from datetime import datetime, timedelta, timezone
import argparse
from pathlib import Path
import scipy.stats as stats
np.set_printoptions(precision=3, suppress=True)
from matplotlib import rcParams
import pygeohash as pgh
import scipy.ndimage as ndimage
rcParams['ytick.labelsize']=7
rcParams['xtick.labelsize']=7

# =====================================================================

def find_nearest(array, value):
    """
    find index in array nearest to value
    """
    import numpy as np
    array = np.asarray(array)
    diff = np.abs(array - value)
    idx = np.argmin(diff)
    return idx

# ============================================================

def read_config_file_extract_6h_of_empirics(config_file,
        directory_object_name):

    """ read appropriate information from the config file
        and return.   This version for o_at_f, i.e.,
        obseved at forecast (no MSWEP climatology now).
    """
    from configparser import ConfigParser

    # ---- Read config.ini file

    config_object = ConfigParser()
    config_object.read(config_file)

    # ---- Get the information from dictionary structure

    directory = config_object[directory_object_name]
    netcdf_6h_obs_data_directory = \
        directory["netcdf_6h_obs_data_directory"]
    netcdf_directory_o_at_f_more = \
        directory["netcdf_directory_o_at_f_more"]
    matlab_forecast_directory = \
        directory["matlab_forecast_directory"]
    empirics_data_directory = \
        directory["empirics_data_directory"]

    GRAFdatadir_conus_old = \
        directory["GRAFdatadir_conus_old"]
    GRAFdatadir_conus_new = \
        directory["GRAFdatadir_conus_new"]

    return netcdf_6h_obs_data_directory, \
        netcdf_directory_o_at_f_more, matlab_forecast_directory, \
        empirics_data_directory, GRAFdatadir_conus_old, \
        GRAFdatadir_conus_new

# ============================================================

def return_empirics_lat_lon_new(empirics_data_directory):
    """
    the matlab data in the forecast file does not contain
    latitude and longitude.   Read a file with a listing of the
    locations of empirics sites.  Return these, as well
    as lon, lat, station names, skyids as python lists.

    """

    empirics_site_list = return_empirics_site_list(empirics_data_directory)
    nstns = len(empirics_site_list)
    lons_stns = np.array(empirics_site_list['lon'].tolist())
    lats_stns = np.array(empirics_site_list['lat'].tolist())
    stnames = np.array(empirics_site_list['stn_name'].tolist())
    stnids = np.array(empirics_site_list['skyID'].tolist())

    # ---- sort arrays by skyid

    argsort = np.argsort(stnids)
    lons_stns = lons_stns[argsort]
    lats_stns = lats_stns[argsort]
    stnames = stnames[argsort]
    stnids = stnids[argsort]

    return nstns, lons_stns, lats_stns, stnames, stnids

# ============================================================

def return_empirics_site_list(empirics_data_directory):
    """
    Read a file with a listing of the locations of empirics sites.
    Return the associated pandas dataframe.

    Coded by Lauriana Gaudet 10 March 2023
    """
    import pandas as pd
    fname = empirics_data_directory + 'empirics_site_list.csv'
    print ('fname = ', fname)
    print(f'INFO: reading empirics from {fname}')
    col_names = ['old_dicast_ID', 'primary_ID', 'secondary_ID', 'lat', 'lon',
             'elev [m]', 'primary_site_type', 'stn_name', 'state','country',
             'unknown1', 'unknown2', 'unknown3', 'unknown4', 'skyID']
    empirics_site_list = pd.read_csv(\
        f'{empirics_data_directory}empirics_site_list.csv', \
        sep=';', header=None, names=col_names)
    return empirics_site_list
    
# ============================================================

def build_infile_obs_name(
    netcdf_6h_obs_data_directory, cyyyymmddhh_begin,
    cyyyymmddhh_end):

    """
    the input file name will be different depending on whether
    this script is run on lenovo1 or laptop.   Handle this,
    returning the file name with correct directory path.
    """

    data_directory_in = netcdf_6h_obs_data_directory
    infile_obs = data_directory_in + \
        cyyyymmddhh_begin + '_to_' + cyyyymmddhh_end + \
        '_6hourly_precip.nc'
    return infile_obs

# ============================================================

def build_datestrings(printit, cyyyymmddhh_end,
        netcdf_6h_obs_data_directory):

    """ build year/month/day/hour date strings for start, end
    of 24-h observation period.
    """

    from dateutils import dateshift
    import os, sys

    if printit == True:
        print ('INFO: Date of the end of the forecast period we '+\
        'are trying to match to obs: ',cyyyymmddhh_end)
    fexist_dayb = True
    if cyyyymmddhh_end[8:10] == '06':
        cyyyymmddhh_endday = dateshift(cyyyymmddhh_end,18)
    elif cyyyymmddhh_end[8:10] == '12':
        cyyyymmddhh_endday = dateshift(cyyyymmddhh_end,12)
    elif cyyyymmddhh_end[8:10] == '18':
        cyyyymmddhh_endday = dateshift(cyyyymmddhh_end,6)
    else: # we need to read earlier day's obs, too.
        # build datestrings for making those filenames
        cyyyymmddhh_endday = dateshift(cyyyymmddhh_end,24)
    cyyyymmddhh_begin = dateshift(cyyyymmddhh_endday, -24)
    print (cyyyymmddhh_endday)

    return  cyyyymmddhh_begin, cyyyymmddhh_endday

# ============================================================

def read_obs_data_and_geohash_them(infile_obs, cyyyymmddhh_valid):

    """
    read the 6-h accumulated precipitation observations from
    the netCDF file produced by save_obs_6hourly_withbool.py.
    Return the list of observations, their location,
    valid times, skyids, # observations and from the lat/lon,
    a geohash for later use in co-location with forecast id'ed
    by geohash.
    """
    
    import os, sys
    from netCDF4 import Dataset
    import numpy as np
    import pygeohash as pgh
    np.set_printoptions(precision=5, suppress=True)

    fexist = os.path.exists(infile_obs)
    if fexist == True:
        print(f'INFO: Reading 6-h observation file {infile_obs}')
        nc = Dataset(infile_obs,'r')
        lons_obs = nc.variables['lons'][:]
        lats_obs = nc.variables['lats'][:]
        precipitation_observations_6h = \
            nc.variables['precipitation'][:]
        validtimes_yyyymmddhh_end = \
            nc.variables['validtimes_yyyymmddhh'][:]
        skyids_obs = nc.variables['skyid'][:]
        obstype_flag = nc.variables['obstype_flag'][:]

        # --- Code below is designed to exclude the Boolean wet
        #     observations from the training data and validation data.
        #     Depending on use case, you may wish to exclude this.

        a = np.where(obstype_flag < 3)[0]
        lons_obs = lons_obs[a]
        lats_obs = lats_obs[a]
        precipitation_observations_6h = precipitation_observations_6h[a]
        validtimes_yyyymmddhh_end = validtimes_yyyymmddhh_end[a]
        skyids_obs = skyids_obs[a]
        nobs = len(skyids_obs)
        na = len(a)
        lons_obs = np.around(lons_obs, decimals=3)
        lats_obs = np.around(lats_obs, decimals=3)

        # --- The forecast data against which we will compare will be
        #     geolocated using geohash with 9 characters instead of
        #     lat/lon.   For later matching, convert the lat/lon to geohash.

        geohashes = []
        for i in range(na):
            geohash = pgh.encode(latitude=lats_obs[i], \
                longitude=lons_obs[i], precision=9)
            geohashes.append(geohash)
        geohashes = np.array(geohashes)
        argsort = np.argsort(geohashes)
        lons_obs = lons_obs[argsort]
        lats_obs = lats_obs[argsort]
        geohashes = geohashes[argsort]
        precipitation_observations_6h = precipitation_observations_6h[argsort]
        validtimes_yyyymmddhh_end = validtimes_yyyymmddhh_end[argsort]
        skyids_obs = skyids_obs[argsort]
        print(f'INFO: Number of 6-h observations in 24-h period = {nobs}')
        nc.close()
        
        # ---- thin to just those with the correct valid time.
        
        a = np.where(validtimes_yyyymmddhh_end == \
            int(cyyyymmddhh_valid))[0]
        if len(a) > 0.:
            lons_obs = lons_obs[a]
            lats_obs = lats_obs[a]
            precipitation_observations_6h = precipitation_observations_6h[a]
            validtimes_yyyymmddhh_end = validtimes_yyyymmddhh_end[a]
            skyids_obs = skyids_obs[a]
            geohashes = geohashes[a]
            nobs = len(lons_obs)
        istat = 0
    else:
        print(f'INFO: {infile_obs} does not exist. Quitting.')
        sys.exit()

    return istat, lons_obs, lats_obs, precipitation_observations_6h,\
        validtimes_yyyymmddhh_end, skyids_obs, geohashes, nobs

# ============================================================

def read_obs_from_ecmwf_file(netcdf_directory_o_at_f_more, \
        date, clead, cyyyymmddhh_valid):

    # --- Read in the observed at empirics locations for this date from a 
    #     previously generated of co-located ECMWF and observations file.
    #     Check to make sure that the valid time is a synoptic time; 
    #     If not, don't bother using by setting observation to -99.99.

    cyyyymm = date[0:6]
    input_directory = netcdf_directory_o_at_f_more + \
        'ecmwf_ensmean/extra/' + cyyyymm + '/'
    input_file = input_directory + \
        'ecmwf_ensmean_6h_empirics_forecast_and_obs_IC' + \
        date + '_lead' + clead + 'h.nc'
    fexist2 = os.path.exists(input_file)
    print ('input_file = ', input_file, fexist2)
    if fexist2 == True:
        nc = Dataset(input_file, 'r')
        observed = \
            nc.variables['precipitation_observation'][:]
        validtimes_yyyymmddhh = \
            nc.variables['validtimes_yyyymmddhh'][:]
        lons_stns = nc.variables['lons'][:]
        lats_stns = nc.variables['lats'][:]
        skyids = nc.variables['skyid'][:]
        nstns = len(skyids)
        a = np.where(validtimes_yyyymmddhh != \
            int(cyyyymmddhh_valid))[0]
        if len(a) > 0.:
            observed[a] = -99.99
        nc.close()  
    
        # --- The forecast data against which we will 
        #     compare will be geolocated using geohash with
        #     9 characters instead of lat/lon.   For later matching
        #     convert the lat/lon to geohash.

        geohashes = []
        for i in range(nstns):
            geohash = pgh.encode(latitude=lats_stns[i], \
                longitude=lons_stns[i], precision=9)
            geohashes.append(geohash)
        geohashes = np.array(geohashes)
        argsort = np.argsort(geohashes)
        lons_stns = lons_stns[argsort]
        lats_stns = lats_stns[argsort]
        geohashes = geohashes[argsort]
        observed = observed[argsort]
        validtimes_yyyymmddhh = validtimes_yyyymmddhh[argsort]
        skyids = skyids[argsort]
        print(f'INFO: Number of 6-h observations in 24-h period = {nstns}')
        istat = 0
    else:
        nstns = 0
        geohashes = np.empty((0), dtype=int)
        lons_stns = np.empty((0), dtype=float)
        lats_stns = np.empty((0), dtype=float)
        observed = np.empty((0), dtype=float)
        validtimes_yyyymmddhh = np.empty((0), dtype=int)
        skyids = np.empty((0), dtype=int)
        istat = -1
    
    return istat, nstns, geohashes, lons_stns, lats_stns, observed, \
        validtimes_yyyymmddhh, skyids

# ========================================================

def read_gribdata(gribfilename, endStep):

    """ read grib data"""

    import os
    import pygrib

    istat = -1
    fexist_grib = False
    fexist_grib = os.path.exists(gribfilename)
    #print ('   reading ',gribfilename, fexist_grib)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep = endStep)[0]
            lats, lons = grb.latlons()
            precipitation = grb.values
            istat = 0
            fcstfile.close()
        except IOError:
            print ('   IOError in read_gribdata reading ', \
                gribfilename)
            istat = -1
        except ValueError:
            print ('   ValueError in read_gribdata reading ', \
                gribfilename)
            istat = -1
        except RuntimeError:
            print ('   RuntimeError in read_gribdata reading ', \
                gribfilename)
            istat = -1
    else:
        print ('grib file does not exist.')
        
    return istat, precipitation, lats, lons

# ============================================================

def find_nearest_graf_gps_v2(lats, lons, ny, nx, nstns, nskip, \
        lats_stns, lons_stns, latmin, latmax, lonmin, lonmax,\
        empirics_data_directory, skyids_stns):


    j_nearest = np.zeros((nstns), dtype=int)
    i_nearest = np.zeros((nstns), dtype=int)
    
    old_algorithm = False
    if old_algorithm == True:
        
        for istn in range(nstns):
            
            if lats_stns[istn] > latmin and lats_stns[istn] < latmax and \
                lons_stns[istn] > lonmin and lons_stns[istn] < lonmax:
            
                # First, find the index of the grid point nearest a specific lat/lon.   
                abslat = np.abs(lats-lats_stns[istn])
                abslon = np.abs(lons-lons_stns[istn])
                c = np.maximum(abslat, abslon)
                cmin = np.min(c)
                if istn%500 == 0:
                    print ('---- istn, cmin = ',istn, cmin, lats_stns[istn], lons_stns[istn])
                if cmin < 0.05:
                    (jne,ine) = np.where(c == np.min(c))
                    if ine > 3*nskip+1 and ine < nx-1-3*nskip and \
                    jne > 3*nskip and jne < ny-1-3*nskip: 
                        #  --- then whole 3 spatial standard deviations 
                        #      are fully inside GRAF domain
                        j_nearest[istn] = jne
                        i_nearest[istn] = ine
                    else:
                        j_nearest[istn] = -99
                        i_nearest[istn] = -99
                else: 
                    j_nearest[istn] = -99
                    i_nearest[istn] = -99
            else:
                j_nearest[istn] = -99
                i_nearest[istn] = -99
    
        a = np.where(i_nearest >= 0)[0]
        print ('number of obs sites in GRAF domain: ', len(a))
        print ('len(i_nearest) = ', len(i_nearest))
        
    else:
        
        # --- these were previously calculated in find_empirics_nearest/
        #     identify_graf_gridpts_nearest_to_obslocs.py
        
        infile = empirics_data_directory + 'GRAF_CONUS_ijlocations.nc'
        print ('infile = ', infile)
        nc = Dataset(infile,'r')
        skyids_allSM = nc.variables['skyid'][:] # all SYNOP/METAR
        i_nearest_allSM = nc.variables['i_nearest'][:]
        j_nearest_allSM = nc.variables['j_nearest'][:]
        nstns_allSM = len(skyids_allSM)
        nc.close()
        
        for istn in range(nstns):
            a = np.where(skyids_allSM == skyids_stns[istn])[0]
            #print ('istn, skyid, a = ', istn, skyids_stns[istn], a[0], ma.is_masked(a[0]))
            if len(a) == 0:
                i_nearest[istn] = -99
                j_nearest[istn] = -99
            elif len(a) == 1:
                i_nearest[istn] = i_nearest_allSM[a[0]]
                j_nearest[istn] = j_nearest_allSM[a[0]]
        print ('max i_nearest = ', np.max(i_nearest))
                
    return j_nearest, i_nearest

# ============================================================

def compute_graf_mean_probs(ny, nx, nstns,  \
    precipitation_forecast, \
    pmean_via_convolution, POP_via_convolution, \
    p1mm_via_convolution, p5mm_via_convolution, \
    p10mm_via_convolution,j_nearest, i_nearest):
        
    """
    we compute various pseudo-ensemble statistics from GRAF,
    randomly sampling around each station location.  We test
    various length scales for sampling, provided in vars_list.
    
    """
          
    mean = np.array([0.0,0.0],dtype=float)
    graf_precipitation = -99.99 * np.ones((nstns), dtype=float)
    graf_mean = -99.99 * np.ones((nstns), dtype=float) \
        # 5 for 5 diff cov length scales
    POP = -99.99 * np.ones((nstns), dtype=float)
    p1mm = -99.99 * np.ones((nstns), dtype=float)
    p5mm = -99.99 * np.ones((nstns), dtype=float)
    p10mm = -99.99 * np.ones((nstns), dtype=float)
    
    for istn in range(nstns):
        if i_nearest[istn] > 0:
            graf_precipitation[istn] = \
                precipitation_forecast[j_nearest[istn],i_nearest[istn]]
            graf_mean[istn] = \
                pmean_via_convolution[j_nearest[istn],i_nearest[istn]]
            POP[istn] = \
                POP_via_convolution[j_nearest[istn],i_nearest[istn]]
            p1mm[istn] = \
                p1mm_via_convolution[j_nearest[istn],i_nearest[istn]]
            p5mm[istn] = \
                p5mm_via_convolution[j_nearest[istn],i_nearest[istn]]
            p10mm[istn] = \
                p10mm_via_convolution[j_nearest[istn],i_nearest[istn]]
                
    return graf_precipitation, graf_mean, POP, p1mm, p5mm, p10mm
 
# ============================================================

def write_fco_to_netCDF_mean_pops(outfile, nstns, skyids, \
        validtimes_yyyymmddhh, precipitation_forecast, \
        precipitation_mean_multisigma, POP_multisigma, \
        p1mm_multisigma, p5mm_multisigma, p10mm_multisigma, \
        observed, lons_stns, lats_stns, nsigmas, sigmas):

    """
    write the coincident 6-h forecast and observations accum.
    to a netCDF file. Include the TWC skyid identifier,
    and latitude / longitude, and the end valid time
    of the accumulation period in yyyymmddhhh format.
    Include a date string of the forecast model name.
    """
    from netCDF4 import Dataset, stringtochar
    import numpy as np

    print ('writing to ', outfile)
    ncout = Dataset(outfile,'w')

    ncout.createDimension('nstns', nstns)
    ncout.createDimension('nsigma', nsigma)
    
    sigmas_out = ncout.createVariable('sigmas','float',('nsigma',),\
        zlib=True,least_significant_digit=3)
    sigmas_out.long_name = "convolution spatial std dev in grid pts"
    sigmas_out.units = "grid pts"
    sigmas_out.valid_range = [0.0, 200.0]
    sigmas_out.missing_value = np.array(-99.99,dtype=float)
    
    lons_out = ncout.createVariable('lons','float',('nstns',),\
        zlib=True,least_significant_digit=3)
    lons_out.long_name = "longitude (negative for degrees west)"
    lons_out.units = "degrees_east"
    lons_out.valid_range = [-180.0,180.0]
    lons_out.missing_value = np.array(-99.99,dtype=float)

    lats_out = ncout.createVariable('lats','float',('nstns',),\
        zlib=True,least_significant_digit=3)
    lats_out.long_name = "latitude (negative for S. Hem)"
    lats_out.units = "degrees north"
    lats_out.valid_range = [-90.0,90.0]
    lats_out.missing_value = np.array(-99.99,dtype=float)

    precipitation_forecast_out = ncout.createVariable(\
        'precipitation_forecast','float',('nstns'),\
        zlib=True,least_significant_digit=3)
    precipitation_forecast_out.long_name = \
        "point precipitation forecast accumulation in 6 h"
    precipitation_forecast_out.units = "mm"
    precipitation_forecast_out.valid_range = [0.0,1000.0]
    precipitation_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)
    
    precipitation_mean_out = ncout.createVariable(\
        'precipitation_mean','float',('nsigma','nstns'),\
        zlib=True,least_significant_digit=3)
    precipitation_mean_out.long_name = \
        "convolved spatial mean precipitation "+\
        "forecast accumulation in 6 h"
    precipitation_mean_out.units = "mm"
    precipitation_mean_out.valid_range = [0.0,1000.0]
    precipitation_mean_out.missing_value = \
        np.array(-99.99,dtype=float)

    pop_raw_forecast_out = ncout.createVariable(\
        'pop_raw_forecast','float',('nsigma','nstns'),\
        zlib=True,least_significant_digit=3)
    pop_raw_forecast_out.long_name = \
        "POP forecast from raw convolved ens"+\
        "relative frequency, 0.254-mm threshold"
    pop_raw_forecast_out.units = "fraction"
    pop_raw_forecast_out.valid_range = [0.0,1.0]
    pop_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)
        
    p1mm_raw_forecast_out = ncout.createVariable(\
        'p1mm_raw_forecast','float',('nsigma','nstns'),\
        zlib=True,least_significant_digit=3)
    p1mm_raw_forecast_out.long_name = \
        "Prob(obs > 1 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p1mm_raw_forecast_out.units = "fraction"
    p1mm_raw_forecast_out.valid_range = [0.0,1.0]
    p1mm_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)

    p5mm_raw_forecast_out = ncout.createVariable(\
        'p5mm_raw_forecast','float',('nsigma','nstns'),\
        zlib=True,least_significant_digit=3)
    p5mm_raw_forecast_out.long_name = \
        "Prob(obs > 5 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p5mm_raw_forecast_out.units = "fraction"
    p5mm_raw_forecast_out.valid_range = [0.0,1.0]
    p5mm_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)
    
    p10mm_raw_forecast_out = ncout.createVariable(\
        'p10mm_raw_forecast','float',('nsigma','nstns'),\
        zlib=True,least_significant_digit=3)
    p10mm_raw_forecast_out.long_name = \
        "Prob(obs > 10 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p10mm_raw_forecast_out.units = "fraction"
    p10mm_raw_forecast_out.valid_range = [0.0,1.0]
    p10mm_raw_forecast_out.missing_value = np.array(-99.99,dtype=float)

    precipitation_observation_out = ncout.createVariable(\
        'precipitation_observation','float',('nstns',),\
        zlib=True,least_significant_digit=3)
    precipitation_observation_out.long_name = \
        "precipitation observation accumulation in 6 hours"
    precipitation_observation_out.units = "mm"
    precipitation_observation_out.valid_range = [0.0,1000.0]
    precipitation_observation_out.missing_value = np.array(-99.99,dtype=float)

    validtimes_yyyymmddhh_out = ncout.createVariable(\
        'validtimes_yyyymmddhh','i4',('nstns',))
    validtimes_yyyymmddhh_out.long_name = \
        'valid end time of 6-h period in year/month/day/hour format'
    validtimes_yyyymmddhh_out.units = 'n/a'
    validtimes_yyyymmddhh_out.missing_value = np.array(-99,dtype=int)

    skyid_out = ncout.createVariable('skyid','i4',('nstns',))
    skyid_out.long_name = \
        "The Weather Company sky id number identifier for station"
    skyid_out.units = "n/a"
    skyid_out.valid_range = [0,100000]
    skyid_out.missing_value = np.array(-99.99,dtype=int)

    # ---- metadata

    ncout.title = 'Synthesized vector of 6-h accumulated precipitation obs\n'+\
        'and co-located precip forecast information. Obs from ../save_obs_6hourly/\n'+\
        'save_obs_6hourly_withbool_crontab.py'
    ncout.history = "Updated 14 Oct 2024: Coded by Tom Hamill, TWC"
    ncout.institution =  "The Weather Company"
    ncout.source = "[insert URL for github here]"

    # ---- initialize

    lons_out[:] = lons_stns[:]
    lats_out[:] = lats_stns[:]
    precipitation_forecast_out[:] = precipitation_forecast[:]
    precipitation_mean_out[:] = precipitation_mean_multisigma[:,:]
    pop_raw_forecast_out[:] = POP_multisigma[:,:]
    p1mm_raw_forecast_out[:] = p1mm_multisigma[:,:]
    p5mm_raw_forecast_out[:] = p5mm_multisigma[:,:]
    p10mm_raw_forecast_out[:] = p10mm_multisigma[:,:]
    precipitation_observation_out[:] = observed[:]
    validtimes_yyyymmddhh_out[:] = validtimes_yyyymmddhh[:]
    skyid_out[:] = skyids[:]
    sigmas_out[:] = sigmas[:]

    ncout.close()
    istat = 0
    return istat

# ============================================================

# --- read from command line.

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

print ('****** save_graflr_at_obs_sites_convolve_multilen.py ' + \
    cyyyymmddhh + ' ' + clead)

# --- various initializations

config_file_name = 'config_hdo.ini'
cyyyymm = cyyyymmddhh[0:6]
cyyyymmdd = cyyyymmddhh[0:8]
chh = cyyyymmddhh[8:10]
cyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
cmodel = 'ecmwf_ensmean' # use obs in ecmwf/obs file.
ilead = int(clead)
ilead_begin = ilead-5
ileads = range(ilead_begin, ilead+1, 1)
printit = False
# spatial std dev for convolution
sigmas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0 ] 
nsigma = len(sigmas)
cskip = '9' # this is used to set a boundary around the domain
nskip = int(cskip) # and don't use obs inside the boundary for convolution sake

# --- Various initialization from config file.

directory_object_name = 'DIRECTORIES'
config_file = '../ini/'+ config_file_name
print ('INFO: reading config items from ', config_file)
netcdf_6h_obs_data_directory, netcdf_directory_o_at_f_more, \
    matlab_forecast_directory, empirics_data_directory, \
    GRAFdatadir_conus_old, GRAFdatadir_conus_new =  \
    read_config_file_extract_6h_of_empirics(\
    config_file, directory_object_name)

# --- Build input observation file name, read obs file, 
#     including geohashes.

cyyyymmddhh_begin, cyyyymmddhh_end = \
    build_datestrings(printit, \
    cyyymmddhh_valid, netcdf_6h_obs_data_directory)
print ('cyyymmddhh_valid, cyyyymmddhh_begin, cyyyymmddhh_end =  ',\
    cyyymmddhh_valid,cyyyymmddhh_begin, cyyyymmddhh_end)
    
infile_obs = build_infile_obs_name(\
    netcdf_6h_obs_data_directory, cyyyymmddhh_begin,
    cyyyymmddhh_end)
istat, lons_stns, lats_stns, observed, \
    validtimes_yyyymmddhh, skyids, geohashes, nstns = \
    read_obs_data_and_geohash_them(infile_obs, cyyymmddhh_valid)
print ('nstns = ', nstns)    

# ---- Read the lon, lat, observed, valid times, skyids
#      from a previously generated f/o file for GRAF data.

if istat == 0: # GRAF/obs data exists

    # ---- Initialize GRAF output vectors
    
    graf_precip = -99.99*np.ones((nstns), dtype=float)
    graf_mean = -99.99*np.ones((nstns), dtype=float)
    prob_POP = -99.99*np.ones((nstns), dtype=float)
    prob_1mm = -99.99*np.ones((nstns), dtype=float)
    prob_5mm = -99.99*np.ones((nstns), dtype=float)
    prob_10mm = -99.99*np.ones((nstns), dtype=float)

    # ---- Read in the hourly GRAFLR CONUS forecasts, accumulate
    #      precipitation

    allthere = True
    ifirst = True
    print ('ileads = ', ileads)
    for il in ileads:
        clead = str(il)
        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]
            
        # ---- Build input GRAF directory and file name for 
        #      this lead.  Read precip and add to 6-h total.

        if int(cyyyymmddhh) > 2024040512:
            input_directory =  GRAFdatadir_conus_new
            prefix = 'grid.hdo-graf_conus.'
            cmodel_out = 'graf_conus'
        else:
            input_directory = GRAFdatadir_conus_old
            prefix = 'grid.hdo-graflr_conus.'
            cmodel_out = 'graflr_conus'
        input_directory = input_directory + \
            cyyyymmdd + '/' + chh + '/'
        input_file = prefix +cyyyymmdd_fcst+\
            'T'+chh_fcst+'0000Z.'+cyyyymmdd+'T'+chh+\
            '0000Z.PT'+clead+'H.CONUS@4km.APCP.SFC.grb2'
        infile = input_directory + input_file
        fexist1 = os.path.exists(infile)
        print (infile, fexist1)
        if fexist1 == True:
            istat, precipitation, lats, lons = \
                read_gribdata(infile, il)
            ny, nx = np.shape(lats)
            if il == ileads[0]:
                total_precipitation = np.copy(precipitation)
                latmax = np.max(lats)
                latmin = np.min(lats)
                lonmax = np.max(lons)
                lonmin = np.min(lons)
            else:
                total_precipitation = total_precipitation + \
                    precipitation
            
            # --- If this is the first time through the loop, determine the 
            #     grid point nearest to each empirics site.

            ones = np.ones((ny, nx), dtype=float)
            zeros = np.zeros((ny, nx), dtype=float)
            if ifirst == True:
                ifirst = False
                j_nearest, i_nearest = find_nearest_graf_gps_v2(\
                    lats, lons, ny, nx, nstns, nskip, lats_stns, \
                    lons_stns, latmin, latmax, lonmin, lonmax, \
                    empirics_data_directory, skyids)
                print ('len(j_nearest) = ', len(j_nearest))
        else:
            total_precipitation = -99.99
            allthere = False
            print ('data unavailable, so quitting.')
                
    # --- now extract at station locations.
    
    if allthere == True:
        
        # --- Get the mean, spread for GRAF ensembles with a 7 x 7 stencil, 
        #     of the chosen spacing, plus probabilities.
    
        POP_binary = np.where(total_precipitation >= 0.254, ones, zeros)
        p1mm_binary = np.where(total_precipitation >= 1.0, ones, zeros)
        p5mm_binary = np.where(total_precipitation >= 5.0, ones, zeros)
        p10mm_binary = np.where(total_precipitation >= 10.0, ones, zeros)
        
        for isig, sigma in enumerate(sigmas):
            print ('processing isig, sigma = ', isig, sigma)
            
            pmean_via_convolution = \
                ndimage.gaussian_filter(total_precipitation, sigma)
            POP_via_convolution = ndimage.gaussian_filter(POP_binary, sigma)
            p1mm_via_convolution = ndimage.gaussian_filter(p1mm_binary, sigma)
            p5mm_via_convolution = ndimage.gaussian_filter(p5mm_binary, sigma)
            p10mm_via_convolution = ndimage.gaussian_filter(p10mm_binary, sigma)
            
            # --- extract at grid points.
              
            graf_precipitation, graf_mean, POP, p1mm, p5mm, p10mm = \
                compute_graf_mean_probs(ny, nx, nstns, total_precipitation, \
                pmean_via_convolution, POP_via_convolution, \
                p1mm_via_convolution, p5mm_via_convolution, \
                p10mm_via_convolution, j_nearest, i_nearest)
            
            if isig == 0:
                graf_mean_multisigma = np.zeros((nsigma,nstns), dtype=float)
                POP_multisigma = np.zeros((nsigma,nstns), dtype=float)
                p1mm_multisigma = np.zeros((nsigma,nstns), dtype=float)
                p5mm_multisigma = np.zeros((nsigma,nstns), dtype=float)
                p10mm_multisigma = np.zeros((nsigma,nstns), dtype=float)                
                
            graf_mean_multisigma[isig,:] = graf_mean[:]
            POP_multisigma[isig,:] = POP[:]
            p1mm_multisigma[isig,:] = p1mm[:]
            p5mm_multisigma[isig,:] = p5mm[:]
            p10mm_multisigma[isig,:] = p10mm[:]
            
        # --- Write to netCDF file.

        output_directory = netcdf_directory_o_at_f_more + \
            cmodel_out + '/' + cyyyymm + '/'
        output_file = output_directory + cmodel_out + \
            '_6h_empirics_forecast_and_obs_multisigma_IC' + \
            cyyyymmddhh + '_lead'+clead + 'h.nc'
        modelname = 'GRAF'
        maxchar = 9
        print ('nstns, nsigma = ', nstns, nsigma)
        istat = write_fco_to_netCDF_mean_pops(output_file, \
            nstns, skyids, validtimes_yyyymmddhh, 
            graf_precipitation, graf_mean_multisigma, \
            POP_multisigma, p1mm_multisigma, p5mm_multisigma, \
            p10mm_multisigma, observed, lons_stns, lats_stns, \
            nsigma, sigmas)
           
    else:
        print ('unable to write file as data were not available')
              

