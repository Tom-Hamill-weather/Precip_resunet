"""
python save_graf_convolve_multilen.py cyyyymmddhh clead

where

cyyyymmddhh is the initial time.
clead is the lead time in hours.

"""
import numpy as np
import numpy.ma as ma
import pygrib
from dateutils import daterange, dateshift
import os, sys
from netCDF4 import Dataset, stringtochar
from datetime import datetime, timedelta, timezone
import argparse
from pathlib import Path
import scipy.stats as stats
from matplotlib import rcParams
import scipy.ndimage as ndimage

# ============================================================

def read_config_file(config_file,
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

    GRAFdatadir_conus_old = \
        directory["GRAFdatadir_conus_old"]
    GRAFdatadir_conus_new = \
        directory["GRAFdatadir_conus_new"]
    probs_directory = \
        directory["probs_directory"]

    return GRAFdatadir_conus_old, GRAFdatadir_conus_new, probs_directory

# ========================================================

def read_gribdata(gribfilename, endStep):

    """ read grib data"""

    import os
    import pygrib

    istat = -1
    precipitation = None
    lats = None
    lons = None
    fexist_grib = os.path.exists(gribfilename)
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

def write_to_netCDF_pops(outfile, ny, nx, lons, lats, \
        p0p25mm_multisigma, p1mm_multisigma, p5mm_multisigma, \
        p10mm_multisigma, sigmas):

    """
    write the coincident 1-h forecast probabilities
    to a netCDF file.
    """
    from netCDF4 import Dataset
    import numpy as np

    nsigma = len(sigmas)
    print ('writing to ', outfile)
    ncout = Dataset(outfile,'w')

    ncout.createDimension('ny', ny)
    ncout.createDimension('nx', nx)
    ncout.createDimension('nsigma', nsigma)

    sigmas_out = ncout.createVariable('sigmas','float',('nsigma',),\
        zlib=True,least_significant_digit=1)
    sigmas_out.long_name = "convolution spatial std dev in grid pts"
    sigmas_out.units = "grid pts"
    sigmas_out.valid_range = [0.0, 200.0]
    sigmas_out.missing_value = np.array(-99.99,dtype=float)

    lons_out = ncout.createVariable('lons','float',('ny','nx'),\
        zlib=True,least_significant_digit=3)
    lons_out.long_name = "longitude (negative for degrees west)"
    lons_out.units = "degrees_east"
    lons_out.valid_range = [-180.0,180.0]
    lons_out.missing_value = np.array(-99.99,dtype=float)

    lats_out = ncout.createVariable('lats','float',('ny','nx'),\
        zlib=True,least_significant_digit=3)
    lats_out.long_name = "latitude (negative for S. Hem)"
    lats_out.units = "degrees north"
    lats_out.valid_range = [-90.0,90.0]
    lats_out.missing_value = np.array(-99.99,dtype=float)

    p0p25mm_raw_forecast_out = ncout.createVariable(\
        'p0p25mm_raw_forecast','float',('nsigma','ny','nx'),\
        zlib=True,least_significant_digit=2)
    p0p25mm_raw_forecast_out.long_name = \
        "0.25 mm prob forecast from raw convolved ens " +\
        "relative frequency, 0.25-mm threshold"
    p0p25mm_raw_forecast_out.units = "fraction"
    p0p25mm_raw_forecast_out.valid_range = [0.0,1.0]
    p0p25mm_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)

    p1mm_raw_forecast_out = ncout.createVariable(\
        'p1mm_raw_forecast','float',('nsigma','ny','nx'),\
        zlib=True,least_significant_digit=2)
    p1mm_raw_forecast_out.long_name = \
        "Prob(obs > 1 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p1mm_raw_forecast_out.units = "fraction"
    p1mm_raw_forecast_out.valid_range = [0.0,1.0]
    p1mm_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)

    p5mm_raw_forecast_out = ncout.createVariable(\
        'p5mm_raw_forecast','float',('nsigma','ny','nx'),\
        zlib=True,least_significant_digit=2)
    p5mm_raw_forecast_out.long_name = \
        "Prob(obs > 5 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p5mm_raw_forecast_out.units = "fraction"
    p5mm_raw_forecast_out.valid_range = [0.0,1.0]
    p5mm_raw_forecast_out.missing_value = \
        np.array(-99.99,dtype=float)

    p10mm_raw_forecast_out = ncout.createVariable(\
        'p10mm_raw_forecast','float',('nsigma','ny','nx'),\
        zlib=True,least_significant_digit=2)
    p10mm_raw_forecast_out.long_name = \
        "Prob(obs > 10 mm) forecast from raw "+\
        "convolved ens. relative frequency"
    p10mm_raw_forecast_out.units = "fraction"
    p10mm_raw_forecast_out.valid_range = [0.0,1.0]
    p10mm_raw_forecast_out.missing_value = np.array(-99.99,dtype=float)

    # ---- metadata

    ncout.title = 'Synthesized vector of 1-h accumulated precipitation prob forecasts.'
    ncout.history = "Updated 21 April 2026: Coded by Tom Hamill, TWC"
    ncout.institution =  "The Weather Company"
    ncout.source = "[insert URL for github here]"

    # ---- write data

    lons_out[:] = lons[:,:]
    lats_out[:] = lats[:,:]
    p0p25mm_raw_forecast_out[:] = p0p25mm_multisigma[:,:,:]
    p1mm_raw_forecast_out[:] = p1mm_multisigma[:,:,:]
    p5mm_raw_forecast_out[:] = p5mm_multisigma[:,:,:]
    p10mm_raw_forecast_out[:] = p10mm_multisigma[:,:,:]
    sigmas_out[:] = sigmas[:]

    ncout.close()
    istat = 0
    return istat

# ============================================================

# --- read from command line.

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

print ('****** save_graf_convolve_multilen.py ' + \
    cyyyymmddhh + ' ' + clead)

# --- various initializations

config_file_name = 'config_hdo.ini'
cyyyymm = cyyyymmddhh[0:6]
cyyyymmdd = cyyyymmddhh[0:8]
chh = cyyyymmddhh[8:10]
cyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
printit = False
# spatial std dev for convolution
sigmas = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0 ]
nsigma = len(sigmas)

# --- Various initialization from config file.

directory_object_name = 'DIRECTORIES'
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file_name)
print ('INFO: reading config items from ', config_file)
GRAFdatadir_conus_old, GRAFdatadir_conus_new, probs_directory = \
    read_config_file(config_file, directory_object_name)

# ---- Build input GRAF directory and file name for this lead.

cyyyymmddhh_fcst = dateshift(cyyyymmddhh, int(clead))
cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
chh_fcst = cyyyymmddhh_fcst[8:10]

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
input_file = prefix + cyyyymmdd_fcst + \
    'T' + chh_fcst + '0000Z.' + cyyyymmdd + 'T' + chh + \
    '0000Z.PT' + clead + 'H.CONUS@4km.APCP.SFC.grb2'
infile = input_directory + input_file
fexist1 = os.path.exists(infile)
print (infile, fexist1)

if fexist1 == True:
    istat, precipitation, lats, lons = \
        read_gribdata(infile, int(clead))
    ny, nx = np.shape(lats)

    # ---- Initialize GRAF output arrays

    p0p25mm_via_convolution = -99.99*np.ones((nsigma,ny,nx), dtype=float)
    p1mm_via_convolution = -99.99*np.ones((nsigma,ny,nx), dtype=float)
    p5mm_via_convolution = -99.99*np.ones((nsigma,ny,nx), dtype=float)
    p10mm_via_convolution = -99.99*np.ones((nsigma,ny,nx), dtype=float)
    ones = np.ones((ny,nx), dtype=float)
    zeros = np.zeros((ny,nx), dtype=float)

    # --- Compute binary fields then convolve with Gaussian of each sigma.

    p0p25mm_binary = np.where(precipitation >= 0.25, ones, zeros)
    p1mm_binary = np.where(precipitation >= 1.0, ones, zeros)
    p5mm_binary = np.where(precipitation >= 5.0, ones, zeros)
    p10mm_binary = np.where(precipitation >= 10.0, ones, zeros)

    for isig, sigma in enumerate(sigmas):
        print ('processing isig, sigma = ', isig, sigma)

        p0p25mm_via_convolution[isig,:,:] = \
            ndimage.gaussian_filter(p0p25mm_binary, sigma)
        p1mm_via_convolution[isig,:,:] = \
            ndimage.gaussian_filter(p1mm_binary, sigma)
        p5mm_via_convolution[isig,:,:] = \
            ndimage.gaussian_filter(p5mm_binary, sigma)
        p10mm_via_convolution[isig,:,:] = \
            ndimage.gaussian_filter(p10mm_binary, sigma)

    # --- Write to netCDF file.

    output_directory = probs_directory + \
        cmodel_out + '/' + cyyyymm + '/'
    os.makedirs(output_directory, exist_ok=True)
    output_file = output_directory + cmodel_out + \
        '_1h_probs_multisigma_IC' + \
        cyyyymmddhh + '_lead' + clead + 'h.nc'
    print ('max p5mm_via_convolution[0,:,:] = ', \
        np.max(p5mm_via_convolution[0,:,:]))
    istat = write_to_netCDF_pops(output_file, ny, nx, lons, lats, \
            p0p25mm_via_convolution, p1mm_via_convolution, \
            p5mm_via_convolution, p10mm_via_convolution, sigmas)
else:
    print ('unable to write file as data were not available')
