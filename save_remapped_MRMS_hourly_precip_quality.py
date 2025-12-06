"""
python save_remapped_MRMS_hourly_precip_quality.py cyyyymmddhh_begin cyyyymmddhh_end

"""
from netCDF4 import Dataset
import numpy as np
import pygrib
from dateutils import daterange, dateshift
import _pickle as cPickle
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, interp
import matplotlib.colors as colors
import matplotlib as mpl
import _pickle as cPickle
import scipy.stats as stats
from datetime import datetime

# ----------------------------------------------------------

def read_MRMS_quality(MRMS_filename, lons_GRAF, lats_GRAF):

    """
    read MRMS data-quality sample.
    """
    import os
    import pygrib

    istat = -1
    MRMS_quality = np.empty((0,0), dtype=float)
    fexist_grib = False
    fexist_grib = os.path.exists(MRMS_filename)
    print ('   reading ',MRMS_filename, fexist_grib)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(MRMS_filename)
            grb = fcstfile.select()[0]
            lats_MRMS, lons_MRMS = grb.latlons()
            MRMS_quality = grb.values
            istat = 0
            fcstfile.close()
        except IOError:
            print ('   IOError in read_MRMS_quality reading ', \
                MRMS_filename)
            istat = -1
        except ValueError:
            print ('   ValueError in read_MRMS_quality reading ', \
                MRMS_filename)
            istat = -1
        except RuntimeError:
            print ('   RuntimeError in read_MRMS_quality reading ', \
                MRMS_filename)
            istat = -1
        
        # --- flip in order for Basemap interp to work;
        #     data and lats must be ordered with ascending lats
    
        MRMS_quality = np.flipud(MRMS_quality)
        lats_MRMS = np.flipud(lats_MRMS)
        lons_MRMS = lons_MRMS -360.
        
        # --- Apply Basemap interp to quality to get on GRAF grid.
        #     nearest sample, not interpolation.

        MRMS_quality_on_GRAF_grid = interp(MRMS_quality, \
            lons_MRMS[0,:], lats_MRMS[:,0], \
            lons_GRAF, lats_GRAF, order=0, masked=True)
    else:
        print ('grib file does not exist.')
        istat = -1

    return istat, MRMS_quality_on_GRAF_grid, lats_MRMS, lons_MRMS

# =============================================================

def read_MRMS(working_dir, cyyymmddhh_valid, lons_GRAF, lats_GRAF):
    
    # ---- Read in the hourly MRMS precipitation analyses and 
    #      interpolate to the GRAF grid
        
    infile = working_dir + 'MRMS_QPE_01h_Pass2_precip_' + \
        cyyymmddhh_valid + '.grib2'    
    fexist1 = os.path.exists(infile)
    print (infile, fexist1)
    if fexist1 == True:
        MRMSfile = pygrib.open(infile)
        grb = MRMSfile.select()[0]
        lats_MRMS, lons_MRMS = grb.latlons()
        MRMS_precip_latlon = grb.values
        istat = 0
        MRMSfile.close()
        
        # --- flip in order for Basemap interp to work;
        #     data and lats must be ordered with ascending lats
    
        MRMS_precip_latlon = np.flipud(MRMS_precip_latlon)
        lats_MRMS = np.flipud(lats_MRMS)
        lons_MRMS = lons_MRMS -360.

        # --- apply basemap interp
    
        MRMS_precip = interp(MRMS_precip_latlon, \
            lons_MRMS[0,:], lats_MRMS[:,0], \
            lons_GRAF, lats_GRAF, order=0, masked=True)
        istat_MRMS = 0
    else:
        istat_MRMS = -1
        MRMS_precip = np.empty((0,0), dtype=float)
        
    return istat_MRMS, MRMS_precip
    
# ---------------------------------------------------------------------

def save_precip_quality_to_netCDF(output_directory, \
    nlats, nlons, lons, lats,  precipitation, data_quality, \
    cyyyymmddhh):

    """ write the 1-h precipitation amt/quality to netCDF file."""

    filename = 'MRMS_1h_pamt_and_data_qual_' + cyyyymmddhh + '.nc'
    outfile = output_directory + filename
    print(f'INFO: writing MRMS data quality to {outfile}')
    ncout = Dataset(outfile,'w',format='NETCDF4_CLASSIC')

    nlats = ncout.createDimension('nlats',nlats)
    nlons = ncout.createDimension('nlons',nlons)

    lons_out = ncout.createVariable(
        'lons','f8',('nlats','nlons',),\
        zlib=True,least_significant_digit=3)
    lons_out.long_name = \
        "longitude (negative for degrees west)"
    lons_out.units = "degrees (west is negative)"
    lons_out.valid_range = [-180.0,180.0]

    lats_out = ncout.createVariable(
        'lats', 'f8', ('nlats','nlons',),\
        zlib=True, least_significant_digit=3)
    lats_out.long_name = "latitude (negative for S. Hem)"
    lats_out.units = "degrees north"
    lats_out.valid_range = [-90.0,90.0]

    data_quality_out = ncout.createVariable(
        'data_quality', 'f4', ('nlats','nlons',),
        zlib=True,least_significant_digit=2)
    data_quality_out.long_name = "MRMS 1-h data quality"
    data_quality_out.units = "n/a"
    data_quality_out.valid_range = [-1.0,1.0]
    
    precipitation_out = ncout.createVariable(
        'precipitation', 'f4', ('nlats','nlons',),
        zlib=True,least_significant_digit=2)
    precipitation_out.long_name = "MRMS 1-h precipitation amount"
    precipitation_out.units = "mm"
    precipitation_out.valid_range = [0.0,200.0]

    # ---- metadata

    ncout.title = 'MRMS 1-h precip amount and data quality from MRMS data store at '+\
        'AWS under Open Data Program'
    ncout.history = "21 Nov 2025: Coded by Tom Hamill, TWC"
    ncout.institution =  "The Weather Company"
    ncout.source = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"

    # ---- write data

    lons_out[:] = lons[:,:]
    lats_out[:] = lats[:,:]
    data_quality_out[:] = data_quality[:,:]
    precipitation_out[:] = precipitation[:,:]

    # ---- close and return

    ncout.close()
    istat = 0
    return istat


# =============================================================
# =============================================================

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]
dates = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 1)
print ('****** save_remapped_MRMS_hourly_precip_quality.py '+ \
    cyyyymmddhh_begin + ' ' + cyyyymmddhh_end)
working_dir = '/storage/home/thamill/work/'
output_dir = '/storage/home/thamill/MRMS/'

# ---- Read GRAF CONUS grid lat/lon

infile = 'GRAF_CONUS_terrain_info.nc'
nc = Dataset(infile,'r')
lats_GRAF = nc.variables['lats'][:,:]
lons_GRAF = nc.variables['lons'][:,:]
ny_GRAF, nx_GRAF = np.shape(lats_GRAF)
nc.close()

# ---- Loop through list of dates and process.

for idate, date in enumerate(dates):
    cyear = date[0:4]
    cmonth = date[4:6]
    cday = date[6:8]
    cyyyymmdd = date[0:8]
    chh = date[8:10]
    print ('------------- ', idate, date)
    
    # ---- Download the hourly MRMS precip, extract grib

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print ('downloading MRMS precip ', current_time)
    URL = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"+\
        "MultiSensor_QPE_01H_Pass2_00.00/"+ cyyyymmdd + "/" + \
        "MRMS_MultiSensor_QPE_01H_Pass2_00.00_" + cyyyymmdd + \
        '-' + chh + '0000.grib2.gz'
    outfile = working_dir + 'MRMS_QPE_01h_Pass2_precip_' + \
        date+ '.grib2.gz'
    cmd = 'wget '+URL+' -O '+outfile
    istat = os.system(cmd)
    cmd = 'gzip -d '+outfile
    istat = os.system(cmd)

    # ---- Download the MRMS data quality, extract grib file.
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print ('downloading MRMS data quality ', current_time)
    URL = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"+\
        "RadarAccumulationQualityIndex_01H_00.00/"+ cyyyymmdd + "/" + \
        "MRMS_RadarAccumulationQualityIndex_01H_00.00_" + \
        cyyyymmdd + "-000000.grib2.gz"
    file_out = working_dir + date + '_data_quality.grib.gz'
    cmd = 'wget '+URL+' -O '+file_out
    istat = os.system(cmd)
    cmd = 'gzip -d '+file_out
    istat = os.system(cmd)
    
    # ---- Read quality grib file, interpolate to GRAF grid.
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print ('reading data quality, interpolating ', current_time)
    MRMS_quality_filename = working_dir + date + '_data_quality.grib'
    istat_quality, MRMS_quality, lats_MRMS, lons_MRMS = \
        read_MRMS_quality(MRMS_quality_filename, lons_GRAF, lats_GRAF)
    print ('np.shape(MRMS_quality) = ', np.shape(MRMS_quality))

    # ---- Read MRMS hourly accumulated precip and interpolate 
    #      to GRAF grid.
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print ('reading precip amount, interpolating ', current_time)
    istat_MRMS, MRMS_precip = read_MRMS(working_dir, date, \
        lons_GRAF, lats_GRAF)
    print ('np.shape(MRMS_precip) = ', np.shape(MRMS_precip))

    # --- Save to netCDF file.
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print ('saving data ', current_time)
    
    cyyyymm = date[0:6]
    output_directory = output_dir + cyyyymm + '/'
    fexist = os.path.exists(output_directory)
    print (output_directory)
    if fexist == False:
        cmd = 'mkdir '+output_directory
        print ('cmd = ', cmd)
        istat = os.system(cmd)
        print ('mkdir istat: ',istat)
    
    istat = save_precip_quality_to_netCDF(output_directory, \
        ny_GRAF, nx_GRAF, lons_GRAF, lats_GRAF, \
        MRMS_precip, MRMS_quality, date)


