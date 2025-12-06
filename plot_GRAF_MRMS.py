"""
This plots 1-h GRAF accumulated precip and MRMS analysis

python plot_GRAF_MRMS_probs.py cyyyymmddhh clead

e.g, 

python plot_GRAF_MRMS_probs.py 2025100700 12

"""
from netCDF4 import Dataset
import numpy as np
import pygrib
from dateutils import daterange, dateshift
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib as mpl
import _pickle as cPickle
import scipy.stats as stats
from configparser import ConfigParser

# ============================================================

def read_config_file(config_file, directory_object_name):

    """ read appropriate information from the config file
        and return.
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

    return GRAFdatadir_conus_old, GRAFdatadir_conus_new

# ----------------------------------------------------------------

class GRAFDataProcessor:
    def __init__(self, config_file):
        """Initialize processor by reading configuration."""
        self.params = {}
        self.dirs = {}
        self._load_config(config_file)
        
    # -------------------------------------------------------------
        
    def _load_config(self, config_file):
        """Reads the config.ini file."""
        print(f'INFO: Loading config from {config_file}')
        config = ConfigParser()
        config.read(config_file)

        if "DIRECTORIES" not in config or "PARAMETERS" not in config:
            raise ValueError(
                "Config file missing DIRECTORIES or PARAMETERS sections")

        self.dirs = config["DIRECTORIES"]
        self.params = config["PARAMETERS"]
        
        # Parse specific needed values
        self.ndays_train = int(self.params.get("ndays_train", 60))

    # -------------------------------------------------------------

    def get_filenames(self, cyyyymmddhh, clead):
        
        """ Generates file paths based on date and 
            logic switch (April 2024)."""
        
        il = int(clead)
        # Date math
        cyyyymmdd = cyyyymmddhh[0:8]
        chh = cyyyymmddhh[8:10]
        
        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]

        # Logic for GRAF file naming convention change
        if int(cyyyymmddhh) > 2024040512:
            base_dir = self.dirs["GRAFdatadir_conus_new"]
            prefix = 'grid.hdo-graf_conus.'
        else:
            base_dir = self.dirs["GRAFdatadir_conus_old"]
            prefix = 'grid.hdo-graflr_conus.'

        input_dir = os.path.join(base_dir, cyyyymmdd, chh)
        filename = (f"{prefix}{cyyyymmdd_fcst}T{chh_fcst}0000Z."
                    f"{cyyyymmdd}T{chh}0000Z.PT{clead}H.CONUS@4km.APCP.SFC.grb2")
            
        
        full_path = os.path.join(input_dir, filename)
        print (full_path)
        return full_path, cyyyymmdd_fcst, chh_fcst

    # -------------------------------------------------------------

    def read_grib_precip(self, grib_path, end_step):
        """
        Reads precipitation from GRIB2 file.
        Returns grid info and precipitation array.
        """
        if not os.path.exists(grib_path):
            print(f'  WARNING: File not found: {grib_path}')
            return -1, None, None, None, None

        try:
            with pygrib.open(grib_path) as grb_file:
                # Select by endStep (lead time)
                grb_msgs = grb_file.select(endStep=end_step)
                if not grb_msgs:
                    print(f'  WARNING: No message found for step {end_step} ')
                    print(f'      in {grib_path}')
                    return -1, None, None, None, None
                
                grb = grb_msgs[0]
                lats, lons = grb.latlons()
                precip = grb.values
                
                # Extract projection params for potential plotting/geolocation
                proj_params = {
                    'lon_0': grb.projparams.get("lon_0", -999),
                    'lat_0': grb.projparams.get("lat_0", -999),
                    'lat_1': grb.projparams.get("lat_1", -999),
                    'lat_2': grb.projparams.get("lat_2", -999)
                }

                # --- Calculate verification time (local)  Approx: 
                #     Longitude * 12 / 180 gives hour offset roughly
                
                tz_offset = lons * 12.0 / 180.0
                
                # --- We need the valid hour from the caller, passed 
                #     via context usually returning raw offset map for 
                #     now, caller adds hour
                
            return 0, precip, lats, lons, proj_params

        except (IOError, ValueError, RuntimeError) as e:
            print(f'  ERROR reading {grib_path}: {e}')
            return -1, None, None, None, None

    # -------------------------------------------------------------

    def read_mrms(self, cyyyymmddhh):
        
        """Reads MRMS NetCDF data."""
        
        cyyyymm = cyyyymmddhh[0:6] + '/'
        filename = f'MRMS_1h_pamt_and_data_qual_{cyyyymmddhh}.nc'
        filepath = os.path.join(\
            self.dirs["mrms_data_directory"], cyyyymm, filename)
        print ('looking for ', filepath)

        if not os.path.exists(filepath):
            print(f'  WARNING: MRMS file not found: {filepath}')
            return -1, None, None

        try:
            with Dataset(filepath, 'r') as nc:
                quality = nc.variables['data_quality'][:,:]
                precip = nc.variables['precipitation'][:,:]
                
                # --- Mask bad quality data 
                
                quality = np.where(quality > 1.0, -1.0, quality)
                
            return 0, precip, quality
        except Exception as e:
            print(f'  ERROR reading MRMS {filepath}: {e}')
            return -1, None, None

# =====================================================================

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]
config_file_name = 'config_hdo.ini'

ilead = int(clead)
print ('****** plot_GRAF_MRMS.py '+ \
    cyyyymmddhh+' '+clead)
cyyyymmdd = cyyyymmddhh[0:8]
cyyyymm = cyyyymmddhh[0:6]
chh = cyyyymmddhh[8:10]
cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    
processor = GRAFDataProcessor('config_hdo.ini')
    
# --- Read GRAF

graf_file, yyyymmdd_fcst, hh_fcst = \
    processor.get_filenames(cyyyymmddhh, clead)
istat_graf, precip_graf, lats, lons, _ = \
    processor.read_grib_precip(graf_file, int(clead))
print ('  max precip_graf = ', np.max(precip_graf))
    
ny, nx = lats.shape

# --- Read MRMS

istat_mrms, precip_mrms, quality_mrms = \
    processor.read_mrms(cyyyymmddhh_valid)
    
if istat_graf == 0 and istat_mrms == 0:
    
    print ('  max precip_mrms = ', np.max(precip_mrms))   
    print ('  max precip_graf = ', np.max(precip_graf))
    ny, nx = lats.shape

    # ===========================================================
    # NOW draw the maps, raw and postprocessed.
    # ===========================================================

    latb = lats[0,0] #20.0
    late = lats[-1,-1] # 53.0
    lonb = lons[0,0] #-123.0
    lone = lons[-1,-1] # -60.0

    m = Basemap(rsphere=(6378137.00,6356752.3142),\
        resolution='l',area_thresh=1000.,projection='lcc',\
        lat_1=35.,lat_2=45,lat_0=40.,lon_0=-100., \
        llcrnrlon=lonb,llcrnrlat=latb,urcrnrlon=lone,\
        urcrnrlat=late)
    x, y = m(lons, lats)
    colorst = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange',\
        '#FCD5D9','#F6A3AE','#f17484']
    clevs_deterministic = np.array([0, 0.1, 0.25, 0.5, 1, \
        2, 3, 5, 7.5, 10,  15, 20, 25, 50]) # units mm

    suptitle = 'MRMS and GRAF 1-h precipitation forecasts, initialized '+\
            cyyyymmddhh + ', lead time +' + clead + ' h'
    fig = plt.figure(figsize=(9.,4.3))
    plt.suptitle(suptitle, fontsize=14)

    for ipanel in range(2):
        if ipanel == 0:
            title = '(a) MRMS precipitation analysis'
            data_to_plot = precip_mrms
            clevs = clevs_deterministic
            clabel = '1-h accumulated precipitation (mm)'
            axloc = 0.01,0.13,0.48,0.74
            caxloc = [0.02,0.08,0.46,0.02]
        if ipanel == 1:
            title = '(b) Raw GRAF precipitation'
            data_to_plot = precip_graf
            clevs = clevs_deterministic
            clabel = '1-h accumulated precipitation (mm)'
            axloc = 0.51,0.13,0.48,0.74
            caxloc = [0.52,0.08,0.46,0.02]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(\
            "", colorst, N=len(colorst))
        norm = colors.BoundaryNorm(boundaries=clevs, \
            ncolors=len(colorst), clip=True)
    
        # --- make plot
    
        ax = fig.add_axes(axloc)
        ax.set_title(title, fontsize=12,color='Black')
        CS = m.pcolormesh(x, y, data_to_plot, cmap=cmap, \
            shading='nearest', norm=norm)
        m.drawcoastlines(linewidth=0.8,color='Gray')
        m.drawcountries(linewidth=0.6,color='Gray')
        m.drawstates(linewidth=0.3,color='Gray')
    
        # --- add color table
    
        cax = fig.add_axes(caxloc)
        cb = plt.colorbar(CS,orientation='horizontal',cax=cax,\
             drawedges=True,ticks=clevs,format='%g', extend='max')
        cb.ax.tick_params(labelsize=6)
        cb.set_label(clabel,fontsize=7)
    
    plot_title = 'GRAF_1h_precip_and_MRMS_IC'+cyyyymmddhh+\
        '_'+clead+'h.png'
    fig.savefig(plot_title, dpi=300)
    plt.close()
    print ('saving plot to file = ',plot_title)

