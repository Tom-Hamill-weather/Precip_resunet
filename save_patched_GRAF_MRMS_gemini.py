"""
save_patched_GRAF_MRMS_gemini.py

Usage:
    python save_patched_GRAF_MRMS_gemini.py cyyyymmddhh clead

Arguments:
    cyyyymmddhh : YearMonthDayHour of the initial condition (e.g., 2023040112)
        This will then load 60 days of data before this date, and data 
        10-12 months prior to this date.
    clead       : Forecast lead time in hours (e.g., 12)

Purpose:
    Reads GRAF forecast data, terrain elevation, and MRMS analyses.
    Extracts 64x64 patches based on precipitation intensity 
        (weighted sampling).
    Splits data into Training (80%), Validation (10%), 
        and Prediction (10%) sets.
    Saves the resulting datasets to pickled files.

Coded by: Tom Hamill, with Google Gemini to speed up (5 Dec 2025)
"""

import os
import sys
import warnings
import _pickle as cPickle
from datetime import datetime
from configparser import ConfigParser
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib as mpl
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import pygrib
import scipy.ndimage as ndimage
from netCDF4 import Dataset
import scipy.stats as stats
import scipy.ndimage as ndimage

# --- Note: Assuming Jeff Whitaker's dateutils.py is available 
#     in the user's environment as per original script
try:
    from dateutils import dateshift, daterange
except ImportError:
    print("Error: 'dateutils' module not found.  Ensure it is installed.")
    sys.exit(1)

# --- Matplotlib imports (conditionally used, but kept 
#     for the plot_map function)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    pass 

# --- Configuration

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

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
            raise ValueError(\
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
        #print ('looking for ', filepath)

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

    # -------------------------------------------------------------

    def read_terrain(self):
        
        """Reads static terrain data."""
        
        infile = 'GRAF_CONUS_terrain_info.nc'
        if not os.path.exists(infile):
            print(f'CRITICAL: Terrain file {infile} not found. Exiting.')
            sys.exit(1)

        with Dataset(infile, 'r') as nc:
            t_height = nc.variables['terrain_height'][:,:]
                # raw elevation
            t_diff = nc.variables['terrain_height_local_difference'][:,:]
                # raw elevation minus smoothed elevation to bring out
                # local differences.
            dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
                # slope of terrain elevation change (smoothed)
            dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]
            
        return t_height, t_diff, dt_dlon, dt_dlat

    # -------------------------------------------------------------
    
    def select_patches_vectorized(self, precip_graf, quality_mrms, \
                ny, nx, nsamps=50):
        
        """
        Vectorized version of block selection with Quality Control; 
        only consider patch locations where the minimum MRMS quality
        is 0.01 or greater.
        """
    
        # --- 1. Gaussian smooth (Heavy operation, do once).  We
        #        select patches based on smoothed precip values
        #        so we're not selecting to favor, say, isolated
        #        convection.
    
        precip_smoothed = ndimage.gaussian_filter(precip_graf, 30)
    
        # --- 2. Define potential patch grid centers using numpy ranges
    
        y_indices = np.arange(ny//8 + 65, ny * 4 // 5, 64)
        x_indices = np.arange(nx//10, 9 * nx // 10, 64)
    
        # --- Create a meshgrid of indices
    
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
    
        # --- Flatten for processing
    
        flat_y = yy.flatten()
        flat_x = xx.flatten()
    
        # --- 3. Extract Smoothed Precip Values
    
        candidate_precips = precip_smoothed[flat_y, flat_x]
        pmax = np.max(candidate_precips**0.4)
        if pmax == 0: pmax = 1.0 

        # --- 4. Calculate initial weights based on power transformation
        #        of precip.  The 0.05 weight allows some chance of 
        #        selecting a totally dry patch.  The 
        #        0.95 * (candidate_precips**0.4) / pmax is designed
        #        to prefer heavy precipitation, but with the power
        #        transformation to not weight the heaviest cases as
        #        much as it would with 1.0 power.
    
        weights = 0.05 + 0.95 * (candidate_precips**0.4) / pmax

        # ============================================================
        # NEW: Quality Control Filter
        # ============================================================
    
        # A. Create a boolean mask where quality is insufficient
        #    (True = Bad, False = Good)
        bad_pixels = (quality_mrms <= 0.01)

        # B. Use maximum_filter to check 64x64 neighborhoods.
        #    If a 64x64 window contains ANY bad pixel, the max() 
        #    of that window will be True.
        patch_has_bad_data = ndimage.maximum_filter(bad_pixels, size=64)

        # C. Check our specific candidate centers against this map
        invalid_candidates = patch_has_bad_data[flat_y, flat_x]

        # D. Zero out weights for any candidate that touches bad data
        weights[invalid_candidates] = 0.0

        # ============================================================

        # --- 5. Normalize weights to probabilities
    
        weight_sum = np.sum(weights)
    
        if weight_sum == 0:
            # Fallback: If all precip-weighted patches were invalid 
            # (or precip was 0), try to select ANY valid patch uniformly.
            valid_mask = (~invalid_candidates).astype(float)
            valid_count = np.sum(valid_mask)
        
            if valid_count == 0:
                print("    WARNING: No patches satisfy MRMS quality > 0.1")
                return np.array([]), np.array([])
        
            probs = valid_mask / valid_count
        else:
            probs = weights / weight_sum
        
        # --- 6. Sample
    
        n_candidates = len(flat_y)
    
        # Ensure we don't ask for more samples than available candidates
        # (considering we might have filtered many out)
        
        valid_indices_indices = np.where(probs > 0)[0]
        n_valid = len(valid_indices_indices)
    
        k = min(nsamps, n_valid)
    
        if k == 0:
            return np.array([]), np.array([])

        chosen_indices = np.random.choice(n_candidates, \
            size=k, replace=False, p=probs)
    
        return flat_y[chosen_indices], flat_x[chosen_indices]

# ----------------------------------------------------------------

def save_dataset(filename, data_dict):
    
    """Helper to save data dictionary to pickle."""
    print(f'INFO: Writing {filename}...')
        
    # Convert lists to numpy arrays if they aren't already
    # Note: We assume data_dict contains lists of 64x64 patches
    
    keys_order = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain', 
        'terrain_diff', 'dt_dlon', 'dt_dlat', 'time']
    
    with open(filename, 'wb') as f:
        for key in keys_order:
            # Stack lists into a (N, 64, 64) array
            if len(data_dict[key]) > 0:
                arr = np.stack(data_dict[key], axis=0)
            else:
                arr = np.empty((0, 64, 64))
            cPickle.dump(arr, f)
            
    print(f'INFO: Done writing {filename}')

# ---------------------------------------------------------------- 

def plot_map(lons, lats, graf_precip, mrms_precip, mrms_qual, 
             proj_params, cyyyymmddhh, clead):
    """
    Legacy plotting function. 
    Only works if Basemap is installed.
    """
    if 'Basemap' not in globals():
        print("Basemap not installed, skipping plot.")
        return

    latb, late = lats[0, 0], lats[-1, -1]
    lonb, lone = lons[0, 0], lons[-1, -1]

    m = Basemap(
        rsphere=(6378137.00, 6356752.3142),
        resolution='l', area_thresh=1000., projection='lcc',
        lat_1=proj_params['lat_1'], lat_2=proj_params['lat_2'],
        lat_0=proj_params['lat_0'], lon_0=proj_params['lon_0'],
        llcrnrlon=lonb, llcrnrlat=latb, urcrnrlon=lone, urcrnrlat=late
    )
    x, y = m(lons, lats)
    
    colorst = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange',\
        '#FCD5D9','#F6A3AE','#f17484'] 
    clevs_deterministic = np.array([0, 0.01, 0.25, 0.66, 1, \
        2, 3, 5, 7.5, 10,  15, 20, 25, 200]) # units mm
    fig = plt.figure(figsize=(9.,5.3))
    suptitle = '1-h accumulated precipitation, initialized '+\
        cyyyymmddhh + ', lead time +' + clead + ' h'
    plt.suptitle(suptitle, fontsize=14)
        
    # ---- panel 1: GRAF precip
    
    axloc = 0.01,0.11,0.48,0.78
    caxloc = [0.01,0.105,0.48,0.02]
    title = '(a) Raw GRAF precipitation'
    clevs = clevs_deterministic
    clabel = '1-h accumulated precipitation (mm)'
    cmap = mpl.colors.LinearSegmentedColormap.from_list(\
        "", colorst, N=len(colorst))
    norm = colors.BoundaryNorm(boundaries=clevs, \
        ncolors=len(colorst), clip=True)

    # --- make plot

    ax = fig.add_axes(axloc)
    ax.set_title(title, fontsize=10,color='Black')
    CS = m.pcolormesh(x, y, GRAF_precipitation, cmap=cmap, \
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

    # --- Panel 2. MRMS.  Add grey where quality < 0.5

    axloc = 0.51,0.11,0.48,0.78
    caxloc = [0.51,0.105,0.48,0.02]
    title = '(b) MRMS precipitation'
    ax = fig.add_axes(axloc)
    ax.set_title(title, fontsize=10,color='Black')
    CS = m.pcolormesh(x, y, MRMS_precip, cmap=cmap, \
        shading='nearest', norm=norm)
    CS3 = m.contourf(x, y, MRMS_quality, levels=[-1.0, 0.5], \
        cmap=None, colors=['LightGray','White'], alpha=0.5)
    m.drawcoastlines(linewidth=0.8,color='Gray')
    m.drawcountries(linewidth=0.6,color='Gray')
    m.drawstates(linewidth=0.3,color='Gray')

    # --- add color table

    cax = fig.add_axes(caxloc)
    cb = plt.colorbar(CS,orientation='horizontal',cax=cax,\
         drawedges=True,ticks=clevs,format='%g', extend='max')
    cb.ax.tick_params(labelsize=6)
    cb.set_label(clabel,fontsize=7)

    plot_title = 'GRAF_MRMS_1h_precip_IC'+cyyyymmddhh+'_'+clead+'h.png'
    fig.savefig(plot_title, dpi=300)
    plt.close()
    print ('saving plot to file = ',plot_title)

    istat = 0
    return istat


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py cyyyymmddhh clead")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    
    print('============================================================')
    print(f'Running GRAF/MRMS Patch Save: IC {cyyyymmddhh} Lead {clead}')
    print(f'Start Time: {datetime.now().strftime("%H:%M:%S")}')
    print('============================================================')

    # --- 1. Setup Processor
    
    processor = GRAFDataProcessor('config_hdo.ini')
    
    # --- 2. Calculate the initial and end dates for which to 
    #        extract forecasts and MRMS analyses.
    
    iday_shift = 1 + int(clead) // 24
    ihour_shift = 24 + iday_shift * 24
    date_end1 = dateshift(cyyyymmddhh, -ihour_shift)
    date_begin1 = dateshift(date_end1, -processor.ndays_train * 24)
    date_begin2 = dateshift(cyyyymmddhh, -365*24)
    date_end2 = dateshift(cyyyymmddhh, -305*24)
    
    # --- 3. Generate list of initial condition dates for the last 60
    #        days and 10-12 months ago.  In this way we have a four
    #        month sample, seasonally centered on the date of interest.
    
    date_list1 = daterange(date_begin1, date_end1, 12)
    date_list2 = daterange(date_begin2, date_end2, 12)
    date_list = date_list1 + date_list2
    #print (date_list)
    
    print(f'INFO: Processing {len(date_list)} dates for training data.')

    # --- 4. Data Containers (Buckets); Instead of 
    #        appending to numpy arrays (slow), we append
    #        to lists (fast) and stack them at the very end.
    
    buckets = {
        'train': {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terrain', 'terrain_diff', 'dt_dlon', 'dt_dlat', 'time']},
        'val':   {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terrain', 'terrain_diff', 'dt_dlon', 'dt_dlat', 'time']},
        'pred':  {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terrain', 'terrain_diff', 'dt_dlon', 'dt_dlat', 'time']}
    }
    
    # --- 5. Load static terrain once
    
    try:
        terr_h, terr_diff, terr_dlon, terr_dlat = \
            processor.read_terrain()
        has_terrain = True
    except Exception as e:
        print(f"Error loading terrain: {e}")
        sys.exit(1)

    # --- 6. Main loop over dates
    
    rlist = []
    sample_ktr = 0
    for idate, date in enumerate(date_list):
        print(f'--- Processing {idate+1}/{len(date_list)}: {date}')
        
        # --- Calculate valid time for this specific date
        
        cyyyymmddhh_valid = dateshift(date, int(clead))
        
        # --- Read GRAF
        
        graf_file, yyyymmdd_fcst, hh_fcst = \
            processor.get_filenames(date, clead)
        istat_graf, precip_graf, lats, lons, _ = \
            processor.read_grib_precip(graf_file, int(clead))
        print ('  max precip_graf = ', np.max(precip_graf))
        
        if istat_graf != 0:
            continue # Skip this date if GRAF missing
            
        ny, nx = lats.shape
        
        # --- Read MRMS
        
        istat_mrms, precip_mrms, quality_mrms = \
            processor.read_mrms(cyyyymmddhh_valid)
        print ('  max precip_mrms = ', np.max(precip_mrms))
        rs, p = stats.spearmanr(precip_graf.flatten(), \
            precip_mrms.flatten())
        print ('mean, spearman correlation = ', \
            np.mean(precip_graf), rs)
        #sys.exit()
        
        if istat_mrms != 0:
            continue # Skip if MRMS missing

        # --- Calculate local lime (approximate) and
        #     create a time array matching grid size
        
        tz_off = lons * 12.0 / 180.0
        verif_local_time = int(hh_fcst) + tz_off

        # --- Select patches
        #     
        j_indices, i_indices = \
            processor.select_patches_vectorized(\
                precip_graf, quality_mrms, ny, nx)
        
        # --- Determine bucket to append the data (Train/Val/Pred)
        #     
        irem = idate % 10
        if irem >= 2:
            target_bucket = buckets['train']
        elif irem == 1:
            target_bucket = buckets['val']
        else:
            target_bucket = buckets['pred']

        # --- Extract and store patches; Define window radius 
        #     (32 for a 64x64 patch)
        
        r = 32
        for jy, ix in zip(j_indices, i_indices):
            
            # --- Slice indices
            
            y_sl = slice(jy - r, jy + r)
            x_sl = slice(ix - r, ix + r)
            
            # --- Store data into list
            
            target_bucket['GRAF'].append(precip_graf[y_sl, x_sl])
            target_bucket['MRMS'].append(precip_mrms[y_sl, x_sl])
            target_bucket['MRMS_qual'].append(quality_mrms[y_sl, x_sl])
            target_bucket['terrain'].append(terr_h[y_sl, x_sl])
            target_bucket['terrain_diff'].append(terr_diff[y_sl, x_sl])
            target_bucket['dt_dlon'].append(terr_dlon[y_sl, x_sl])
            target_bucket['dt_dlat'].append(terr_dlat[y_sl, x_sl])
            target_bucket['time'].append(verif_local_time[y_sl, x_sl])
            
            fcst = precip_graf[y_sl, x_sl]+\
                np.random.normal(loc=0.0,scale=0.001,size=(64,64))
            obs = precip_mrms[y_sl, x_sl]+\
                np.random.normal(loc=0.0,scale=0.001,size=(64,64))
            rs, p = stats.spearmanr(fcst.flatten(), obs.flatten())
            if np.mean(fcst) > 1.0:
                print ('date, jy, ix, mean, spearman correlation = ', \
                    date, jy, ix, np.mean(fcst), rs)
                rlist.append(rs)
            sample_ktr = sample_ktr + 1

    # --- 7. Save Output
    
    rlist = np.array(rlist)
    print ('mean correlation for mean precip > 1 mm = ', np.mean(rlist))
    print ('number of samples: ', sample_ktr)
    print ('buckets dictionary size = ', sys.getsizeof(buckets))
    
    print('============================================================')
    print('Data extraction complete. Saving files...')
    
    base_path = '../resnet_data'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # --- Save Train
    
    save_dataset(
        f'{base_path}/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick', 
        buckets['train']
    )
    
    # --- Save Test (Val in logic)
    
    save_dataset(
        f'{base_path}/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick', 
        buckets['val']
    )
    
    # --- Save Predict
    
    save_dataset(
        f'{base_path}/GRAF_Unet_data_predict_{cyyyymmddhh}_{clead}h.cPick', 
        buckets['pred']
    )

    print(f'INFO: Finished at {datetime.now().strftime("%H:%M:%S")}')


if __name__ == "__main__":
    main()