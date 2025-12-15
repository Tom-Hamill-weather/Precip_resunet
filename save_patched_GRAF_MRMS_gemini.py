"""save_patched_GRAF_MRMS_gemini.py

Usage:
    python save_patched_GRAF_MRMS_gemini.py cyyyymmddhh clead

Arguments:
    cyyyymmddhh : YearMonthDayHour of the initial condition
    clead       : Forecast lead time in hours

Purpose:
    Reads GRAF forecast data and MRMS analyses.
    Extracts 96x96 patches using DYNAMIC SAMPLING:
      1. Macro: More patches on wet days, fewer on dry days.
      2. Micro: Preferentially selects patches with wet means.
      3. Overlap: Uses dense sliding window to find more wet candidates.
    Saves to pickled files for U-Net training.

Latest Update: 15 Dec 2025
"""

import os
import sys
import warnings
import _pickle as cPickle
from datetime import datetime
from configparser import ConfigParser
import numpy as np
import scipy.ndimage as ndimage
from netCDF4 import Dataset
import pygrib

# --- Note: Assuming Jeff Whitaker's dateutils.py is available 
try:
    from dateutils import dateshift, daterange
except ImportError:
    print("Error: 'dateutils' module not found. Ensure it is installed.")
    sys.exit(1)

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
        
    def _load_config(self, config_file):
        """Reads the config.ini file."""
        print(f'INFO: Loading config from {config_file}')
        config = ConfigParser()
        config.read(config_file)

        if "DIRECTORIES" not in config or "PARAMETERS" not in config:
            raise ValueError("Config file missing DIRECTORIES or PARAMETERS sections")

        self.dirs = config["DIRECTORIES"]
        self.params = config["PARAMETERS"]
        self.ndays_train = int(self.params.get("ndays_train", 60))

    def get_filenames(self, cyyyymmddhh, clead):
        """ Generates file paths based on date and logic switch (April 2024)."""
        il = int(clead)
        cyyyymmdd = cyyyymmddhh[0:8]
        chh = cyyyymmddhh[8:10]
        
        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]

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

    def read_grib_precip(self, grib_path, end_step):
        """Reads precipitation from GRIB2 file."""
        if not os.path.exists(grib_path):
            print(f'  WARNING: File not found: {grib_path}')
            return -1, None, None, None, None

        try:
            with pygrib.open(grib_path) as grb_file:
                grb_msgs = grb_file.select(endStep=end_step)
                if not grb_msgs:
                    print(f'  WARNING: No message found, step {end_step} in {grib_path}')
                    return -1, None, None, None, None
                
                grb = grb_msgs[0]
                lats, lons = grb.latlons()
                precip = grb.values
                
                # Clip excessively large precip
                precip = np.where(precip > 75., 75.0, precip)
                
                proj_params = {
                    'lon_0': grb.projparams.get("lon_0", -999),
                    'lat_0': grb.projparams.get("lat_0", -999),
                    'lat_1': grb.projparams.get("lat_1", -999),
                    'lat_2': grb.projparams.get("lat_2", -999)
                }
                
            return 0, precip, lats, lons, proj_params

        except (IOError, ValueError, RuntimeError) as e:
            print(f'  ERROR reading {grib_path}: {e}')
            return -1, None, None, None, None

    def read_mrms(self, cyyyymmddhh):
        """Reads MRMS NetCDF data."""
        cyyyymm = cyyyymmddhh[0:6] + '/'
        filename = f'MRMS_1h_pamt_and_data_qual_{cyyyymmddhh}.nc'
        filepath = os.path.join(self.dirs["mrms_data_directory"], \
            cyyyymm, filename)

        if not os.path.exists(filepath):
            print(f'  WARNING: MRMS file not found: {filepath}')
            return -1, None, None

        try:
            with Dataset(filepath, 'r') as nc:
                quality = nc.variables['data_quality'][:,:]
                precip = nc.variables['precipitation'][:,:]
                # Mask bad quality data 
                quality = np.where(quality > 1.0, -1.0, quality)
            return 0, precip, quality
        except Exception as e:
            print(f'  ERROR reading MRMS {filepath}: {e}')
            return -1, None, None

    def read_terrain(self):
        """Reads static terrain data."""
        infile = 'GRAF_CONUS_terrain_info.nc'
        if not os.path.exists(infile):
            print(f'CRITICAL: Terrain file {infile} not found. Exiting.')
            sys.exit(1)

        with Dataset(infile, 'r') as nc:
            t_diff = nc.variables['terrain_height_local_difference'][:,:]
            dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
            dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]
            
        return t_diff, dt_dlon, dt_dlat
    
    # -------------------------------------------------------------
    # MICRO-SAMPLING LOGIC (FIXED)
    # -------------------------------------------------------------

    def select_patches_vectorized(self, precip_graf, quality_mrms, \
                    ny, nx, nsamps=35):
        
        # --- 1. Identify "Interesting" Patches (RESTORED)
        # Use GAUSSIAN filter to find widespread, organized precip.
        # Sigma=30 creates a broad smoothing (approx 120km scale), ensuring
        # we target large storm systems rather than isolated single-pixel peaks.
        precip_smoothed = ndimage.gaussian_filter(precip_graf, 30)

        # --- 2. Define grid centers with HIGH OVERLAP
        # RETAINED: Stride 24 (vs 96) creates dense overlap. 
        # This is crucial to finding enough "wet" patches on days 
        # where rain is sparse but exists.
        
        stride = 24 
    
        y_indices = np.arange(ny//8 + 65, ny * 4 // 5, stride)
        x_indices = np.arange(nx//10, 9 * nx // 10, stride)
    
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
        flat_y = yy.flatten()
        flat_x = xx.flatten()

        # --- 3. Calculate Sampling Weights. We look up the
        #        SMOOTHED value at the center of the patch.
        
        candidate_val = precip_smoothed[flat_y, flat_x]
        power = 2.0 # Use power=2.0 to emphasize heavier widespread rain.
        pmax = np.max(candidate_val**power) 
        if pmax < 1e-6: pmax = 1.0 

        # Base weight 0.0001 ensures we rarely pick dry patches 
        # if any rain is available.  But there are lots of dry patches
        # at any given hour, so we pick them up not infrequently 
        # any way.
        
        weights = 0.0001 + 0.9999 * (candidate_val**power) / pmax

        # --- 4. Quality Control (RELAXED)
        
        bad_pixels = (quality_mrms <= 0.01).astype(float)
    
        # Check fraction of bad pixels (Mean filter)
        
        bad_pixel_fraction = ndimage.uniform_filter(bad_pixels, size=96)
    
        # We allow a patch if it has less than 10% bad data.
        
        patch_is_too_bad = (bad_pixel_fraction > 0.10)
    
        invalid_candidates = patch_is_too_bad[flat_y, flat_x]
        weights[invalid_candidates] = 0.0

        # --- 5. Normalize
        
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            # Fallback
            valid_mask = (~invalid_candidates).astype(float)
            valid_count = np.sum(valid_mask)
            if valid_count == 0:
                print("    WARNING: No patches satisfy MRMS quality criteria")
                return np.array([]), np.array([])
            probs = valid_mask / valid_count
        else:
            probs = weights / weight_sum
    
        # --- 6. Sample
        
        n_candidates = len(flat_y)
        k = min(nsamps, n_candidates)
        if k == 0:
            return np.array([]), np.array([])

        # Sample without replacement
        
        chosen_indices = np.random.choice(n_candidates, size=k, \
            replace=False, p=probs)
        
        return flat_y[chosen_indices], flat_x[chosen_indices]

# ----------------------------------------------------------------

def save_dataset(filename, data_dict):
    
    """Helper to save data dictionary to pickle."""
    print(f'INFO: Writing {filename}...')
    keys_order = ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF', 
        'terrain_diff', 'dt_dlon', 'dt_dlat']
    
    with open(filename, 'wb') as f:
        for key in keys_order:
            if len(data_dict[key]) > 0:
                arr = np.stack(data_dict[key], axis=0)
            else:
                arr = np.empty((0, 96, 96))
            cPickle.dump(arr, f)
    print(f'INFO: Done writing {filename}')

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: $ python save_patched_GRAF_MRMS_gemini.py cyyyymmddhh clead")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    
    print('============================================================')
    print(f'Running GRAF/MRMS Patch Save: IC {cyyyymmddhh} Lead {clead}')
    print('============================================================')

    processor = GRAFDataProcessor('config_hdo.ini')
    
    # Calculate dates
    
    iday_shift = 1 + int(clead) // 24
    ihour_shift = 24 + iday_shift * 24
    date_end1 = dateshift(cyyyymmddhh, -ihour_shift)
    date_begin1 = dateshift(date_end1, -processor.ndays_train * 24)
    date_begin2 = dateshift(cyyyymmddhh, -365*24)
    date_end2 = dateshift(cyyyymmddhh, -305*24)
    date_list = daterange(date_begin1, date_end1, 6) + \
        daterange(date_begin2, date_end2, 6)
    print(f'INFO: Processing {len(date_list)} dates for training data.')

    buckets = {
        'train': {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terdiff_x_GRAF', 'terrain_diff', 'dt_dlon', 'dt_dlat']},
        'val':   {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terdiff_x_GRAF', 'terrain_diff', 'dt_dlon', 'dt_dlat']},
        'pred':  {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', \
            'terdiff_x_GRAF', 'terrain_diff', 'dt_dlon', 'dt_dlat']}
    }
    
    try:
        terrain_diff, terr_dlon, terr_dlat = processor.read_terrain()
    except Exception as e:
        print(f"Error loading terrain: {e}")
        sys.exit(1)

    # --- MAIN LOOP ---
    
    for idate, date in enumerate(date_list):
        
        # Progress indicator
        print(f'--- Processing {idate+1}/{len(date_list)}: {date}', end=" ")
        
        cyyyymmddhh_valid = dateshift(date, int(clead))
    
        graf_file, _, _ = processor.get_filenames(date, clead)
        istat_graf, precip_graf, lats, lons, _ = \
            processor.read_grib_precip(graf_file, int(clead))
    
        if istat_graf != 0:
            print(" | GRAF Missing")
            continue 
        
        ny, nx = lats.shape
        
        # -------------------------------------------------------------
        # MACRO-SAMPLING LOGIC (Day Level). Fewer samples on days with
        #    small domain-average precipitation.
        # -------------------------------------------------------------
        
        domain_mean = np.mean(precip_graf)
        if domain_mean < 0.10:
            current_nsamps = 20
        elif domain_mean > 0.15:
            current_nsamps = 50
        else:
            current_nsamps = 35
            
        print(f"| Mean: {domain_mean:.3f} mm -> nsamps: {current_nsamps}")
        # -------------------------------------------------------------

        istat_mrms, precip_mrms, quality_mrms = \
            processor.read_mrms(cyyyymmddhh_valid)
        if istat_mrms != 0:
            print(" | MRMS Missing")
            continue 

        # Pass current_nsamps to sampler
        
        j_indices, i_indices = processor.select_patches_vectorized(
            precip_graf, quality_mrms, ny, nx, nsamps=current_nsamps
        )
    
        # Determine bucket (Train/Val/Pred)
        
        irem = idate % 10
        if irem >= 2: target_bucket = buckets['train']
        elif irem == 1: target_bucket = buckets['val']
        else: target_bucket = buckets['pred']

        r = 48
        for jy, ix in zip(j_indices, i_indices):
            y_sl = slice(jy - r, jy + r)
            x_sl = slice(ix - r, ix + r)
        
            target_bucket['GRAF'].append(\
                precip_graf[y_sl, x_sl].astype(np.float32))
            target_bucket['MRMS'].append(\
                precip_mrms[y_sl, x_sl].astype(np.float32))
            target_bucket['MRMS_qual'].append(\
                quality_mrms[y_sl, x_sl].astype(np.float32))
            target_bucket['terdiff_x_GRAF'].append(\
                terrain_diff[y_sl, x_sl] * \
                precip_graf[y_sl, x_sl].astype(np.float32))
            target_bucket['terrain_diff'].append(\
                terrain_diff[y_sl, x_sl].astype(np.float32))
            target_bucket['dt_dlon'].append(\
                terr_dlon[y_sl, x_sl].astype(np.float32))
            target_bucket['dt_dlat'].append(\
                terr_dlat[y_sl, x_sl].astype(np.float32))

        # Memory cleanup
        del precip_graf, precip_mrms, quality_mrms
        import gc; gc.collect()

    # --- Save Output ---
    
    print('============================================================')
    print('Data extraction complete. Saving files...')
    
    base_path = '../resnet_data'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    save_dataset(f'{base_path}/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick', \
        buckets['train'])
    save_dataset(f'{base_path}/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick', \
        buckets['val'])
    save_dataset(f'{base_path}/GRAF_Unet_data_predict_{cyyyymmddhh}_{clead}h.cPick', \
        buckets['pred'])

    print(f'INFO: Finished at {datetime.now().strftime("%H:%M:%S")}')

if __name__ == "__main__":
    main()

