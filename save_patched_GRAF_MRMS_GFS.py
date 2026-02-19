"""save_patched_GRAF_MRMS_GFS.py

Usage:
    python save_patched_GRAF_MRMS_GFS.py cyyyymmddhh clead

Arguments:
    cyyyymmddhh : YearMonthDayHour of the initial condition
    clead       : Forecast lead time in hours

Purpose:
    Reads GRAF forecast data, MRMS analyses, and GFS forecast data.
    Extracts 96x96 patches using DYNAMIC SAMPLING:
      1. Macro: More patches on wet days, fewer on dry days.
      2. Micro: Preferentially selects patches with wet means.
      3. Overlap: Uses dense sliding window to find more wet candidates.
    Now includes GFS features: PWAT, column-average relative humidity (r), and CAPE
    These are spatially and temporally interpolated to match GRAF patches.
    Saves to pickled files for U-Net training.

Latest Update: Modified from save_patched_GRAF_MRMS_gemini.py to include GFS data
"""

import os
import sys
import warnings
import _pickle as cPickle
from datetime import datetime
from configparser import ConfigParser
import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
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
        self.graf_transition_date = self.params.get("GRAF_transition_date", "2024040512")

    def get_filenames(self, cyyyymmddhh, clead):
        """ Generates file paths based on date and logic switch (April 2024)."""
        il = int(clead)
        cyyyymmdd = cyyyymmddhh[0:8]
        chh = cyyyymmddhh[8:10]

        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]

        if int(cyyyymmddhh) > int(self.graf_transition_date):
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
        filepath = os.path.join(self.dirs["mrms_data_directory"], cyyyymm, filename)

        if not os.path.exists(filepath):
            print(f'  WARNING: MRMS file not found: {filepath}')
            return -1, None, None

        try:
            with Dataset(filepath, 'r') as nc:
                quality = nc.variables['data_quality'][:,:]
                precip = nc.variables['precipitation'][:,:]
                quality = np.where(quality > 1.0, -1.0, quality)
            return 0, precip, quality
        except Exception as e:
            print(f'  ERROR reading MRMS {filepath}: {e}')
            return -1, None, None

    def read_gfs(self, cyyyymmddhh, clead):
        """
        Reads GFS NetCDF data and extracts PWAT, r (relative humidity), and CAPE.

        Returns:
            istat: 0 if successful, -1 if failed
            gfs_data: dict with keys 'pwat', 'r', 'cape', 'lats', 'lons', 'step'
        """
        gfs_dir = self.dirs.get("gfs_data_directory", "/storage1/home/thamill/resnet/resnet_data/gfs")
        filename = f'gfs_subset_{cyyyymmddhh}.nc'
        filepath = os.path.join(gfs_dir, filename)

        if not os.path.exists(filepath):
            print(f'  WARNING: GFS file not found: {filepath}')
            return -1, None

        try:
            with Dataset(filepath, 'r') as nc:
                # Read coordinate variables
                lats = nc.variables['latitude'][:]  # 1D array, decreasing
                lons = nc.variables['longitude'][:] # 1D array, 0-360
                steps = nc.variables['step'][:]     # Forecast hours

                # Find the closest step to requested lead time
                ilead = int(clead)
                step_diffs = np.abs(steps - ilead)
                step_idx = np.argmin(step_diffs)

                if step_diffs[step_idx] > 0:
                    print(f'  INFO: GFS exact lead {ilead}h not found. Using step {steps[step_idx]}h')

                # Read the three variables at the selected step
                pwat = nc.variables['pwat'][step_idx, :, :]   # (latitude, longitude)
                r = nc.variables['r'][step_idx, :, :]         # (latitude, longitude)

                # CAPE may have an extra pressureFromGroundLayer dimension in some files.
                # Collapse it by taking the max (most unstable CAPE) regardless of axis position.
                cape_var = nc.variables['cape']
                cape_raw = cape_var[step_idx, :]
                if 'pressureFromGroundLayer' in cape_var.dimensions:
                    level_axis = list(cape_var.dimensions[1:]).index('pressureFromGroundLayer')
                    cape = np.max(cape_raw, axis=level_axis)
                else:
                    cape = cape_raw

                # Handle NaN values
                pwat = np.where(np.isnan(pwat), 0.0, pwat)
                r = np.where(np.isnan(r), 0.0, r)
                cape = np.where(np.isnan(cape), 0.0, cape)

                gfs_data = {
                    'pwat': pwat,
                    'r': r,
                    'cape': cape,
                    'lats': lats,
                    'lons': lons,
                    'step': steps[step_idx]
                }

            return 0, gfs_data

        except Exception as e:
            print(f'  ERROR reading GFS {filepath}: {e}')
            return -1, None

    def interpolate_gfs_to_patches(self, gfs_data, graf_lats, graf_lons, j_indices, i_indices):
        """
        Interpolates GFS data (on lat/lon grid) to GRAF patch locations.

        Args:
            gfs_data: dict with 'pwat', 'r', 'cape', 'lats', 'lons'
            graf_lats: 2D array of GRAF latitudes (ny, nx)
            graf_lons: 2D array of GRAF longitudes (ny, nx)
            j_indices: array of patch center j-indices
            i_indices: array of patch center i-indices

        Returns:
            List of dicts, each with 'pwat', 'r', 'cape' as 96x96 arrays
        """
        # GFS grid info
        gfs_lats = gfs_data['lats']  # 1D, decreasing (90 to -90)
        gfs_lons = gfs_data['lons']  # 1D, 0 to 360

        # Create interpolators for each variable (using bilinear interpolation)
        # Note: RegularGridInterpolator expects ascending coordinates
        # GFS lats are descending, so we need to flip
        gfs_lats_asc = gfs_lats[::-1]

        interp_pwat = RegularGridInterpolator(
            (gfs_lats_asc, gfs_lons),
            gfs_data['pwat'][::-1, :],  # Flip to match ascending lats
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        interp_r = RegularGridInterpolator(
            (gfs_lats_asc, gfs_lons),
            gfs_data['r'][::-1, :],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        interp_cape = RegularGridInterpolator(
            (gfs_lats_asc, gfs_lons),
            gfs_data['cape'][::-1, :],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Extract patches
        patches = []
        r = 48  # Half-width of 96x96 patch

        for jy, ix in zip(j_indices, i_indices):
            y_sl, x_sl = slice(jy - r, jy + r), slice(ix - r, ix + r)

            # Get lat/lon for this patch
            patch_lats = graf_lats[y_sl, x_sl]
            patch_lons = graf_lons[y_sl, x_sl]

            # Convert GRAF lons from -180:180 to 0:360 for GFS
            patch_lons_360 = np.where(patch_lons < 0, patch_lons + 360, patch_lons)

            # Create points for interpolation (flatten)
            points = np.column_stack([patch_lats.ravel(), patch_lons_360.ravel()])

            # Interpolate each variable
            pwat_patch = interp_pwat(points).reshape(96, 96).astype(np.float32)
            r_patch = interp_r(points).reshape(96, 96).astype(np.float32)
            cape_patch = interp_cape(points).reshape(96, 96).astype(np.float32)

            patches.append({
                'pwat': pwat_patch,
                'r': r_patch,
                'cape': cape_patch
            })

        return patches

    def read_terrain(self):
        """Reads static terrain data."""
        infile = self.dirs.get("terrain_file", "GRAF_CONUS_terrain_info.nc")
        if not os.path.exists(infile):
            print(f'CRITICAL: Terrain file {infile} not found. Exiting.')
            sys.exit(1)

        with Dataset(infile, 'r') as nc:
            t_diff = nc.variables['terrain_height_local_difference'][:,:]
            dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
            dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]

        return t_diff, dt_dlon, dt_dlat

    def select_patches_vectorized(self, precip_graf, quality_mrms, ny, nx, nsamps=35):
        """MICRO-SAMPLING LOGIC: Identifies interesting patches via Gaussian filtering."""
        precip_smoothed = ndimage.gaussian_filter(precip_graf, 30)
        stride = 24
        y_indices = np.arange(ny//8 + 65, ny * 4 // 5, stride)
        x_indices = np.arange(nx//10, 9 * nx // 10, stride)
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
        flat_y, flat_x = yy.flatten(), xx.flatten()

        candidate_val = precip_smoothed[flat_y, flat_x]
        pmax = np.max(candidate_val**2.0)
        if pmax < 1e-6: pmax = 1.0
        weights = 0.0001 + 0.9999 * (candidate_val**2.0) / pmax

        bad_pixels = (quality_mrms <= 0.01).astype(float)
        bad_pixel_fraction = ndimage.uniform_filter(bad_pixels, size=96)
        patch_is_too_bad = (bad_pixel_fraction > 0.10)
        weights[patch_is_too_bad[flat_y, flat_x]] = 0.0

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.array([]), np.array([])

        probs = weights / weight_sum
        chosen_indices = np.random.choice(len(flat_y), size=min(nsamps, len(flat_y)), replace=False, p=probs)
        return flat_y[chosen_indices], flat_x[chosen_indices]

# ----------------------------------------------------------------

def save_dataset(filename, data_dict):
    """Helper to save data dictionary to pickle, including time stamps and GFS data."""
    print(f'INFO: Writing {filename}...')
    # Original 7 keys + 2 time keys + 3 GFS keys = 12 keys total
    keys_order = ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF',
                  'terrain_diff', 'dt_dlon', 'dt_dlat',
                  'init_times', 'valid_times',
                  'GFS_pwat', 'GFS_r', 'GFS_cape']

    with open(filename, 'wb') as f:
        for key in keys_order:
            if key in ['init_times', 'valid_times']:
                # Save lists of strings directly
                cPickle.dump(data_dict[key], f)
            else:
                if len(data_dict[key]) > 0:
                    arr = np.stack(data_dict[key], axis=0)
                else:
                    arr = np.empty((0, 96, 96))
                cPickle.dump(arr, f)
    print(f'INFO: Done writing {filename}')

# ----------------------------------------------------------------

def detect_config():
    """Select the appropriate config file based on the runtime environment."""
    if os.path.exists('/data2/resnet_data'):
        return 'config_aws.ini'
    elif os.path.exists('/storage2/library/archive/grid'):
        return 'config_hdo.ini'
    else:
        return 'config_laptop.ini'


def main():
    if len(sys.argv) < 3:
        print("Usage: $ python save_patched_GRAF_MRMS_GFS.py cyyyymmddhh clead")
        sys.exit(1)

    cyyyymmddhh, clead = sys.argv[1], sys.argv[2]
    config_file = detect_config()
    processor = GRAFDataProcessor(config_file)

    # Date generation logic
    iday_shift = 1 + int(clead) // 24
    ihour_shift = 24 + iday_shift * 24
    date_end1 = dateshift(cyyyymmddhh, -ihour_shift)
    date_begin1 = dateshift(date_end1, -processor.ndays_train * 24)
    date_begin2 = dateshift(cyyyymmddhh, -365*24)
    date_end2 = dateshift(cyyyymmddhh, -305*24)

    cyyyymmddhh_yearprior = dateshift(cyyyymmddhh, -365*24)
    date_end3 = dateshift(cyyyymmddhh_yearprior, -ihour_shift)
    date_begin3 = dateshift(date_end3, -processor.ndays_train * 24)
    date_begin4 = dateshift(cyyyymmddhh_yearprior, -365*24)
    date_end4 = dateshift(cyyyymmddhh_yearprior, -305*24)

    date_list = daterange(date_begin1, date_end1, 6) + daterange(date_begin2, date_end2, 6) + \
                daterange(date_begin3, date_end3, 6) + daterange(date_begin4, date_end4, 6)

    print(f'INFO: Processing {len(date_list)} dates for init={cyyyymmddhh} lead={clead}h')

    # Buckets initialized with time-stamp lists and GFS data lists
    buckets = {
        'train': {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF',
                                  'terrain_diff', 'dt_dlon', 'dt_dlat',
                                  'init_times', 'valid_times',
                                  'GFS_pwat', 'GFS_r', 'GFS_cape']},
        'val':   {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF',
                                  'terrain_diff', 'dt_dlon', 'dt_dlat',
                                  'init_times', 'valid_times',
                                  'GFS_pwat', 'GFS_r', 'GFS_cape']},
        'pred':  {k: [] for k in ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF',
                                  'terrain_diff', 'dt_dlon', 'dt_dlat',
                                  'init_times', 'valid_times',
                                  'GFS_pwat', 'GFS_r', 'GFS_cape']}
    }

    terrain_diff, terr_dlon, terr_dlat = processor.read_terrain()

    ndates_ok = 0
    for idate, date in enumerate(date_list):
        if idate % 50 == 0:
            n_train = len(buckets['train']['GRAF'])
            n_val   = len(buckets['val']['GRAF'])
            n_pred  = len(buckets['pred']['GRAF'])
            print(f'INFO: Date {idate+1}/{len(date_list)} ({date})  '
                  f'patches so far: train={n_train} val={n_val} pred={n_pred}')

        cyyyymmddhh_valid = dateshift(date, int(clead))
        graf_file, _, _ = processor.get_filenames(date, clead)
        istat_graf, precip_graf, lats, lons, _ = processor.read_grib_precip(graf_file, int(clead))
        if istat_graf != 0: continue

        domain_mean = np.mean(precip_graf)
        nsamps = 50 if domain_mean > 0.15 else (28 if domain_mean < 0.10 else 35)

        istat_mrms, precip_mrms, quality_mrms = processor.read_mrms(cyyyymmddhh_valid)
        if istat_mrms != 0: continue

        # Read GFS data
        istat_gfs, gfs_data = processor.read_gfs(date, clead)
        if istat_gfs != 0:
            print(f'  WARNING: Skipping date {date} due to missing GFS data')
            continue

        j_indices, i_indices = processor.select_patches_vectorized(precip_graf, quality_mrms, lats.shape[0], lats.shape[1], nsamps=nsamps)

        if len(j_indices) == 0:
            continue

        ndates_ok += 1

        # Interpolate GFS to patches
        gfs_patches = processor.interpolate_gfs_to_patches(gfs_data, lats, lons, j_indices, i_indices)

        irem = idate % 10
        target_bucket = buckets['train'] if irem >= 2 else (buckets['val'] if irem == 1 else buckets['pred'])

        r = 48
        for idx, (jy, ix) in enumerate(zip(j_indices, i_indices)):
            y_sl, x_sl = slice(jy - r, jy + r), slice(ix - r, ix + r)
            target_bucket['GRAF'].append(precip_graf[y_sl, x_sl].astype(np.float32))
            target_bucket['MRMS'].append(precip_mrms[y_sl, x_sl].astype(np.float32))
            target_bucket['MRMS_qual'].append(quality_mrms[y_sl, x_sl].astype(np.float32))
            target_bucket['terdiff_x_GRAF'].append(terrain_diff[y_sl, x_sl] * precip_graf[y_sl, x_sl])
            target_bucket['terrain_diff'].append(terrain_diff[y_sl, x_sl].astype(np.float32))
            target_bucket['dt_dlon'].append(terr_dlon[y_sl, x_sl].astype(np.float32))
            target_bucket['dt_dlat'].append(terr_dlat[y_sl, x_sl].astype(np.float32))
            # Appending time stamps for each patch
            target_bucket['init_times'].append(date)
            target_bucket['valid_times'].append(cyyyymmddhh_valid)
            # Append GFS patches
            target_bucket['GFS_pwat'].append(gfs_patches[idx]['pwat'])
            target_bucket['GFS_r'].append(gfs_patches[idx]['r'])
            target_bucket['GFS_cape'].append(gfs_patches[idx]['cape'])

        import gc; gc.collect()

    print(f'INFO: Loop complete. {ndates_ok}/{len(date_list)} dates yielded patches.')
    print(f'INFO: Final patch counts: train={len(buckets["train"]["GRAF"])} '
          f'val={len(buckets["val"]["GRAF"])} pred={len(buckets["pred"]["GRAF"])}')

    base_path = processor.dirs.get("resnet_data_directory", "../resnet_data")
    if not os.path.exists(base_path): os.makedirs(base_path)
    save_dataset(f'{base_path}/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick', buckets['train'])
    save_dataset(f'{base_path}/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick', buckets['val'])
    save_dataset(f'{base_path}/GRAF_Unet_data_predict_{cyyyymmddhh}_{clead}h.cPick', buckets['pred'])

if __name__ == "__main__":
    main()
