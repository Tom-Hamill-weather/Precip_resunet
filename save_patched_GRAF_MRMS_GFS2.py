"""save_patched_GRAF_MRMS_GFS2.py

Usage:
    python save_patched_GRAF_MRMS_GFS2.py cyyyymmddhh clead

Arguments:
    cyyyymmddhh : YearMonthDayHour of the initial condition
    clead       : Forecast lead time in hours

Purpose:
    Reads GRAF forecast data, MRMS analyses, and GFS forecast data.
    Extracts 96x96 patches using NON-OVERLAPPING TILED SAMPLING:
      1. Stride-96 tile grid with a date-seeded random global (shift_y, shift_x)
         so terrain appears at different patch-local positions across training days.
      2. Tiles with >50% bad MRMS pixels excluded (training masks them individually
         via ignore_index=-1, so the 50% threshold is conservative but safe).
      3. Wet tiles (patch max GRAF >= 0.5 mm) sampled by mean_precip^1.5 weight;
         n_wet = 35/25/10 by domain wetness.
      4. Dry tiles sampled uniformly; n_dry = 5-8.  No padding to a fixed total.
    Includes GFS features: PWAT, column-average relative humidity (r), and CAPE.
    Saves to NetCDF4 format for U-Net training.
"""

import os
import sys
import warnings
import multiprocessing
from multiprocessing import Pool
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
        self.aws_base_path = self._detect_aws_base()
        self._load_config(config_file)

    def _detect_aws_base(self):
        """Detect AWS base path (G5 vs CPU instance)."""
        if os.path.exists('/data/resnet_data'):
            return '/data/resnet_data'
        elif os.path.exists('/data2/resnet_data'):
            return '/data2/resnet_data'
        return None

    def _load_config(self, config_file):
        """Reads the config.ini file and adapts paths for AWS environment."""
        print(f'INFO: Loading config from {config_file}')
        config = ConfigParser()
        config.read(config_file)

        if "DIRECTORIES" not in config or "PARAMETERS" not in config:
            raise ValueError("Config file missing DIRECTORIES or PARAMETERS sections")

        # If on AWS, replace /data/ prefix with detected base path
        self.dirs = {}
        for key, value in config["DIRECTORIES"].items():
            if self.aws_base_path and config_file == 'config_aws.ini':
                # Replace /data/resnet_data with detected base path
                value = value.replace('/data/resnet_data', self.aws_base_path)
            self.dirs[key] = value

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
            base_dir = self.dirs["grafdatadir_conus_new"]
            prefix = 'grid.hdo-graf_conus.'
        else:
            base_dir = self.dirs["grafdatadir_conus_old"]
            prefix = 'grid.hdo-graflr_conus.'

        input_dir = os.path.join(base_dir, cyyyymmdd, chh)
        filename = (f"{prefix}{cyyyymmdd_fcst}T{chh_fcst}0000Z."
                    f"{cyyyymmdd}T{chh}0000Z.PT{clead}H.CONUS@4km.APCP.SFC.grb2")

        full_path = os.path.join(input_dir, filename)
        return full_path, cyyyymmdd_fcst, chh_fcst

    def read_grib_precip(self, grib_path, end_step, compute_latlons=True):
        """Reads precipitation from GRIB2 file.

        compute_latlons: set False to skip grb.latlons() when the caller already
        has the grid cached (all GRAF files share the same CONUS@4km projection).
        """
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
                lats, lons = (grb.latlons() if compute_latlons else (None, None))
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
        gfs_dir = self.dirs.get("gfs_data_directory")
        if not gfs_dir:
            print('  ERROR: gfs_data_directory not defined in config file')
            return -1, None
        # Extract YYYYMM from cyyyymmddhh for subdirectory
        cyyyymm = cyyyymmddhh[0:6]
        filename = f'gfs_subset_{cyyyymmddhh}.nc'
        filepath = os.path.join(gfs_dir, cyyyymm, filename)

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

        # Gather all patch lat/lon points into one array, then call each
        # interpolator once instead of once-per-patch.
        n_patches = len(j_indices)
        r = 48
        all_lats = np.empty((n_patches, 96, 96), dtype=np.float64)
        all_lons = np.empty((n_patches, 96, 96), dtype=np.float64)
        for k, (jy, ix) in enumerate(zip(j_indices, i_indices)):
            y_sl, x_sl = slice(jy - r, jy + r), slice(ix - r, ix + r)
            all_lats[k] = graf_lats[y_sl, x_sl]
            lp = graf_lons[y_sl, x_sl]
            all_lons[k] = np.where(lp < 0, lp + 360, lp)  # −180:180 → 0:360

        points = np.column_stack([all_lats.ravel(), all_lons.ravel()])  # (N×9216, 2)

        all_pwat = interp_pwat(points).reshape(n_patches, 96, 96).astype(np.float32)
        all_r    = interp_r(points).reshape(n_patches, 96, 96).astype(np.float32)
        all_cape = interp_cape(points).reshape(n_patches, 96, 96).astype(np.float32)

        return [{'pwat': all_pwat[k], 'r': all_r[k], 'cape': all_cape[k]}
                for k in range(n_patches)]

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

    def select_patches_nonoverlapping(self, precip_graf, quality_mrms, ny, nx, cyyyymmddhh):
        """
        Non-overlapping 96x96 patch selection with date-seeded random global shift.

        Tiles the valid domain with stride=96 starting from a random (shift_y, shift_x)
        drawn from [0, 96) so terrain features appear at different patch-local positions
        across training days (prevents over-learning fixed terrain patterns).

        Wet tiles (patch max >= 0.5 mm) are sampled by mean_precip^1.5 weight.
        Dry tiles are sampled uniformly.  Quality threshold is 50% bad pixels
        (individual bad pixels are masked by ignore_index=-1 during training).
        """
        seed = int(cyyyymmddhh) % (2**31)
        rng = np.random.default_rng(seed)

        shift_y = int(rng.integers(0, 96))
        shift_x = int(rng.integers(0, 96))

        y_min = ny // 8 + 65
        y_max = ny * 4 // 5
        x_min = nx // 10
        x_max = 9 * nx // 10

        centers_y = np.arange(y_min + shift_y, y_max, 96)
        centers_x = np.arange(x_min + shift_x, x_max, 96)
        centers_y = centers_y[(centers_y - 48 >= 0) & (centers_y + 48 < ny)]
        centers_x = centers_x[(centers_x - 48 >= 0) & (centers_x + 48 < nx)]

        yy, xx = np.meshgrid(centers_y, centers_x, indexing='ij')
        fy, fx = yy.ravel(), xx.ravel()

        if len(fy) == 0:
            return np.array([]), np.array([])

        patch_mean = ndimage.uniform_filter(precip_graf.astype(float), size=96)
        patch_max  = ndimage.maximum_filter(precip_graf.astype(float), size=96)
        bad_frac   = ndimage.uniform_filter((quality_mrms <= 0.01).astype(float), size=96)

        pmean = patch_mean[fy, fx]
        pmax  = patch_max[fy, fx]
        bfrac = bad_frac[fy, fx]

        valid = bfrac <= 0.50
        fy, fx, pmean, pmax = fy[valid], fx[valid], pmean[valid], pmax[valid]

        if len(fy) == 0:
            return np.array([]), np.array([])

        wet_mask = pmax >= 0.5
        fy_wet, fx_wet = fy[wet_mask],  fx[wet_mask]
        fy_dry, fx_dry = fy[~wet_mask], fx[~wet_mask]
        pm_wet = pmean[wet_mask]

        domain_mean = float(precip_graf.mean())
        n_wet = 35 if domain_mean > 0.15 else (25 if domain_mean >= 0.10 else 10)
        n_dry = int(rng.integers(5, 9))  # 5–8 inclusive

        j_out, i_out = [], []

        if len(fy_wet) > 0:
            w = pm_wet ** 1.5
            w /= w.sum()
            n_take = min(n_wet, len(fy_wet))
            idx = rng.choice(len(fy_wet), size=n_take, replace=False, p=w)
            j_out.extend(fy_wet[idx]);  i_out.extend(fx_wet[idx])

        if len(fy_dry) > 0:
            n_take = min(n_dry, len(fy_dry))
            idx = rng.choice(len(fy_dry), size=n_take, replace=False)
            j_out.extend(fy_dry[idx]);  i_out.extend(fx_dry[idx])

        return np.array(j_out, dtype=int), np.array(i_out, dtype=int)

# ----------------------------------------------------------------

_NC_FLOAT_VARS = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain_diff', 'dt_dlon', 'dt_dlat',
                  'GFS_pwat', 'GFS_r', 'GFS_cape']

def _nc_filename(filename):
    if filename.endswith('.cPick'):
        return filename.replace('.cPick', '.nc')
    return filename if filename.endswith('.nc') else filename + '.nc'


def open_nc_file(filename):
    """Open a new NetCDF4 file for incremental patch appending."""
    filename = _nc_filename(filename)
    from datetime import datetime
    nc = Dataset(filename, 'w', format='NETCDF4')
    nc.createDimension('patch', None)   # unlimited — grows as patches are flushed
    nc.createDimension('y', 96)
    nc.createDimension('x', 96)
    nc.createDimension('time_str_len', 10)

    comp = {'zlib': True, 'complevel': 1, 'shuffle': True}
    chunks = (1, 96, 96)
    for var in _NC_FLOAT_VARS:
        nc.createVariable(var, 'f4', ('patch', 'y', 'x'), chunksizes=chunks, **comp)
    nc.createVariable('init_times',  'S1', ('patch', 'time_str_len'))
    nc.createVariable('valid_times', 'S1', ('patch', 'time_str_len'))

    nc.description = 'Training patches with GRAF, MRMS, terrain, and GFS data'
    nc.history     = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    nc.patch_size  = '96x96'
    nc.format      = 'NetCDF4 with zlib compression (level 4)'
    nc.variables['GRAF'].units            = 'mm';       nc.variables['GRAF'].long_name           = 'GRAF precipitation forecast'
    nc.variables['MRMS'].units            = 'mm';       nc.variables['MRMS'].long_name           = 'MRMS precipitation analysis'
    nc.variables['MRMS_qual'].long_name   = 'MRMS data quality'
    nc.variables['terrain_diff'].units    = 'm';        nc.variables['terrain_diff'].long_name   = 'Local terrain height difference'
    nc.variables['GFS_pwat'].units        = 'kg m-2';   nc.variables['GFS_pwat'].long_name       = 'GFS precipitable water'
    nc.variables['GFS_r'].units           = '%';        nc.variables['GFS_r'].long_name          = 'GFS column-average relative humidity'
    nc.variables['GFS_cape'].units        = 'J kg-1';   nc.variables['GFS_cape'].long_name       = 'GFS CAPE'
    return nc


def flush_to_nc(nc, batch):
    """Append one date's worth of patches to an open NetCDF4 file."""
    n = len(batch['GRAF'])
    if n == 0:
        return
    start = len(nc.dimensions['patch'])
    for var in _NC_FLOAT_VARS:
        nc.variables[var][start:start + n] = np.stack(batch[var])
    for i, (init_t, valid_t) in enumerate(zip(batch['init_times'], batch['valid_times'])):
        nc.variables['init_times'][start + i]  = list(init_t[:10])
        nc.variables['valid_times'][start + i] = list(valid_t[:10])

# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Parallelisation support
# ----------------------------------------------------------------

_BATCH_KEYS = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain_diff', 'dt_dlon', 'dt_dlat',
               'init_times', 'valid_times', 'GFS_pwat', 'GFS_r', 'GFS_cape']

# Per-worker-process globals set by _worker_init (fork-safe on Linux).
_W_PROCESSOR = None   # GRAFDataProcessor instance
_W_TERRAIN   = None   # (terrain_diff, dt_dlon, dt_dlat) tuple
_W_LATLONS   = None   # (lats, lons) cached after first GRIB read


def _worker_init(config_file):
    """Runs once in each worker process at Pool creation."""
    global _W_PROCESSOR, _W_TERRAIN, _W_LATLONS
    warnings.filterwarnings("ignore")
    _W_PROCESSOR = GRAFDataProcessor(config_file)
    _W_TERRAIN   = _W_PROCESSOR.read_terrain()
    _W_LATLONS   = None


def _process_date(args):
    """
    Process one date: read inputs, select patches, interpolate GFS.
    Returns (idate, bucket_name, batch_dict) or None on any failure.
    Called in worker processes via Pool.imap.
    """
    global _W_PROCESSOR, _W_TERRAIN, _W_LATLONS
    idate, date, clead = args
    processor                          = _W_PROCESSOR
    terrain_diff, terr_dlon, terr_dlat = _W_TERRAIN

    cyyyymmddhh_valid = dateshift(date, int(clead))

    # Read GRAF precipitation; skip expensive latlons() after first date
    # because all GRAF files share the same CONUS@4km grid.
    graf_file, _, _ = processor.get_filenames(date, clead)
    need_ll = (_W_LATLONS is None)
    istat, precip_graf, lats, lons, _ = processor.read_grib_precip(
        graf_file, int(clead), compute_latlons=need_ll)
    if istat != 0:
        return None
    if need_ll:
        _W_LATLONS = (lats, lons)
    lats, lons = _W_LATLONS

    istat, precip_mrms, quality_mrms = processor.read_mrms(cyyyymmddhh_valid)
    if istat != 0:
        return None

    istat, gfs_data = processor.read_gfs(date, clead)
    if istat != 0:
        return None

    j_indices, i_indices = processor.select_patches_nonoverlapping(
        precip_graf, quality_mrms, lats.shape[0], lats.shape[1], date)
    if len(j_indices) == 0:
        return None

    gfs_patches = processor.interpolate_gfs_to_patches(
        gfs_data, lats, lons, j_indices, i_indices)

    irem = idate % 10
    bucket = 'train' if irem >= 2 else ('val' if irem == 1 else 'pred')

    r = 48
    batch = {k: [] for k in _BATCH_KEYS}
    for idx, (jy, ix) in enumerate(zip(j_indices, i_indices)):
        y_sl, x_sl = slice(jy - r, jy + r), slice(ix - r, ix + r)
        batch['GRAF'].append(precip_graf[y_sl, x_sl].astype(np.float32))
        batch['MRMS'].append(precip_mrms[y_sl, x_sl].astype(np.float32))
        batch['MRMS_qual'].append(quality_mrms[y_sl, x_sl].astype(np.float32))
        batch['terrain_diff'].append(terrain_diff[y_sl, x_sl].astype(np.float32))
        batch['dt_dlon'].append(terr_dlon[y_sl, x_sl].astype(np.float32))
        batch['dt_dlat'].append(terr_dlat[y_sl, x_sl].astype(np.float32))
        batch['init_times'].append(date)
        batch['valid_times'].append(cyyyymmddhh_valid)
        batch['GFS_pwat'].append(gfs_patches[idx]['pwat'])
        batch['GFS_r'].append(gfs_patches[idx]['r'])
        batch['GFS_cape'].append(gfs_patches[idx]['cape'])

    return idate, bucket, batch


# ----------------------------------------------------------------

def detect_config():
    """Select the appropriate config file based on the runtime environment."""
    # Check for AWS paths (prioritize /data over /data2 for G5 GPU instance)
    if os.path.exists('/data/resnet_data'):
        print('INFO: Detected AWS G5 GPU instance environment')
        return 'config_aws.ini'
    elif os.path.exists('/data2/resnet_data'):
        print('INFO: Detected AWS CPU instance environment')
        return 'config_aws.ini'
    elif os.path.exists('/storage2/library/archive/grid'):
        print('INFO: Detected Cray HPC environment')
        return 'config_hdo.ini'
    else:
        print('INFO: Detected local laptop environment')
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

    # Determine subdirectory: patch_data on G5 GPU, trainings elsewhere
    subdirectory = 'patch_data' if processor.aws_base_path == '/data/resnet_data' else 'trainings'
    base_path = os.path.join(processor.dirs.get("resnet_data_directory", "../resnet_data"), subdirectory)
    os.makedirs(base_path, exist_ok=True)

    # Open output files up front with unlimited patch dimension so we can flush
    # each date's patches immediately instead of accumulating everything in RAM.
    nc_files = {
        'train': open_nc_file(f'{base_path}/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick'),
        'val':   open_nc_file(f'{base_path}/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick'),
        'pred':  open_nc_file(f'{base_path}/GRAF_Unet_data_predict_{cyyyymmddhh}_{clead}h.cPick'),
    }

    n_workers = max(1, multiprocessing.cpu_count() - 2)
    args_list = [(idate, date, clead) for idate, date in enumerate(date_list)]
    print(f'INFO: Using {n_workers} parallel workers')

    ndates_ok = 0
    with Pool(processes=n_workers,
              initializer=_worker_init,
              initargs=(config_file,)) as pool:
        for idx, result in enumerate(pool.imap(_process_date, args_list, chunksize=1)):
            if idx % 50 == 0:
                n_train = len(nc_files['train'].dimensions['patch'])
                n_val   = len(nc_files['val'].dimensions['patch'])
                n_pred  = len(nc_files['pred'].dimensions['patch'])
                print(f'INFO: Date {idx+1}/{len(date_list)}  '
                      f'patches so far: train={n_train} val={n_val} pred={n_pred}')
            if result is None:
                continue
            _, bucket_name, batch = result
            flush_to_nc(nc_files[bucket_name], batch)
            ndates_ok += 1

    print(f'INFO: Loop complete. {ndates_ok}/{len(date_list)} dates yielded patches.')

    counts = {}
    for name, nc in nc_files.items():
        counts[name] = len(nc.dimensions['patch'])
        filename = nc.filepath()
        nc.close()
        size_mb = os.path.getsize(filename) / 1024**2
        print(f'INFO: Done writing {filename} ({size_mb:.1f} MB)')
    print(f'INFO: Final patch counts: train={counts["train"]} val={counts["val"]} pred={counts["pred"]}')

if __name__ == "__main__":
    main()
