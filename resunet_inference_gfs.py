"""
python resunet_inference_gfs.py cyyyymmddhh clead
e.g.,
python resunet_inference_gfs.py 2025120412 12

This procedure is run for the desired initial condition
time and forecast lead time.  It uses the GRAF case data
previously loaded with copy_graf_to_laptop.py, GFS data
from ../resnet_data/gfs/, and the training weights generated
by previously running pytorch_train_resunet_gfs_nopwat.py to generate
CONUS-scale probabilities of precipitation for common event
thresholds.  It saves to netCDF for later plotting by
make_plots_gfs.py

This is the GFS-enhanced version WITHOUT PWAT or CAPE that uses 7 input channels:
- GRAF precipitation
- Terrain elevation deviation (local terrain height difference)
- GFS column-average RH
- Interaction: GRAF × terrain elevation deviation
- Interaction: GRAF × GFS relative humidity
- Terrain gradient (longitude direction)
- Terrain gradient (latitude direction)

PWAT and CAPE were removed due to poor discriminatory power at low precip rates.

Coded by Tom Hamill with Claude Code assistance, Feb 2026
"""

from configparser import ConfigParser
import numpy as np
import os, sys
import glob
import re
from dateutils import daterange, dateshift
import torch
from mpl_toolkits.basemap import Basemap
from pytorch_train_resunet_gfs import AttnResUNet
from netCDF4 import Dataset
import scipy.stats as stats
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import _pickle as cPickle
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)

# --- Set device for inference ---

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print (f"Running on: {DEVICE}")

TRAIN_DIR = '../resnet_data/trainings'
GFS_DATA_DIR = '../resnet_data/gfs'

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]
    GRAFdatadir_conus_laptop = directory["GRAFdatadir_conus_laptop"]
    GRAFprobsdir_conus_laptop = directory["GRAFprobsdir_conus_laptop"]
    return GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop

# ---------------------------------------------------------------

def define_manhattan(N):
    """
    This defines a weighting function for each patch.
    """
    ilocs = np.arange(N)
    jlocs = np.copy(ilocs)
    manhattan = np.zeros((N,N), dtype=float)
    for j in jlocs:
        wj = np.max([0.0, 1. - 2.*np.abs(j+0.5-N/2)/N])
        for i in ilocs:
            wi = np.max([0.0, 1. - 2.*np.abs(i+0.5-N/2)/N])
            manhattan[j,i] = 0.5*wj*wi
    return manhattan

# ---------------------------------------------------------------

def init_sigma(cyyyymmddhh, clead):
    """
    Comparison sigma for raw GRAF smoothing.
    """
    sigmas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
    cmm = cyyyymmddhh[4:6]
    imm = int(cmm)
    # Using float logic for clead
    lc = int(clead)
    if lc <= 6:   sigma = 5.0 * 4./3.
    elif lc <= 12: sigma = 10.0 * 4./3.
    elif lc <= 18: sigma = 10.0 * 4./3.
    elif lc <= 24: sigma = 15.0 * 4./3.
    elif lc <= 30: sigma = 25.0 * 4./3.
    elif lc <= 36: sigma = 30.0 * 4./3.
    elif lc <= 42: sigma = 30.0 * 4./3.
    elif lc <= 48: sigma = 40.0 * 4./3.
    elif lc <= 54: sigma = 50.0 * 4./3.
    elif lc <= 60: sigma = 50.0 * 4./3.
    else:          sigma = 60.0 * 4./3.
    return sigma

# ---------------------------------------------------------------

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
            precipitation = grb.values
            precipitation = np.where(precipitation > 75., 75., precipitation)
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
        lon_0=0; lat_0=0; lat_1=0; lat_2=0

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

# ---------------------------------------------------------------

def read_gfs_data(cyyyymmddhh, clead, gfs_data_dir, graf_lats, graf_lons):
    """
    Read GFS data (RH only) from netCDF files and interpolate to GRAF grid.
    PWAT and CAPE removed due to poor discriminatory power.
    GFS file structure:
    - latitude (1D, decreasing)
    - longitude (1D, 0-360)
    - step (forecast hours)
    - r[step, latitude, longitude] (relative humidity)
    """
    il = int(clead)
    cyyyymm = cyyyymmddhh[0:6]  # Extract YYYYMM for subdirectory
    filename = f'gfs_subset_{cyyyymmddhh}.nc'
    gfs_file = os.path.join(gfs_data_dir, cyyyymm, filename)

    fexist = os.path.exists(gfs_file)
    print(f'GFS file: {gfs_file}, exists: {fexist}')

    if fexist:
        try:
            nc = Dataset(gfs_file, 'r')

            # Read coordinate variables
            lats_gfs = nc.variables['latitude'][:]  # 1D array, decreasing
            lons_gfs = nc.variables['longitude'][:] # 1D array, 0-360
            steps = nc.variables['step'][:]         # Forecast hours

            # Find the closest step to requested lead time
            step_diffs = np.abs(steps - il)
            step_idx = np.argmin(step_diffs)

            if step_diffs[step_idx] > 0:
                print(f'  INFO: GFS exact lead {il}h not found. Using step {steps[step_idx]}h')

            # Read only RH at the selected step (no PWAT, no CAPE)
            r_gfs = nc.variables['r'][step_idx, :, :]         # (latitude, longitude)

            nc.close()

            # Handle NaN values
            r_gfs = np.where(np.isnan(r_gfs), 0.0, r_gfs)

            # Interpolate from GFS lat/lon grid to GRAF grid
            # GFS lats are descending, flip for interpolation
            lats_gfs_asc = lats_gfs[::-1]

            # Create interpolator for RH only
            interp_r = RegularGridInterpolator(
                (lats_gfs_asc, lons_gfs),
                r_gfs[::-1, :],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            # Convert GRAF lons from -180:180 to 0:360 for GFS
            graf_lons_360 = np.where(graf_lons < 0, graf_lons + 360, graf_lons)

            # Create points for interpolation
            ny, nx = graf_lats.shape
            points = np.column_stack([graf_lats.ravel(), graf_lons_360.ravel()])

            # Interpolate
            rh = interp_r(points).reshape(ny, nx)

            istat = 0
            return istat, rh

        except Exception as e:
            print(f'   Error reading GFS data: {e}')
            import traceback
            traceback.print_exc()
            istat = -1
            return istat, None
    else:
        print(f'   Could not find GFS file: {gfs_file}')
        istat = -1
        return istat, None

# ---------------------------------------------------------------

def read_terrain_characteristics(infile):
    fexist1 = os.path.exists(infile)
    if fexist1 == True:
        nc = Dataset(infile, 'r')
        terrain = nc.variables['terrain_height'][:,:]
        t_diff = nc.variables['terrain_height_local_difference'][:,:]
        dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
        dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]
        nc.close()
    else:
        print ('  Could not find desired terrain file.')
        print ('  ',infile)
        sys.exit()
    return terrain, t_diff, dt_dlon, dt_dlat

# ---------------------------------------------------------------

def generate_features(nchannels, date, clead, \
        ny, nx, precipitation_GRAF, terrain, t_diff, dt_dlon, \
        dt_dlat, verif_local_time, gfs_rh, norm_stats=None):

    def normalize_stats(data, idx):
        if norm_stats is None: return data
        vmin = float(norm_stats['min'][idx])
        vmax = float(norm_stats['max'][idx])
        denom = vmax - vmin
        if denom == 0: denom = 1e-8
        return (data - vmin) / denom

    Xpredict_all = np.zeros((1,nchannels,ny,nx), dtype=float)

    # Consistent with Training GRAF_Dataset features (7 channels)
    # Order: GRAF, terrain_diff, RH, GRAF×terrain, GRAF×RH, dlon, dlat
    Xpredict_all[0,0,:,:] = normalize_stats(precipitation_GRAF[:,:], 0)
    Xpredict_all[0,1,:,:] = normalize_stats(t_diff[:,:], 1)
    Xpredict_all[0,2,:,:] = normalize_stats(gfs_rh[:,:], 2)
    interaction_terrain = precipitation_GRAF[:,:] * t_diff[:,:]
    Xpredict_all[0,3,:,:] = normalize_stats(interaction_terrain, 3)
    interaction_rh = precipitation_GRAF[:,:] * gfs_rh[:,:]
    Xpredict_all[0,4,:,:] = normalize_stats(interaction_rh, 4)
    Xpredict_all[0,5,:,:] = normalize_stats(dt_dlon[:,:], 5)
    Xpredict_all[0,6,:,:] = normalize_stats(dt_dlat[:,:], 6)

    return Xpredict_all, precipitation_GRAF

# ---------------------------------------------------------------

def read_pytorch(cyyyymmddhh, clead, num_classes):

    inference_date_int = int(cyyyymmddhh)
    target_lead = int(clead)
    glob_pattern = os.path.join(TRAIN_DIR, "resunet_ordinal_gfs_nopwat_*_best.pth")
    files = glob.glob(glob_pattern)

    if not files:
        print(f"   No GFS nopwat training files in {TRAIN_DIR} match pattern")
        return None, None

    valid_candidates = []
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.search\
            (r"resunet_ordinal_gfs_nopwat_(\d{10})_(\d+)h_best\.pth", basename)
        if match:
            fdate = int(match.group(1))
            flead = int(match.group(2))
            if fdate <= inference_date_int:
                valid_candidates.append({'path': fpath, 'date': fdate, \
                    'lead': flead})

    if not valid_candidates:
        print("   No valid GFS nopwat training checkpoints found.")
        return None, None

    available_leads = set(c['lead'] for c in valid_candidates)
    nearest_lead = min(available_leads, key=lambda x: abs(x - target_lead))
    print(f"   Requested Lead: {target_lead}h. Found: {nearest_lead}h")

    best_candidates = [c for c in valid_candidates if c['lead'] == nearest_lead]
    best_candidates.sort(key=lambda x: x['date'], reverse=True)
    b_can = best_candidates[0]
    best_file = b_can['path']
    print(f"   Loading: {best_file}")

    model = AttnResUNet(in_channels=7, num_classes=num_classes)
    normalization_stats = None

    try:
        checkpoint = torch.load(best_file, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            normalization_stats = checkpoint.get('normalization_stats', None)
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        if normalization_stats: print('   Normalization statistics loaded.')
        else: print('   WARNING: No normalization stats found.')
        return model, normalization_stats
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None

# -------------------------------------------------------------
# Modular Function 1: Compute Raw GRAF Probabilities
# -------------------------------------------------------------

def calc_raw_probabilities(precipitation_GRAF, sigma):
    raw_probs = {}
    thresholds = {
        '0p25': 0.25, '1': 1.0, '2p5': 2.5,
        '5': 5.0, '10': 10.0, '25': 25.0
    }
    for key, val in thresholds.items():
        binary_field = np.where(precipitation_GRAF >= val, 1., 0.)
        smoothed_prob = ndimage.gaussian_filter(binary_field, sigma)
        raw_probs[key] = smoothed_prob
    return raw_probs

# -------------------------------------------------------------
# Modular Function 2: Compute Deep Learning Probabilities (FIXED)
# -------------------------------------------------------------

def calc_dl_probabilities(model, Xpredict_all, manhattan, \
        N, ny, nx, num_classes):

    precip_fcst_prob_all = np.zeros((num_classes, ny, nx), dtype=float)
    sumweights_all = np.zeros((ny, nx), dtype=float)

    jcenter1 = range(N//2, ny-N//2+1, N//2)
    icenter1 = range(N//2, nx-N//2+1, N//2)
    jcenter2 = range(N//2 + N//4, ny-3*N//4, N//2)
    icenter2 = range(N//2 + N//4, nx-3*N//4, N//2)

    def process_patches(jcenters, icenters):
        for j in jcenters:
            jmin = j-N//2; jmax = j+N//2
            for i in icenters:
                imin = i-N//2; imax = i+N//2

                # Extract Patch
                Xpatch = Xpredict_all[:,:,jmin:jmax,imin:imax]

                # --- FIX: Handle Edge Cases via Padding ---
                # At the edges of the domain, slicing might return a patch
                # smaller than (N, N). U-Net requires (N, N) [multiple of 16].
                # We pad, run inference, and then crop back.

                _, _, h_curr, w_curr = Xpatch.shape
                pad_h = N - h_curr
                pad_w = N - w_curr

                if pad_h > 0 or pad_w > 0:
                    # Pad (Right and Bottom only)
                    Xpatch = np.pad(Xpatch, \
                        ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='edge')

                input_tensor = torch.from_numpy(Xpatch).float().to(DEVICE)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                # Crop back if we padded
                if pad_h > 0 or pad_w > 0:
                    probs = probs[:, :, :h_curr, :w_curr]

                # Accumulate
                # Note: Manhattan also needs cropping if at edge
                mh_weight = manhattan
                if pad_h > 0 or pad_w > 0:
                    mh_weight = manhattan[:h_curr, :w_curr]

                precip_fcst_prob_all[:, jmin:jmax, imin:imax] += \
                    probs[0, :, :, :] * mh_weight[None, :, :]
                sumweights_all[jmin:jmax, imin:imax] += mh_weight[:,:]

    print('Inference Pass 1...')
    process_patches(jcenter1, icenter1)
    print('Inference Pass 2...')
    process_patches(jcenter2, icenter2)

    # Normalize weighted averages
    valid_mask = sumweights_all > 1e-9
    for icat in range(num_classes):
        norm_layer = np.zeros((ny, nx), dtype=float)
        np.divide(precip_fcst_prob_all[icat,:,:], sumweights_all, \
                  out=norm_layer, where=valid_mask)

        if icat == 0: norm_layer[~valid_mask] = 1.0
        else: norm_layer[~valid_mask] = 0.0
        precip_fcst_prob_all[icat,:,:] = norm_layer

    dl_probs = {}

    # > 0.25 mm: Start at Class 1 (0.25 to 0.50)
    dl_probs['0p25'] = np.sum(precip_fcst_prob_all[1:,:,:], axis=0)
    # > 1.0 mm: Start at Class 4
    dl_probs['1']    = np.sum(precip_fcst_prob_all[4:,:,:], axis=0)
    # > 2.5 mm: Start at Class 10
    dl_probs['2p5']  = np.sum(precip_fcst_prob_all[10:,:,:], axis=0)
    # > 5.0 mm: Start at Class 20
    dl_probs['5']    = np.sum(precip_fcst_prob_all[20:,:,:], axis=0)
    # > 10.0 mm: Start at Class 40
    dl_probs['10']   = np.sum(precip_fcst_prob_all[40:,:,:], axis=0)

    return dl_probs

# -------------------------------------------------------------
# Modular Function 3: Write NetCDF
# -------------------------------------------------------------

def write_probabilities_to_netcdf(filename, lats, lons, \
        raw_probs, dl_probs):

    ny, nx = lats.shape
    print(f"   Saving probabilities to {filename}")

    try:
        ncfile = Dataset(filename, 'w', format='NETCDF4')
        ncfile.createDimension('y', ny)
        ncfile.createDimension('x', nx)

        # Grid Variables (keep as float32 for coordinates)
        lat_var = ncfile.createVariable('lat', 'f4', ('y', 'x'), zlib=True, complevel=4)
        lon_var = ncfile.createVariable('lon', 'f4', ('y', 'x'), zlib=True, complevel=4)
        lat_var[:] = lats
        lon_var[:] = lons

        keys = ['0p25', '1', '2p5', '5', '10']

        for key in keys:
            # Raw Variables - stored as int16 with scale_factor for compression
            raw_name = f'raw_p{key}mm_prob'
            if key in raw_probs:
                v = ncfile.createVariable(raw_name, 'i2', ('y', 'x'),
                                          zlib=True, complevel=4)
                v.scale_factor = 0.0001  # Gives 0.01% precision
                v.add_offset = 0.0
                # Write actual values [0, 1]; netCDF will auto-scale to int16
                v[:] = np.clip(raw_probs[key], 0.0, 1.0)
                v.long_name = f'Raw GRAF probability > {key.replace("p", ".")} mm'
                v.units = '1 (dimensionless, 0-1 range)'

            # DL Variables - stored as int16 with scale_factor
            dl_name = f'dl_p{key}mm_prob'
            if key in dl_probs:
                v = ncfile.createVariable(dl_name, 'i2', ('y', 'x'),
                                          zlib=True, complevel=4)
                v.scale_factor = 0.0001  # Gives 0.01% precision
                v.add_offset = 0.0
                # Write actual values [0, 1]; netCDF will auto-scale to int16
                v[:] = np.clip(dl_probs[key], 0.0, 1.0)
                v.long_name = f'Deep learning probability > {key.replace("p", ".")} mm'
                v.units = '1 (dimensionless, 0-1 range)'

        ncfile.description = \
            "Precipitation probabilities (Raw GRAF vs Deep Learning with GFS RH feature)"
        ncfile.history = "Generated by resunet_inference_gfs.py"
        ncfile.close()

    except Exception as e:
        print(f"   Error saving NetCDF: {e}")

# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python resunet_inference_gfs.py <YYYYMMDDHH> <lead>")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    sigma = init_sigma(cyyyymmddhh, clead)

    N = 96
    ny = 1308; nx = 1524
    nchannels = 7  # 7 channels: GRAF, terrain, RH, interactions, gradients

    THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
    THRESHOLDS.append(200.0)
    NUM_CLASSES = len(THRESHOLDS)

    config_file_name = 'config_laptop.ini'
    GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop = \
        read_config_file(config_file_name, 'DIRECTORIES')
    manhattan = define_manhattan(N)

    # --- read GRAF forecast

    istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
        lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
        GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop)

    # --- read GFS data (needs GRAF lats/lons for interpolation)

    istat_GFS, gfs_rh = \
        read_gfs_data(cyyyymmddhh, clead, GFS_DATA_DIR, lats, lons)

    if istat_GRAF == 0 and istat_GFS == 0:

        # --- Compute raw probabilities

        raw_probs = calc_raw_probabilities(precipitation_GRAF, sigma)

        # --- Read terrain elevation.

        terrain, t_diff, dt_dlon, dt_dlat = \
            read_terrain_characteristics('GRAF_CONUS_terrain_info.nc')
        model, norm_stats = read_pytorch(cyyyymmddhh, clead, NUM_CLASSES)

        if model:
            # --- Build array of features (7 channels with interactions).
            model = model.float()
            Xpredict_all, _ = generate_features(nchannels, cyyyymmddhh, \
                clead, ny, nx, precipitation_GRAF, terrain, \
                t_diff, dt_dlon, dt_dlat, verif_local_time, \
                gfs_rh, norm_stats)

            # 2. Compute Deep Learning Probabilities
            dl_probs = calc_dl_probabilities(model, Xpredict_all, \
                manhattan, N, ny, nx, NUM_CLASSES)

            # 3. Save to NetCDF with _gfs suffix
            probs_out_dir = GRAFprobsdir_conus_laptop
            if not os.path.exists(probs_out_dir):
                try:
                    os.makedirs(probs_out_dir)
                except OSError as e:
                    print(f"Error creating directory {probs_out_dir}: {e}")

            nc_filename = probs_out_dir + cyyyymmddhh + \
                '_' + clead + '_probs_gfs.nc'
            write_probabilities_to_netcdf(nc_filename, \
                lats, lons, raw_probs, dl_probs)

        else:
            print("Model load failed.")
    else:
        if istat_GRAF != 0:
            print('GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
