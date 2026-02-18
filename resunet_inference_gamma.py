"""
python resunet_inference_gamma.py cyyyymmddhh clead
e.g.,
python resunet_inference_gamma.py 2025120412 12

This procedure runs inference using the Gamma mixture model trained by
pytorch_train_resunet_gamma.py.

==============================================================================
GAMMA MIXTURE MODEL INFERENCE
==============================================================================

Instead of predicting 102 categorical probabilities, the model predicts 3
parameters per pixel that define a zero-inflated Gamma distribution:

(1) fraction_zero (p₀): Probability of exactly zero precipitation [0, 1]
(2) shape (α): Gamma distribution shape parameter (α > 0)
(3) scale (θ): Gamma distribution scale parameter (θ > 0)

From these parameters, we compute precipitation probabilities:

    P(X > threshold) = (1 - p₀) × P(Gamma(α, θ) > threshold)
                     = (1 - p₀) × (1 - F_gamma(threshold; α, θ))

where F_gamma is the Gamma CDF.

This script:
1. Loads trained Gamma model weights
2. Reads GRAF, GFS, and terrain data
3. Runs patch-based inference with overlapping patches
4. Computes probabilities for standard thresholds (0.25, 1, 2.5, 5, 10 mm)
5. Saves to netCDF for plotting

Input features (7 channels):
- GRAF precipitation
- Terrain elevation deviation (local terrain height difference)
- GFS column-average RH
- Interaction: GRAF × terrain elevation deviation
- Interaction: GRAF × GFS relative humidity
- Terrain gradient (longitude direction)
- Terrain gradient (latitude direction)

Coded by Tom Hamill with Claude Code assistance, Feb 2026
"""

from configparser import ConfigParser
import numpy as np
import os, sys
import glob
import re
from dateutils import daterange, dateshift
import torch
import torch.nn.functional as F
from pytorch_train_resunet_gamma import AttnResUNet
from netCDF4 import Dataset
import scipy.stats as stats
from scipy.interpolate import RegularGridInterpolator
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
print(f"Running on: {DEVICE}")

TRAIN_DIR = '../resnet_data/trainings'
GFS_DATA_DIR = '../resnet_data/gfs'

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    """Read configuration file for directory paths."""
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
    Define weighting function for patch blending.
    Linear falloff from center to edges prevents discontinuities.
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
    Smoothing sigma for raw GRAF probabilities (for comparison).
    Increases with lead time to account for growing uncertainty.
    """
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
    """Read GRAF precipitation from GRIB2 file."""
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
        print('grib file does not exist.')
        istat = -1
        precipitation = np.empty((0,0))
        lats = np.empty((0,0))
        lons = np.empty((0,0))
        lon_0=0; lat_0=0; lat_1=0; lat_2=0

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2

# ---------------------------------------------------------------

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop):
    """Read GRAF precipitation forecast."""
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
    print(infile, fexist1)

    if fexist1 == True:
        istat, precipitation, lats, lons, lon_0, \
            lat_0, lat_1, lat_2 = read_gribdata(infile, il)
        ny, nx = np.shape(lats)
        latmax = np.max(lats); latmin = np.min(lats)
        lonmax = np.max(lons); lonmin = np.min(lons)
        tzoff = lons*12/180.
        verif_local_time = int(chh_fcst) + tzoff
    else:
        print('  could not find ', infile)
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
    """
    il = int(clead)
    cyyyymm = cyyyymmddhh[0:6]
    filename = f'gfs_subset_{cyyyymmddhh}.nc'
    gfs_file = os.path.join(gfs_data_dir, cyyyymm, filename)

    fexist = os.path.exists(gfs_file)
    print(f'GFS file: {gfs_file}, exists: {fexist}')

    if fexist:
        try:
            nc = Dataset(gfs_file, 'r')

            lats_gfs = nc.variables['latitude'][:]
            lons_gfs = nc.variables['longitude'][:]
            steps = nc.variables['step'][:]

            step_diffs = np.abs(steps - il)
            step_idx = np.argmin(step_diffs)

            if step_diffs[step_idx] > 0:
                print(f'  INFO: GFS exact lead {il}h not found. Using step {steps[step_idx]}h')

            r_gfs = nc.variables['r'][step_idx, :, :]

            nc.close()

            r_gfs = np.where(np.isnan(r_gfs), 0.0, r_gfs)

            lats_gfs_asc = lats_gfs[::-1]

            interp_r = RegularGridInterpolator(
                (lats_gfs_asc, lons_gfs),
                r_gfs[::-1, :],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            graf_lons_360 = np.where(graf_lons < 0, graf_lons + 360, graf_lons)

            ny, nx = graf_lats.shape
            points = np.column_stack([graf_lats.ravel(), graf_lons_360.ravel()])

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
    """Read terrain elevation and gradients."""
    fexist1 = os.path.exists(infile)
    if fexist1 == True:
        nc = Dataset(infile, 'r')
        terrain = nc.variables['terrain_height'][:,:]
        t_diff = nc.variables['terrain_height_local_difference'][:,:]
        dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
        dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]
        nc.close()
    else:
        print('  Could not find desired terrain file.')
        print('  ',infile)
        sys.exit()
    return terrain, t_diff, dt_dlon, dt_dlat

# ---------------------------------------------------------------

def generate_features(nchannels, date, clead, \
        ny, nx, precipitation_GRAF, terrain, t_diff, dt_dlon, \
        dt_dlat, verif_local_time, gfs_rh, norm_stats=None):
    """
    Generate 7-channel feature array for model input.

    Channels:
    0: GRAF precipitation
    1: Terrain elevation deviation
    2: GFS RH
    3: GRAF × terrain
    4: GRAF × RH
    5: dlon gradient
    6: dlat gradient
    """
    def normalize_stats(data, idx):
        if norm_stats is None: return data
        vmin = float(norm_stats['min'][idx])
        vmax = float(norm_stats['max'][idx])
        denom = vmax - vmin
        if denom == 0: denom = 1e-8
        return (data - vmin) / denom

    Xpredict_all = np.zeros((1,nchannels,ny,nx), dtype=float)

    # Match training order: GRAF, terrain_diff, RH, GRAF×terrain, GRAF×RH, dlon, dlat
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

def read_pytorch(cyyyymmddhh, clead):
    """
    Load trained Gamma model weights.

    Returns model with 3 output channels and climatology parameters.
    """
    inference_date_int = int(cyyyymmddhh)
    target_lead = int(clead)
    glob_pattern = os.path.join(TRAIN_DIR, "resunet_gamma_*_best.pth")
    files = glob.glob(glob_pattern)

    if not files:
        print(f"   No Gamma model training files in {TRAIN_DIR} match pattern")
        return None, None, None

    valid_candidates = []
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.search\
            (r"resunet_gamma_(\d{10})_(\d+)h_best\.pth", basename)
        if match:
            fdate = int(match.group(1))
            flead = int(match.group(2))
            if fdate <= inference_date_int:
                valid_candidates.append({'path': fpath, 'date': fdate, \
                    'lead': flead})

    if not valid_candidates:
        print("   No valid Gamma model training checkpoints found.")
        return None, None, None

    available_leads = set(c['lead'] for c in valid_candidates)
    nearest_lead = min(available_leads, key=lambda x: abs(x - target_lead))
    print(f"   Requested Lead: {target_lead}h. Found: {nearest_lead}h")

    best_candidates = [c for c in valid_candidates if c['lead'] == nearest_lead]
    best_candidates.sort(key=lambda x: x['date'], reverse=True)
    b_can = best_candidates[0]
    best_file = b_can['path']
    print(f"   Loading: {best_file}")

    # Gamma model has 3 outputs
    model = AttnResUNet(in_channels=7, num_outputs=3)
    normalization_stats = None
    climatology = None

    try:
        checkpoint = torch.load(best_file, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            normalization_stats = checkpoint.get('normalization_stats', None)
            climatology = checkpoint.get('climatology', None)
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        if normalization_stats: print('   Normalization statistics loaded.')
        else: print('   WARNING: No normalization stats found.')
        if climatology:
            print(f'   Climatology loaded: shape_min={climatology["shape_min"]:.4f}, '
                  f'scale_min={climatology["scale_min"]:.4f}')
        return model, normalization_stats, climatology
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None, None

# -------------------------------------------------------------
# Modular Function 1: Compute Raw GRAF Probabilities
# -------------------------------------------------------------

def calc_raw_probabilities(precipitation_GRAF, sigma):
    """Compute smoothed GRAF probabilities for comparison."""
    raw_probs = {}
    thresholds = {
        '0p25': 0.25, '1': 1.0, '2p5': 2.5,
        '5': 5.0, '10': 10.0
    }
    for key, val in thresholds.items():
        binary_field = np.where(precipitation_GRAF >= val, 1., 0.)
        smoothed_prob = ndimage.gaussian_filter(binary_field, sigma)
        raw_probs[key] = smoothed_prob
    return raw_probs

# -------------------------------------------------------------
# Modular Function 2: Compute Gamma Model Probabilities
# -------------------------------------------------------------

def calc_gamma_probabilities(model, Xpredict_all, manhattan, \
        N, ny, nx, shape_min, scale_min):
    """
    Run patch-based inference with Gamma model.

    For each pixel, the model predicts:
    - fraction_zero (p0)
    - shape (alpha)
    - scale (theta)

    From these, compute P(X > threshold) for standard thresholds.
    """
    # Accumulate weighted parameter predictions
    fraction_zero_accum = np.zeros((ny, nx), dtype=float)
    shape_accum = np.zeros((ny, nx), dtype=float)
    scale_accum = np.zeros((ny, nx), dtype=float)
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

                # Handle edge cases via padding
                _, _, h_curr, w_curr = Xpatch.shape
                pad_h = N - h_curr
                pad_w = N - w_curr

                if pad_h > 0 or pad_w > 0:
                    Xpatch = np.pad(Xpatch, \
                        ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='edge')

                input_tensor = torch.from_numpy(Xpatch).float().to(DEVICE)

                with torch.no_grad():
                    logits = model(input_tensor)

                    # Transform to parameters
                    p0 = torch.sigmoid(logits[:, 0, :, :])
                    alpha = shape_min + F.softplus(logits[:, 1, :, :])
                    theta = scale_min + F.softplus(logits[:, 2, :, :])

                    # Move to CPU for numpy operations
                    p0 = p0.cpu().numpy()
                    alpha = alpha.cpu().numpy()
                    theta = theta.cpu().numpy()

                # Crop back if we padded
                if pad_h > 0 or pad_w > 0:
                    p0 = p0[:, :h_curr, :w_curr]
                    alpha = alpha[:, :h_curr, :w_curr]
                    theta = theta[:, :h_curr, :w_curr]

                # Accumulate weighted parameters
                mh_weight = manhattan
                if pad_h > 0 or pad_w > 0:
                    mh_weight = manhattan[:h_curr, :w_curr]

                fraction_zero_accum[jmin:jmax, imin:imax] += p0[0, :, :] * mh_weight
                shape_accum[jmin:jmax, imin:imax] += alpha[0, :, :] * mh_weight
                scale_accum[jmin:jmax, imin:imax] += theta[0, :, :] * mh_weight
                sumweights_all[jmin:jmax, imin:imax] += mh_weight

    print('Inference Pass 1...')
    process_patches(jcenter1, icenter1)
    print('Inference Pass 2...')
    process_patches(jcenter2, icenter2)

    # Normalize weighted averages
    valid_mask = sumweights_all > 1e-9

    fraction_zero = np.zeros((ny, nx), dtype=float)
    shape_params = np.zeros((ny, nx), dtype=float)
    scale_params = np.zeros((ny, nx), dtype=float)

    np.divide(fraction_zero_accum, sumweights_all, out=fraction_zero, where=valid_mask)
    np.divide(shape_accum, sumweights_all, out=shape_params, where=valid_mask)
    np.divide(scale_accum, sumweights_all, out=scale_params, where=valid_mask)

    # Set defaults for invalid pixels
    fraction_zero[~valid_mask] = 1.0  # Assume dry
    shape_params[~valid_mask] = 1.0
    scale_params[~valid_mask] = 1.0

    # Compute probabilities from Gamma mixture
    # P(X > threshold) = (1 - p0) * P(Gamma > threshold)
    #                  = (1 - p0) * (1 - CDF(threshold))
    #                  = (1 - p0) * SF(threshold)  [survival function]

    print('Computing probabilities from Gamma mixture...')

    gamma_probs = {}
    thresholds = {
        '0p25': 0.25,
        '1': 1.0,
        '2p5': 2.5,
        '5': 5.0,
        '10': 10.0
    }

    for key, threshold in thresholds.items():
        # Vectorized computation using scipy.stats.gamma.sf (survival function)
        # P(X > threshold | X > 0) = sf(threshold, shape, scale=scale)
        prob_exceed = (1.0 - fraction_zero) * stats.gamma.sf(threshold,
                                                              shape_params,
                                                              scale=scale_params)
        gamma_probs[key] = prob_exceed

    # Also return the parameters for potential post-processing
    return gamma_probs, fraction_zero, shape_params, scale_params

# -------------------------------------------------------------
# Modular Function 3: Write NetCDF
# -------------------------------------------------------------

def write_probabilities_to_netcdf(filename, lats, lons, \
        raw_probs, gamma_probs, fraction_zero, shape_params, scale_params):
    """
    Write probabilities and parameters to netCDF.

    Includes:
    - Raw GRAF probabilities (for comparison)
    - Gamma model probabilities
    - Gamma parameters (fraction_zero, shape, scale)
    """
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

            # Gamma Model Variables - stored as int16 with scale_factor
            gamma_name = f'gamma_p{key}mm_prob'
            if key in gamma_probs:
                v = ncfile.createVariable(gamma_name, 'i2', ('y', 'x'),
                                          zlib=True, complevel=4)
                v.scale_factor = 0.0001  # Gives 0.01% precision
                v.add_offset = 0.0
                # Write actual values [0, 1]; netCDF will auto-scale to int16
                v[:] = np.clip(gamma_probs[key], 0.0, 1.0)
                v.long_name = f'Gamma model probability > {key.replace("p", ".")} mm'
                v.units = '1 (dimensionless, 0-1 range)'

        # Save Gamma parameters for diagnostics/post-processing (compressed float32)
        # fraction_zero is a probability (0-1), use int16
        p0_var = ncfile.createVariable('fraction_zero', 'i2', ('y', 'x'),
                                        zlib=True, complevel=4)
        p0_var.scale_factor = 0.0001
        p0_var.add_offset = 0.0
        # Write actual values [0, 1]; netCDF will auto-scale to int16
        p0_var[:] = np.clip(fraction_zero, 0.0, 1.0)
        p0_var.long_name = 'Probability of zero precipitation'
        p0_var.units = '1 (dimensionless, 0-1 range)'

        # Shape and scale parameters: use compressed float32 (not bounded 0-1)
        shape_var = ncfile.createVariable('gamma_shape', 'f4', ('y', 'x'),
                                          zlib=True, complevel=4, least_significant_digit=3)
        shape_var[:] = shape_params
        shape_var.long_name = 'Gamma distribution shape parameter (alpha)'

        scale_var = ncfile.createVariable('gamma_scale', 'f4', ('y', 'x'),
                                          zlib=True, complevel=4, least_significant_digit=3)
        scale_var[:] = scale_params
        scale_var.long_name = 'Gamma distribution scale parameter (theta)'

        # Conditional mean in mm: use compressed float32
        mean_var = ncfile.createVariable('conditional_mean', 'f4', ('y', 'x'),
                                          zlib=True, complevel=4, least_significant_digit=3)
        mean_var[:] = shape_params * scale_params
        mean_var.long_name = 'Conditional mean precipitation given non-zero (mm)'
        mean_var.units = 'mm'

        ncfile.description = \
            "Precipitation probabilities (Raw GRAF vs Gamma Mixture Model with GFS RH)"
        ncfile.history = "Generated by resunet_inference_gamma.py"
        ncfile.close()

    except Exception as e:
        print(f"   Error saving NetCDF: {e}")

# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python resunet_inference_gamma.py <YYYYMMDDHH> <lead>")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    sigma = init_sigma(cyyyymmddhh, clead)

    N = 96
    ny = 1308; nx = 1524
    nchannels = 7  # 7 channels with interactions

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

        # --- Load Gamma model
        model, norm_stats, climatology = read_pytorch(cyyyymmddhh, clead)

        if model and climatology:
            # --- Build array of features (7 channels with interactions).
            model = model.float()
            Xpredict_all, _ = generate_features(nchannels, cyyyymmddhh, \
                clead, ny, nx, precipitation_GRAF, terrain, \
                t_diff, dt_dlon, dt_dlat, verif_local_time, \
                gfs_rh, norm_stats)

            # Get shape and scale minimums from climatology
            shape_min = climatology['shape_min']
            scale_min = climatology['scale_min']

            # --- Compute Gamma Model Probabilities
            gamma_probs, fraction_zero, shape_params, scale_params = \
                calc_gamma_probabilities(model, Xpredict_all, \
                    manhattan, N, ny, nx, shape_min, scale_min)

            # --- Save to NetCDF with _gamma suffix
            probs_out_dir = GRAFprobsdir_conus_laptop
            if not os.path.exists(probs_out_dir):
                try:
                    os.makedirs(probs_out_dir)
                except OSError as e:
                    print(f"Error creating directory {probs_out_dir}: {e}")

            nc_filename = probs_out_dir + cyyyymmddhh + \
                '_' + clead + '_probs_gamma.nc'
            write_probabilities_to_netcdf(nc_filename, \
                lats, lons, raw_probs, gamma_probs,
                fraction_zero, shape_params, scale_params)

            print(f"\nInference complete!")
            print(f"Output saved to: {nc_filename}")

            # Print summary statistics
            print(f"\nSummary statistics:")
            print(f"  Fraction zero: mean={np.mean(fraction_zero):.3f}, "
                  f"min={np.min(fraction_zero):.3f}, max={np.max(fraction_zero):.3f}")
            print(f"  Shape (α): mean={np.mean(shape_params):.3f}, "
                  f"min={np.min(shape_params):.3f}, max={np.max(shape_params):.3f}")
            print(f"  Scale (θ): mean={np.mean(scale_params):.3f}, "
                  f"min={np.min(scale_params):.3f}, max={np.max(scale_params):.3f}")
            mean_given_wet = shape_params * scale_params
            print(f"  Mean|wet: mean={np.mean(mean_given_wet):.3f} mm, "
                  f"max={np.max(mean_given_wet):.3f} mm")

        else:
            print("Model load failed.")
    else:
        if istat_GRAF != 0:
            print('GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
