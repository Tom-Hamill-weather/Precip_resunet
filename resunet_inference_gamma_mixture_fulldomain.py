"""
python resunet_inference_gamma_mixture_fulldomain.py cyyyymmddhh clead
e.g.,
python resunet_inference_gamma_mixture_fulldomain.py 2025120412 12

Whole-domain (full-field) inference for the 2-component Gamma mixture model.

AttnResUNet is fully convolutional — no Linear or Flatten layers — so it can
process inputs of arbitrary spatial size at inference time.  The only
requirement is that spatial dimensions are divisible by 2^4 = 16 (four
MaxPool2d(2) downsampling stages).  The full CONUS feature tensor is padded
to the next multiple of 16, passed through the model in a single forward
pass, then cropped back to the native 1308x1524 domain.

Benefits over the patch-based version:
- Simpler code: no patch loop, no overlap management, no weighted blending.
- No seam artefacts from patch boundaries.
- Faster in practice: single large forward pass vs. hundreds of small ones.

Memory note: padded to 1312x1536 requires ~2-3 GB GPU VRAM (skip connections
plus activations); well within the G5 24 GB limit.

==============================================================================
2-COMPONENT GAMMA MIXTURE MODEL INFERENCE
==============================================================================

Instead of predicting 102 categorical probabilities, the model predicts 6
parameters per pixel that define a zero-inflated 2-component Gamma mixture:

(1) fraction_zero (p0): Probability of exactly zero precipitation [0, 1]
(2) weight (w): Mixing weight for first component [0, 1]
(3) shape1 (a1): First Gamma shape parameter (light precipitation)
(4) scale1 (t1): First Gamma scale parameter (light precipitation)
(5) shape2 (a2): Second Gamma shape parameter (heavy precipitation, a2 > a1)
(6) scale2 (t2): Second Gamma scale parameter (heavy precipitation)

    P(X > threshold) = (1 - p0) x [w x SF1(threshold) + (1-w) x SF2(threshold)]

Input features (7 channels):
- GRAF precipitation
- Terrain elevation deviation (local terrain height difference)
- GFS column-average RH
- Interaction: GRAF x terrain elevation deviation
- Interaction: GRAF x GFS relative humidity
- Terrain gradient (longitude direction)
- Terrain gradient (latitude direction)
"""

from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os, sys
import glob
import re
from dateutils import daterange, dateshift
import torch
import torch.nn.functional as F
from torch.distributions import Gamma
from pytorch_train_resunet_gamma_mixture import AttnResUNet
from netCDF4 import Dataset
import scipy.ndimage as ndimage
import warnings
from scipy.interpolate import RegularGridInterpolator
warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)

# Pixels to clip from each edge of the output domain.
# Edge pixels are influenced by replicate padding and have no real-data context
# outside the domain; clipping removes the least reliable predictions.
BOUNDARY_CLIP = 16

# --- Set device for inference ---

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Running on: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Running on: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"Running on: {DEVICE}")

# --- Auto-detect environment (AWS vs local) ---
def detect_environment():
    """Detect if running on AWS or local laptop."""
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path

    aws_training_paths = ['/data/trainings', '/data2/trainings']
    for path in aws_training_paths:
        if os.path.exists(path):
            parent = os.path.dirname(path)
            print(f"Detected AWS environment (found {path})")
            return 'aws', parent

    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

if ENVIRONMENT == 'aws':
    TRAIN_DIR = f'{AWS_BASE_PATH}/trainings'
    GFS_DATA_DIR = f'{AWS_BASE_PATH}/gfs'
    print(f"Using AWS paths: TRAIN_DIR={TRAIN_DIR}, GFS_DATA_DIR={GFS_DATA_DIR}")
else:
    TRAIN_DIR = '../resnet_data/trainings'
    GFS_DATA_DIR = '../resnet_data/gfs'
    print(f"Using local paths: TRAIN_DIR={TRAIN_DIR}, GFS_DATA_DIR={GFS_DATA_DIR}")

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    """Read configuration file for directory paths."""
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if "GRAFdatadir_conus_laptop" in directory:
        GRAFdatadir_conus_new = directory["GRAFdatadir_conus_laptop"]
        GRAFdatadir_conus_old = directory["GRAFdatadir_conus_laptop"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus_laptop"]
    else:
        GRAFdatadir_conus_new = directory.get("GRAFdatadir_conus_new")
        GRAFdatadir_conus_old = directory.get("GRAFdatadir_conus_old", GRAFdatadir_conus_new)
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"

    print(f"  GRAF new path: {GRAFdatadir_conus_new}")
    print(f"  GRAF old path: {GRAFdatadir_conus_old}")
    print(f"  Probs path: {GRAFprobsdir_conus}")

    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus

# ---------------------------------------------------------------

def init_sigma(cyyyymmddhh, clead):
    lc = int(clead)
    if   lc <= 12: return 15.
    elif lc <= 24: return 20.
    elif lc <= 36: return 25.
    elif lc <= 48: return 25.
    elif lc <= 60: return 30.
    else:          return 30.

# ---------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    """Read GRAF precipitation from GRIB2 file."""
    import pygrib
    istat = -1
    fexist_grib = os.path.exists(gribfilename)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep=endStep)[0]
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

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    """Read GRAF precipitation forecast."""
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm = cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_old
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = (prefix + cyyyymmdd_fcst +
                  'T' + chh_fcst + '0000Z.' + cyyyymmdd + 'T' + chh +
                  '0000Z.PT' + clead + 'H.CONUS@4km.APCP.SFC.grb2')
    infile = input_directory + input_file
    fexist1 = os.path.exists(infile)
    print(infile, fexist1)

    if fexist1:
        istat, precipitation, lats, lons, lon_0, \
            lat_0, lat_1, lat_2 = read_gribdata(infile, il)
        ny, nx = np.shape(lats)
        latmax = np.max(lats); latmin = np.min(lats)
        lonmax = np.max(lons); lonmin = np.min(lons)
        tzoff = lons * 12 / 180.
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

    return (istat, precipitation, lats, lons, ny, nx,
            latmin, latmax, lonmin, lonmax, verif_local_time,
            lon_0, lat_0, lat_1, lat_2)

# ---------------------------------------------------------------

def read_gfs_data(cyyyymmddhh, clead, gfs_data_dir, graf_lats, graf_lons):
    """Read GFS column-average RH from netCDF and interpolate to GRAF grid."""
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
            return 0, rh

        except Exception as e:
            print(f'   Error reading GFS data: {e}')
            import traceback
            traceback.print_exc()
            return -1, None
    else:
        print(f'   Could not find GFS file: {gfs_file}')
        return -1, None

# ---------------------------------------------------------------

def read_terrain_characteristics(infile):
    """Read terrain elevation and gradients."""
    if not os.path.exists(infile):
        print('  Could not find desired terrain file.')
        print('  ', infile)
        sys.exit()
    nc = Dataset(infile, 'r')
    terrain  = nc.variables['terrain_height'][:,:]
    t_diff   = nc.variables['terrain_height_local_difference'][:,:]
    dt_dlon  = nc.variables['dterrain_dlon_smoothed'][:,:]
    dt_dlat  = nc.variables['dterrain_dlat_smoothed'][:,:]
    nc.close()
    return terrain, t_diff, dt_dlon, dt_dlat

# ---------------------------------------------------------------

def generate_features(nchannels, date, clead,
                      ny, nx, precipitation_GRAF, terrain, t_diff, dt_dlon,
                      dt_dlat, verif_local_time, gfs_rh,
                      norm_stats=None, power_transform=1.0):
    """
    Generate 7-channel feature array for model input and return as GPU tensor.

    Channels: GRAF precip, terrain diff, GFS RH,
              GRAF x terrain, GRAF x RH, dlon gradient, dlat gradient.
    """
    def normalize_stats(data, idx):
        if norm_stats is None: return data
        vmin = float(norm_stats['min'][idx])
        vmax = float(norm_stats['max'][idx])
        denom = vmax - vmin
        if denom == 0: denom = 1e-8
        return (data - vmin) / denom

    if power_transform != 1.0:
        precipitation_GRAF = np.power(precipitation_GRAF, power_transform)

    channels = np.stack([
        normalize_stats(precipitation_GRAF,              0),
        normalize_stats(t_diff,                          1),
        normalize_stats(gfs_rh,                          2),
        normalize_stats(precipitation_GRAF * t_diff,     3),
        normalize_stats(precipitation_GRAF * gfs_rh,     4),
        normalize_stats(dt_dlon,                         5),
        normalize_stats(dt_dlat,                         6),
    ], axis=0).astype(np.float32)

    Xpredict_tensor = torch.from_numpy(channels[np.newaxis]).to(DEVICE)
    return Xpredict_tensor, precipitation_GRAF

# ---------------------------------------------------------------

def read_pytorch(cyyyymmddhh, clead):
    """Load trained 2-component Gamma mixture model weights."""
    inference_date_int = int(cyyyymmddhh)
    target_lead = int(clead)
    glob_pattern = os.path.join(TRAIN_DIR, "resunet_gamma_mixture_*_best.pth")
    files = glob.glob(glob_pattern)

    if not files:
        print(f"   No Gamma mixture model training files in {TRAIN_DIR} match pattern")
        return None, None, None, 1.0

    valid_candidates = []
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.search(
            r"resunet_gamma_mixture_(\d{10})_(\d+)h_best\.pth", basename)
        if match:
            fdate = int(match.group(1))
            flead = int(match.group(2))
            if fdate <= inference_date_int:
                valid_candidates.append({'path': fpath, 'date': fdate, 'lead': flead})

    if not valid_candidates:
        print("   No valid Gamma mixture model training checkpoints found.")
        return None, None, None, 1.0

    available_leads = set(c['lead'] for c in valid_candidates)
    nearest_lead = min(available_leads, key=lambda x: abs(x - target_lead))
    print(f"   Requested Lead: {target_lead}h. Found: {nearest_lead}h")

    best_candidates = sorted(
        [c for c in valid_candidates if c['lead'] == nearest_lead],
        key=lambda x: x['date'], reverse=True)
    best_file = best_candidates[0]['path']
    print(f"   Loading: {best_file}")

    model = AttnResUNet(in_channels=7, num_outputs=6)
    try:
        checkpoint = torch.load(best_file, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            normalization_stats = checkpoint.get('normalization_stats', None)
            climatology = checkpoint.get('climatology', None)
        else:
            model.load_state_dict(checkpoint)
            normalization_stats = None
            climatology = None
        model.to(DEVICE)
        model.eval()
        power_transform = checkpoint.get('power_transform', 1.0)
        if normalization_stats: print('   Normalization statistics loaded.')
        else: print('   WARNING: No normalization stats found.')
        if climatology:
            print(f'   Climatology loaded: shape_min={climatology["shape_min"]:.4f}')
        if power_transform != 1.0:
            print(f'   Power transformation: GRAF^{power_transform}')
        return model, normalization_stats, climatology, power_transform
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None, None, 1.0

# -------------------------------------------------------------
# Modular Function 1: Compute Raw GRAF Probabilities
# -------------------------------------------------------------

def calc_raw_probabilities(precipitation_GRAF, sigma):
    """Compute smoothed GRAF probabilities for comparison."""
    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}

    def compute_one(key_val):
        key, val = key_val
        binary_field = np.where(precipitation_GRAF >= val, 1., 0.)
        return key, ndimage.gaussian_filter(binary_field, sigma)

    with ThreadPoolExecutor(max_workers=len(thresholds)) as executor:
        results = executor.map(compute_one, thresholds.items())

    return dict(results)

# -------------------------------------------------------------
# Modular Function 2: Whole-Domain Gamma Model Inference
# -------------------------------------------------------------

def calc_gamma_probabilities_fulldomain(model, Xpredict_tensor, ny, nx,
                                        shape_min, scale_min):
    """
    Whole-domain inference: pad to a multiple of 16, single forward pass, crop.

    AttnResUNet has 4 MaxPool2d(2) stages so spatial dims must be divisible
    by 2^4 = 16.  Padding uses 'replicate' mode (same as the patch version's
    edge handling) and is applied only to the bottom and right edges.
    """
    DIVISOR = 16  # 2 ** num_downsampling_stages
    _, _, h, w = Xpredict_tensor.shape
    pad_h = (DIVISOR - h % DIVISOR) % DIVISOR
    pad_w = (DIVISOR - w % DIVISOR) % DIVISOR

    if pad_h > 0 or pad_w > 0:
        print(f"Padding from {h}x{w} to {h+pad_h}x{w+pad_w} (divisible by {DIVISOR})")
        Xpad = F.pad(Xpredict_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    else:
        Xpad = Xpredict_tensor

    print(f"Running single forward pass on {Xpad.shape[2]}x{Xpad.shape[3]} domain...")
    with torch.no_grad():
        logits = model(Xpad).float()
        logits = torch.clamp(logits, min=-10, max=10)

        p0 = torch.sigmoid(logits[0, 0])
        w  = torch.sigmoid(logits[0, 1])
        a1 = shape_min + F.softplus(logits[0, 2])
        t1 = scale_min + F.softplus(logits[0, 3])
        a2 = a1 + F.softplus(logits[0, 4]) + 0.5
        t2 = scale_min + F.softplus(logits[0, 5])

    # Crop to native domain
    p0 = p0[:ny, :nx]
    w  = w[:ny, :nx]
    a1 = a1[:ny, :nx]
    t1 = t1[:ny, :nx]
    a2 = a2[:ny, :nx]
    t2 = t2[:ny, :nx]

    # Clamp and NaN-clean interior values before CDF computation
    a1 = torch.nan_to_num(torch.clamp(a1, min=0.1),  nan=1.0)
    t1 = torch.nan_to_num(torch.clamp(t1, min=0.01), nan=1.0)
    a2 = torch.nan_to_num(torch.clamp(a2, min=0.1),  nan=2.0)
    t2 = torch.nan_to_num(torch.clamp(t2, min=0.01), nan=1.0)
    w  = torch.nan_to_num(torch.clamp(w,  0., 1.),   nan=0.5)
    p0 = torch.nan_to_num(p0,                         nan=0.5)

    print("Computing probabilities from Gamma mixture (GPU-accelerated)...")
    g1 = Gamma(concentration=a1, rate=1.0/t1, validate_args=False)
    g2 = Gamma(concentration=a2, rate=1.0/t2, validate_args=False)

    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}
    gamma_probs = {}
    for key, thr in thresholds.items():
        t_tensor = torch.tensor(thr, device=DEVICE, dtype=torch.float32)
        sf1 = torch.clamp(1.0 - g1.cdf(t_tensor), 0., 1.)
        sf2 = torch.clamp(1.0 - g2.cdf(t_tensor), 0., 1.)
        prob = torch.clamp((1.0 - p0) * (w * sf1 + (1.0 - w) * sf2), 0., 1.)
        gamma_probs[key] = torch.nan_to_num(prob, nan=0.0)

    print("Transferring results to CPU...")
    gamma_probs_np = {k: v.cpu().numpy() for k, v in gamma_probs.items()}
    p0_np = p0.cpu().numpy(); w_np  = w.cpu().numpy()
    a1_np = a1.cpu().numpy(); t1_np = t1.cpu().numpy()
    a2_np = a2.cpu().numpy(); t2_np = t2.cpu().numpy()

    # Fill border pixels with NaN — contaminated by replicate / zero-padding
    # in the forward pass.  NaN is written as the netCDF fill value so these
    # pixels are masked and not plotted.
    clip = BOUNDARY_CLIP
    for arr in (p0_np, w_np, a1_np, t1_np, a2_np, t2_np):
        arr[:clip, :] = np.nan;  arr[-clip:, :] = np.nan
        arr[:, :clip] = np.nan;  arr[:, -clip:] = np.nan
    for arr in gamma_probs_np.values():
        arr[:clip, :] = np.nan;  arr[-clip:, :] = np.nan
        arr[:, :clip] = np.nan;  arr[:, -clip:] = np.nan

    return (gamma_probs_np, p0_np, w_np, a1_np, t1_np, a2_np, t2_np)

# -------------------------------------------------------------
# Modular Function 3: Write NetCDF
# -------------------------------------------------------------

def write_probabilities_to_netcdf(filename, lats, lons, raw_probs, gamma_probs,
                                  fraction_zero, weight_params,
                                  shape1_params, scale1_params,
                                  shape2_params, scale2_params):
    """Write probabilities and parameters to netCDF."""
    ny, nx = lats.shape
    print(f"   Saving probabilities to {filename}")

    try:
        ncfile = Dataset(filename, 'w', format='NETCDF4')
        ncfile.createDimension('y', ny)
        ncfile.createDimension('x', nx)

        lat_var = ncfile.createVariable('lat', 'f4', ('y', 'x'), zlib=True, complevel=4)
        lon_var = ncfile.createVariable('lon', 'f4', ('y', 'x'), zlib=True, complevel=4)
        lat_var[:] = lats
        lon_var[:] = lons

        # int16 probability variables use a fill_value so that NaN border pixels
        # are stored as the missing-data sentinel and returned as masked arrays
        # on read.  scale_factor maps stored int16 → float [0, 1].
        INT16_FILL = -32767

        def write_prob_i2(name, data, long_name):
            v = ncfile.createVariable(name, 'i2', ('y', 'x'),
                                      zlib=True, complevel=4, fill_value=INT16_FILL)
            v.scale_factor = 0.0001
            v.add_offset = 0.0
            v[:] = np.ma.masked_invalid(np.clip(data, 0.0, 1.0))
            v.long_name = long_name
            v.units = '1 (dimensionless, 0-1 range)'

        keys = ['0p25', '1', '2p5', '5', '10']
        for key in keys:
            if key in raw_probs:
                write_prob_i2(f'raw_p{key}mm_prob', raw_probs[key],
                              f'Raw GRAF probability > {key.replace("p", ".")} mm')
            if key in gamma_probs:
                write_prob_i2(f'gamma_p{key}mm_prob', gamma_probs[key],
                              f'Gamma model probability > {key.replace("p", ".")} mm')

        write_prob_i2('fraction_zero', fraction_zero,
                      'Probability of zero precipitation')
        write_prob_i2('mixture_weight', weight_params,
                      'Mixture weight for component 1 (light precipitation)')

        for vname, arr, ln in [
            ('gamma_shape1', shape1_params, 'Gamma component 1 shape parameter (alpha1, light precip)'),
            ('gamma_scale1', scale1_params, 'Gamma component 1 scale parameter (theta1, light precip)'),
            ('gamma_shape2', shape2_params, 'Gamma component 2 shape parameter (alpha2, heavy precip)'),
            ('gamma_scale2', scale2_params, 'Gamma component 2 scale parameter (theta2, heavy precip)'),
        ]:
            v = ncfile.createVariable(vname, 'f4', ('y', 'x'),
                                      zlib=True, complevel=4, least_significant_digit=3)
            v[:] = np.ma.masked_invalid(arr)
            v.long_name = ln

        mean1_var = ncfile.createVariable('conditional_mean1', 'f4', ('y', 'x'),
                                           zlib=True, complevel=4, least_significant_digit=3)
        mean1_var[:] = np.ma.masked_invalid(shape1_params * scale1_params)
        mean1_var.long_name = 'Component 1 mean precipitation given non-zero (mm)'
        mean1_var.units = 'mm'

        mean2_var = ncfile.createVariable('conditional_mean2', 'f4', ('y', 'x'),
                                           zlib=True, complevel=4, least_significant_digit=3)
        mean2_var[:] = np.ma.masked_invalid(shape2_params * scale2_params)
        mean2_var.long_name = 'Component 2 mean precipitation given non-zero (mm)'
        mean2_var.units = 'mm'

        mixture_mean = (weight_params * shape1_params * scale1_params +
                       (1 - weight_params) * shape2_params * scale2_params)
        mean_mix_var = ncfile.createVariable('mixture_mean', 'f4', ('y', 'x'),
                                              zlib=True, complevel=4, least_significant_digit=3)
        mean_mix_var[:] = np.ma.masked_invalid(mixture_mean)
        mean_mix_var.long_name = 'Mixture mean precipitation given non-zero (mm)'
        mean_mix_var.units = 'mm'

        ncfile.description = \
            "Precipitation probabilities (Raw GRAF vs 2-Component Gamma Mixture) - FULL DOMAIN"
        ncfile.history = "Generated by resunet_inference_gamma_mixture_fulldomain.py"
        ncfile.close()

    except Exception as e:
        print(f"   Error saving NetCDF: {e}")

# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python resunet_inference_gamma_mixture_fulldomain.py <YYYYMMDDHH> <lead>")
        sys.exit(1)

    import time
    start_time = time.time()

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    sigma = init_sigma(cyyyymmddhh, clead)

    ny = 1308; nx = 1524
    nchannels = 7

    config_file_name = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    print(f"Using config file: {config_file_name}")
    GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus_laptop = \
        read_config_file(config_file_name, 'DIRECTORIES')

    # --- Read GRAF forecast ---
    istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
        lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
        GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)

    # --- Read GFS column-average RH ---
    istat_GFS, gfs_rh = read_gfs_data(cyyyymmddhh, clead, GFS_DATA_DIR, lats, lons)

    if istat_GRAF == 0 and istat_GFS == 0:

        raw_probs = calc_raw_probabilities(precipitation_GRAF, sigma)

        if ENVIRONMENT == 'aws':
            terrain_file = f'{AWS_BASE_PATH}/terrain/GRAF_CONUS_terrain_info.nc'
        else:
            terrain_file = 'GRAF_CONUS_terrain_info.nc'

        print(f"Reading terrain from: {terrain_file}")
        terrain, t_diff, dt_dlon, dt_dlat = read_terrain_characteristics(terrain_file)

        model, norm_stats, climatology, power_transform = read_pytorch(cyyyymmddhh, clead)

        if model and climatology:
            model = model.float()
            inference_start = time.time()

            Xpredict_tensor, _ = generate_features(
                nchannels, cyyyymmddhh, clead,
                ny, nx, precipitation_GRAF, terrain,
                t_diff, dt_dlon, dt_dlat, verif_local_time,
                gfs_rh, norm_stats, power_transform=power_transform)

            shape_min = climatology['shape_min']
            scale_min = climatology['scale_min']

            (gamma_probs, fraction_zero, weight_params,
             shape1_params, scale1_params, shape2_params, scale2_params) = \
                calc_gamma_probabilities_fulldomain(
                    model, Xpredict_tensor, ny, nx, shape_min, scale_min)

            inference_time = time.time() - inference_start
            print(f"\nInference time: {inference_time:.2f} seconds")

            probs_out_dir = GRAFprobsdir_conus_laptop
            os.makedirs(probs_out_dir, exist_ok=True)

            nc_filename = (probs_out_dir + cyyyymmddhh +
                           '_' + clead + '_probs_gamma_mixture.nc')
            write_probabilities_to_netcdf(
                nc_filename, lats, lons, raw_probs, gamma_probs,
                fraction_zero, weight_params,
                shape1_params, scale1_params, shape2_params, scale2_params)

            total_time = time.time() - start_time
            print(f"\nInference complete!")
            print(f"Output saved to: {nc_filename}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Inference-only time: {inference_time:.2f} seconds")

        else:
            print("Model load failed.")
    else:
        if istat_GRAF != 0:
            print('GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
