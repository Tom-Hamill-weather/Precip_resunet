"""
resunet_inference_quantiler.py

Usage example:

$ python resunet_inference_quantiler.py 2025120412 12

Runs inference using the trained quantile regression U-Net to generate
probabilistic precipitation forecasts.

==============================================================================
QUANTILE REGRESSION INFERENCE
==============================================================================

The model outputs 13 quantiles per pixel representing the precipitation
distribution at percentile levels: 0.01, 0.05, 0.10, 0.20, ..., 0.95, 0.99

To compute exceedance probabilities P(X > threshold):
1. Find quantiles bracketing the threshold
2. Interpolate to find the percentile at the threshold
3. P(X > threshold) = 1 - percentile

Example: If threshold = 2.5mm and quantiles are:
    q_0.60 = 2.0mm, q_0.70 = 3.0mm

Linear interpolation:
    percentile = 0.60 + (0.70-0.60) × (2.5-2.0)/(3.0-2.0) = 0.65

Therefore: P(X > 2.5mm) = 1 - 0.65 = 0.35 (35%)

==============================================================================
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from netCDF4 import Dataset
import pygrib
from configparser import ConfigParser

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Running on: mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on: cuda")
else:
    DEVICE = torch.device("cpu")
    print("Running on: cpu")

PATCH_SIZE = 96
NUM_QUANTILES = 13
QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                             0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

# GFS data directory
GFS_DATA_DIR = '../resnet_data/gfs'

# ---------------------------------------------------------------
# NEURAL NETWORK ARCHITECTURE (same as training)
# ---------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttnResUNet(nn.Module):
    def __init__(self, in_channels=7, num_outputs=NUM_QUANTILES+1):
        super(AttnResUNet, self).__init__()
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))
        self.bridge = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec1 = ResidualBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec2 = ResidualBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResidualBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec4 = ResidualBlock(128, 64)
        self.outc = nn.Conv2d(64, num_outputs, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)
        x = self.up1(x5)
        x4_att = self.att1(g=x, x=x4)
        x = torch.cat([x4_att, x], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x3_att = self.att2(g=x, x=x3)
        x = torch.cat([x3_att, x], dim=1)
        x = self.dec2(x)
        x = self.up3(x)
        x2_att = self.att3(g=x, x=x2)
        x = torch.cat([x2_att, x], dim=1)
        x = self.dec3(x)
        x = self.up4(x)
        x1_att = self.att4(g=x, x=x1)
        x = torch.cat([x1_att, x], dim=1)
        x = self.dec4(x)
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------

def enforce_monotonicity(raw_outputs, max_precip=150.0, max_shift=20.0):
    """
    Shifted quantile regression: extract shift and compute censored quantiles.

    Args:
        raw_outputs: [B, 14, H, W] unconstrained model outputs (1 shift + 13 quantiles)
        max_precip: maximum precipitation value in mm (for scaling)
        max_shift: maximum absolute shift value in mm

    Returns:
        shift: [B, 1, H, W] shift parameter in mm (can be negative)
        quantiles: [B, 13, H, W] censored quantiles in mm, Q = max(0, q + shift)
    """
    # Split outputs
    shift_raw = raw_outputs[:, 0:1, :, :]       # [B, 1, H, W]
    quantiles_raw = raw_outputs[:, 1:, :, :]    # [B, 13, H, W]

    # Shift parameter: use tanh to bound to [-max_shift, +max_shift]
    shift = max_shift * torch.tanh(shift_raw)   # [B, 1, H, W]

    # Build monotonic underlying quantiles (before censoring)
    q_first = F.relu(quantiles_raw[:, 0:1, :, :])  # [B, 1, H, W]

    if quantiles_raw.shape[1] > 1:
        deltas_rest = F.softplus(quantiles_raw[:, 1:, :, :])  # [B, 12, H, W]
        all_deltas = torch.cat([q_first, deltas_rest], dim=1)  # [B, 13, H, W]
        cumulative = torch.cumsum(all_deltas, dim=1)  # [B, 13, H, W]
    else:
        cumulative = q_first

    # Apply saturation
    q_underlying = max_precip * cumulative / (cumulative + max_precip)

    # Apply shift and censoring
    quantiles = torch.clamp(q_underlying + shift, min=0.0)  # [B, 13, H, W]

    return shift, quantiles

def define_manhattan(N):
    """Define Manhattan distance weighting for patch blending."""
    ilocs = np.arange(N)
    jlocs = np.copy(ilocs)
    manhattan = np.zeros((N, N), dtype=float)
    for j in jlocs:
        wj = np.max([0.0, 1. - 2.*np.abs(j+0.5-N/2)/N])
        for i in ilocs:
            wi = np.max([0.0, 1. - 2.*np.abs(i+0.5-N/2)/N])
            manhattan[j, i] = 0.5*wj*wi
    return manhattan

def init_sigma(cyyyymmddhh, clead):
    """
    Smoothing sigma for raw GRAF probabilities (for comparison).
    Larger sigma = more spatial smoothing.
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

def read_config_file(config_file, directory_object_name):
    """Read configuration file for directory paths."""
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]
    GRAFdatadir_conus_laptop = directory["GRAFdatadir_conus_laptop"]
    GRAFprobsdir_conus_laptop = directory["GRAFprobsdir_conus_laptop"]
    return GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop

def dateshift(cyyyymmddhh, fcsthr):
    """Shift date by forecast hours."""
    from dateutils import dateshift as ds
    return ds(cyyyymmddhh, fcsthr)

def read_gribdata(gribfilename, endStep):
    """Read GRIB2 precipitation data."""
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
            print(f'   Error reading {gribfilename}: {e}')
            istat = -1
    else:
        istat = -1
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0))
        lons = np.empty((0, 0))
        lon_0 = 0; lat_0 = 0; lat_1 = 0; lat_2 = 0

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop):
    """Read GRAF precipitation forecast."""
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) > 2024040512:
        input_directory = GRAFdatadir_conus_laptop
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_laptop
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = prefix + cyyyymmdd_fcst + \
        'T' + chh_fcst + '0000Z.' + cyyyymmdd + 'T' + chh + \
        '0000Z.PT' + clead + 'H.CONUS@4km.APCP.SFC.grb2'
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
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0), dtype=float)
        lons = np.empty((0, 0), dtype=float)
        verif_local_time = np.empty((0, 0), dtype=float)

    return istat, precipitation, lats, lons, ny, nx, \
        latmin, latmax, lonmin, lonmax, verif_local_time, \
        lon_0, lat_0, lat_1, lat_2

def read_gfs_data(cyyyymmddhh, clead, gfs_data_dir, lats_graf, lons_graf):
    """
    Read GFS data (RH only) from netCDF files and interpolate to GRAF grid.
    """
    from scipy.interpolate import RegularGridInterpolator

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

            graf_lons_360 = np.where(lons_graf < 0, lons_graf + 360, lons_graf)

            ny, nx = lats_graf.shape
            points = np.column_stack([lats_graf.ravel(), graf_lons_360.ravel()])

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

def read_terrain_characteristics(terrain_file):
    """Read terrain elevation and gradients."""
    fexist = os.path.exists(terrain_file)
    if fexist:
        nc = Dataset(terrain_file, 'r')
        terrain = nc.variables['terrain_height'][:, :]
        t_diff = nc.variables['terrain_height_local_difference'][:, :]
        dt_dlon = nc.variables['dterrain_dlon_smoothed'][:, :]
        dt_dlat = nc.variables['dterrain_dlat_smoothed'][:, :]
        nc.close()
        return terrain, t_diff, dt_dlon, dt_dlat
    else:
        print(f'  Could not find terrain file: {terrain_file}')
        return None, None, None, None

def generate_features(nchannels, cyyyymmddhh, clead, ny, nx,
                      precipitation_GRAF, terrain, t_diff,
                      dt_dlon, dt_dlat, verif_local_time,
                      gfs_rh, normalization_stats):
    """Generate 7-channel feature array using min-max normalization."""
    Xpredict_all = np.zeros((nchannels, ny, nx), dtype=float)

    mins = normalization_stats['min']
    maxes = normalization_stats['max']

    # Channel 0: GRAF precipitation
    Xpredict_all[0, :, :] = (precipitation_GRAF - mins[0]) / (maxes[0] - mins[0])

    # Channel 1: Terrain deviation
    Xpredict_all[1, :, :] = (t_diff - mins[1]) / (maxes[1] - mins[1])

    # Channel 2: GFS RH
    Xpredict_all[2, :, :] = (gfs_rh - mins[2]) / (maxes[2] - mins[2])

    # Channel 3: GRAF × terrain interaction
    interaction_terrain = precipitation_GRAF * t_diff
    Xpredict_all[3, :, :] = (interaction_terrain - mins[3]) / (maxes[3] - mins[3])

    # Channel 4: GRAF × RH interaction
    interaction_rh = precipitation_GRAF * gfs_rh
    Xpredict_all[4, :, :] = (interaction_rh - mins[4]) / (maxes[4] - mins[4])

    # Channel 5: Terrain gradient (longitude)
    Xpredict_all[5, :, :] = (dt_dlon - mins[5]) / (maxes[5] - mins[5])

    # Channel 6: Terrain gradient (latitude)
    Xpredict_all[6, :, :] = (dt_dlat - mins[6]) / (maxes[6] - mins[6])

    return Xpredict_all, verif_local_time

def calc_raw_probabilities(precipitation_GRAF, sigma):
    """Compute smoothed GRAF probabilities for comparison."""
    from scipy import ndimage

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

def read_pytorch(cyyyymmddhh, clead):
    """
    Load trained quantile model weights.
    Searches for available models and uses the best match by lead time and date.
    """
    import glob
    import re

    train_dir = '../resnet_data/trainings'
    inference_date_int = int(cyyyymmddhh)
    target_lead = int(clead)

    glob_pattern = os.path.join(train_dir, "resunet_quantiler_*_best.pth")
    files = glob.glob(glob_pattern)

    if not files:
        print(f"   No quantile model training files in {train_dir}")
        return None, None

    valid_candidates = []
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.search(r"resunet_quantiler_(\d{10})_(\d+)h_best\.pth", basename)
        if match:
            fdate = int(match.group(1))
            flead = int(match.group(2))
            if fdate <= inference_date_int:
                valid_candidates.append({'path': fpath, 'date': fdate, 'lead': flead})

    if not valid_candidates:
        print("   No valid quantile model checkpoints found.")
        return None, None

    available_leads = set(c['lead'] for c in valid_candidates)
    nearest_lead = min(available_leads, key=lambda x: abs(x - target_lead))
    print(f"   Requested Lead: {target_lead}h. Found: {nearest_lead}h")

    best_candidates = [c for c in valid_candidates if c['lead'] == nearest_lead]
    best_candidates.sort(key=lambda x: x['date'], reverse=True)
    b_can = best_candidates[0]
    best_file = b_can['path']
    print(f"   Loading: {best_file}")

    model = AttnResUNet(in_channels=7, num_outputs=NUM_QUANTILES+1)  # +1 for shift parameter

    try:
        checkpoint = torch.load(best_file, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        normalization_stats = checkpoint.get('normalization_stats', None)

        if normalization_stats:
            print('   Normalization statistics loaded.')
        else:
            print('   WARNING: No normalization stats found.')

        return model, normalization_stats

    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None

def compute_exceedance_probability(quantile_values, threshold):
    """
    Compute P(X > threshold) from quantile predictions.

    Args:
        quantile_values: [Q] array of precipitation amounts at each quantile
        threshold: precipitation threshold (mm)

    Returns:
        prob: P(X > threshold)
    """
    # Edge case 1: threshold below all quantiles
    if threshold < quantile_values[0]:
        return 1.0

    # Edge case 2: threshold above all quantiles
    if threshold > quantile_values[-1]:
        return 0.0

    # Find bracketing quantiles
    idx_upper = np.searchsorted(quantile_values, threshold)
    idx_lower = idx_upper - 1

    # Interpolate
    q_lower = QUANTILE_LEVELS[idx_lower]
    q_upper = QUANTILE_LEVELS[idx_upper]
    v_lower = quantile_values[idx_lower]
    v_upper = quantile_values[idx_upper]

    # Handle exact match or very close values
    if abs(v_upper - v_lower) < 1e-8:
        percentile = q_lower
    else:
        percentile = q_lower + (q_upper - q_lower) * (threshold - v_lower) / (v_upper - v_lower)

    return 1.0 - percentile

def calc_quantile_probabilities(model, Xpredict_all, manhattan, N, ny, nx):
    """
    Run model on overlapping patches and compute probabilities via interpolation.

    Returns:
        quantile_probs: dict of probabilities for each threshold
        quantile_fields: [Q, ny, nx] array of quantile predictions
    """
    # Accumulation arrays for quantiles
    quantile_accum = np.zeros((NUM_QUANTILES, ny, nx), dtype=float)
    sumweights_all = np.zeros((ny, nx), dtype=float)

    # Two-pass inference with offset grids
    jcenter1 = np.arange(N//2, ny, N//2)
    icenter1 = np.arange(N//2, nx, N//2)
    jcenter2 = np.arange(N//4, ny, N//2)
    icenter2 = np.arange(N//4, nx, N//2)

    def process_patches(jcenters, icenters):
        for jc in jcenters:
            for ic in icenters:
                jmin = jc - N//2
                jmax = jc + N//2
                imin = ic - N//2
                imax = ic + N//2

                # Bounds checking
                if jmin < 0 or jmax > ny or imin < 0 or imax > nx:
                    continue

                # Extract patch
                patch = Xpredict_all[:, jmin:jmax, imin:imax]

                # Run model
                input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(DEVICE)

                with torch.no_grad():
                    raw_outputs = model(input_tensor)
                    shift, quantiles = enforce_monotonicity(raw_outputs)
                    quantiles_np = quantiles[0, :, :, :].cpu().numpy()  # [Q, H, W]

                # Accumulate weighted quantiles
                for q in range(NUM_QUANTILES):
                    quantile_accum[q, jmin:jmax, imin:imax] += quantiles_np[q, :, :] * manhattan

                sumweights_all[jmin:jmax, imin:imax] += manhattan

    print('Inference Pass 1...')
    process_patches(jcenter1, icenter1)
    print('Inference Pass 2...')
    process_patches(jcenter2, icenter2)

    # Normalize
    print('Normalizing...')
    valid_mask = sumweights_all > 1e-9
    quantile_fields = np.zeros((NUM_QUANTILES, ny, nx), dtype=float)

    for q in range(NUM_QUANTILES):
        quantile_fields[q, valid_mask] = quantile_accum[q, valid_mask] / sumweights_all[valid_mask]

    # Compute probabilities for standard thresholds
    print('Computing exceedance probabilities...')
    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}
    quantile_probs = {}

    for key, thresh in thresholds.items():
        prob_field = np.zeros((ny, nx), dtype=float)

        # Vectorized computation (still needs loop over pixels, but more efficient)
        for j in range(ny):
            for i in range(nx):
                q_vals = quantile_fields[:, j, i]
                prob_field[j, i] = compute_exceedance_probability(q_vals, thresh)

        quantile_probs[key] = prob_field
        print(f'  Computed P(X > {thresh}mm): mean={np.mean(prob_field):.3f}')

    return quantile_probs, quantile_fields

def write_probabilities_to_netcdf(filename, lats, lons,
                                   raw_probs, quantile_probs, quantile_fields):
    """
    Write probabilities and quantile fields to netCDF file.

    Args:
        filename: output file path
        lats, lons: coordinate arrays
        raw_probs: dict of raw GRAF probabilities
        quantile_probs: dict of quantile-based probabilities
        quantile_fields: [Q, ny, nx] quantile predictions
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

            # Quantile Model Variables - stored as int16 with scale_factor
            quantile_name = f'quantile_p{key}mm_prob'
            if key in quantile_probs:
                v = ncfile.createVariable(quantile_name, 'i2', ('y', 'x'),
                                          zlib=True, complevel=4)
                v.scale_factor = 0.0001  # Gives 0.01% precision
                v.add_offset = 0.0
                # Write actual values [0, 1]; netCDF will auto-scale to int16
                v[:] = np.clip(quantile_probs[key], 0.0, 1.0)
                v.long_name = f'Quantile model probability > {key.replace("p", ".")} mm'
                v.units = '1 (dimensionless, 0-1 range)'

        ncfile.description = \
            "Precipitation probabilities (Raw GRAF vs Quantile Regression Model)"
        ncfile.history = "Generated by resunet_inference_quantiler.py"
        ncfile.close()

    except Exception as e:
        print(f"   Error saving NetCDF: {e}")

# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python resunet_inference_quantiler.py <YYYYMMDDHH> <lead>")
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

        # --- Load Quantile model
        model, norm_stats = read_pytorch(cyyyymmddhh, clead)

        if model:
            # --- Build array of features (7 channels with interactions).
            model = model.float()
            Xpredict_all, _ = generate_features(nchannels, cyyyymmddhh,
                clead, ny, nx, precipitation_GRAF, terrain,
                t_diff, dt_dlon, dt_dlat, verif_local_time,
                gfs_rh, norm_stats)

            # --- Compute Quantile Model Probabilities
            quantile_probs, quantile_fields = \
                calc_quantile_probabilities(model, Xpredict_all,
                    manhattan, N, ny, nx)

            # --- Save to NetCDF with _quantiler suffix
            probs_out_dir = GRAFprobsdir_conus_laptop
            if not os.path.exists(probs_out_dir):
                try:
                    os.makedirs(probs_out_dir)
                except OSError as e:
                    print(f"Error creating directory {probs_out_dir}: {e}")

            nc_filename = probs_out_dir + cyyyymmddhh + \
                '_' + clead + '_probs_quantiler.nc'
            write_probabilities_to_netcdf(nc_filename,
                lats, lons, raw_probs, quantile_probs, quantile_fields)

            print(f"\nInference complete!")
            print(f"Output saved to: {nc_filename}")

            # Print summary statistics
            print(f"\nSummary statistics:")
            for i, level in enumerate(QUANTILE_LEVELS):
                q_mean = np.mean(quantile_fields[i, :, :])
                q_max = np.max(quantile_fields[i, :, :])
                print(f"  q_{level:.2f}: mean={q_mean:.3f} mm, max={q_max:.3f} mm")

        else:
            print("Model load failed.")
    else:
        if istat_GRAF != 0:
            print('GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
