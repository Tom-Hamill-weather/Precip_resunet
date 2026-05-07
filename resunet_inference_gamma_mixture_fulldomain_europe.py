"""
python resunet_inference_gamma_mixture_fulldomain_europe.py cyyyymmddhh clead
e.g.,
python resunet_inference_gamma_mixture_fulldomain_europe.py 2025120412 12

European-domain whole-field inference for the 2-component Gamma mixture model.

AttnResUNet is fully convolutional — no Linear or Flatten layers — so it can
process inputs of arbitrary spatial size at inference time.  The only
requirement is that spatial dimensions are divisible by 2^4 = 16 (four
MaxPool2d(2) downsampling stages).  The full European feature tensor is padded
to the next multiple of 16, passed through the model in a single forward
pass, then cropped back to the native 723x666 domain.

Key differences from the CONUS version:
  - Target grid: 723 x 666 (European LCC, ~4 km)
  - GRAF file naming: hdo-graf_europe / hdo-graflr_europe, EUROPE@4km.APCP
  - Terrain: GRAF_Europe_terrain_info.nc
  - GFS RH: downloaded on the fly from s3://noaa-gfs-bdp-pds via HTTP
    byte-range requests.
  - Output: <probs_dir>/<cyyyymmddhh>_<lead>_probs_europe_gamma_mixture.nc

Transfer learning: uses the same gamma-mixture weights trained on CONUS data.
"""

from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os, sys, tempfile, re, glob
import requests
from dateutils import daterange, dateshift
import torch
import torch.nn.functional as F
from torch.distributions import Gamma
from pytorch_train_resunet_gamma_mixture import AttnResUNet
from netCDF4 import Dataset
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)

# Pixels to clip from each edge of the output domain.
BOUNDARY_CLIP = 16

# ---------------------------------------------------------------
# Device
# ---------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Running on: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Running on: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"Running on: {DEVICE}")

# ---------------------------------------------------------------
# Environment detection (identical to CONUS version)
# ---------------------------------------------------------------

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    for path in ['/data/trainings', '/data2/trainings']:
        if os.path.exists(path):
            parent = os.path.dirname(path)
            print(f"Detected AWS environment (found {path})")
            return 'aws', parent
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

if ENVIRONMENT == 'aws':
    TRAIN_DIR = f'{AWS_BASE_PATH}/trainings'
    print(f"Using AWS paths: TRAIN_DIR={TRAIN_DIR}")
else:
    TRAIN_DIR = '../resnet_data/trainings'
    print(f"Using local paths: TRAIN_DIR={TRAIN_DIR}")

# ---------------------------------------------------------------
# GFS on-the-fly download from NOAA S3
# ---------------------------------------------------------------

GFS_S3_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'

def _gfs_url(cyyyymmddhh, forecast_hour):
    """Build GFS 0.25-degree pgrb2 URL and index URL for a given IC and lead."""
    cyyyymmdd = cyyyymmddhh[:8]
    chh       = cyyyymmddhh[8:10]
    fhr_str   = f'{int(forecast_hour):03d}'
    base = (f'{GFS_S3_BASE}/gfs.{cyyyymmdd}/{chh}/atmos/'
            f'gfs.t{chh}z.pgrb2.0p25.f{fhr_str}')
    return base, base + '.idx'


def _parse_index(idx_text, search_str):
    """
    Return (byte_start, byte_end) for the first line in the GFS index that
    contains search_str.  byte_end is the start of the next entry minus 1,
    or None (meaning read to end-of-file) for the last entry.
    """
    lines = [l for l in idx_text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if search_str in line:
            byte_start = int(line.split(':')[1])
            if i + 1 < len(lines):
                byte_end = int(lines[i + 1].split(':')[1]) - 1
            else:
                byte_end = None
            return byte_start, byte_end
    return None, None


def _download_field(data_url, byte_start, byte_end):
    """HTTP byte-range download; returns raw bytes of one GRIB2 message."""
    range_hdr = (f'bytes={byte_start}-{byte_end}'
                 if byte_end is not None
                 else f'bytes={byte_start}-')
    resp = requests.get(data_url, headers={'Range': range_hdr}, timeout=60)
    if resp.status_code not in (200, 206):
        raise IOError(f'HTTP {resp.status_code} fetching {data_url} '
                      f'range {range_hdr}')
    return resp.content


def read_gfs_data_europe(cyyyymmddhh, clead, euro_lats, euro_lons):
    """
    Download GFS column-average RH for the European domain on the fly from
    s3://noaa-gfs-bdp-pds (via HTTPS).

    The field is "RH: entire atmosphere (considered as a single layer)" —
    the same field stored as 'r' in the CONUS gfs_subset_*.nc training files.
    """
    import pygrib

    il = int(clead)
    data_url, idx_url = _gfs_url(cyyyymmddhh, il)
    print(f'GFS index URL: {idx_url}')

    try:
        idx_resp = requests.get(idx_url, timeout=30)
        if idx_resp.status_code != 200:
            print(f'  GFS index not found (HTTP {idx_resp.status_code}): {idx_url}')
            return -1, None
        idx_text = idx_resp.text
    except Exception as e:
        print(f'  Failed to fetch GFS index: {e}')
        return -1, None

    rh_start, rh_end = _parse_index(
        idx_text, 'RH:entire atmosphere (considered as a single layer)')
    if rh_start is None:
        print('  Column-average RH entry not found in GFS index.')
        return -1, None

    print(f'  Downloading column-avg RH: bytes {rh_start}-{rh_end}')
    try:
        rh_bytes = _download_field(data_url, rh_start, rh_end)
    except Exception as e:
        print(f'  Failed to download RH field: {e}')
        return -1, None

    with tempfile.NamedTemporaryFile(suffix='.grb2', delete=False) as tmp:
        tmp.write(rh_bytes)
        tmpname = tmp.name

    try:
        f = pygrib.open(tmpname)
        grb_rh = f.read(1)[0]
        lats_gfs, lons_gfs = grb_rh.latlons()
        rh_global = grb_rh.values.astype(np.float32)
        f.close()
    except Exception as e:
        print(f'  Error reading RH grib: {e}')
        os.unlink(tmpname)
        return -1, None
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)

    rh_global = np.where(np.isnan(rh_global), 0.0, rh_global)

    lats_1d = lats_gfs[:, 0]
    lons_1d = lons_gfs[0, :]

    lats_asc = lats_1d[::-1]
    rh_asc   = rh_global[::-1, :]

    # Roll to -180..179.75 so the European domain (-20..+29 lon) is contiguous.
    roll_by   = lons_1d.size // 2
    lons_rolled = np.concatenate(
        [lons_1d[roll_by:] - 360.0, lons_1d[:roll_by]])
    rh_rolled   = np.concatenate(
        [rh_asc[:, roll_by:], rh_asc[:, :roll_by]], axis=1)

    interp = RegularGridInterpolator(
        (lats_asc, lons_rolled),
        rh_rolled,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    ny, nx = euro_lats.shape
    pts = np.column_stack([euro_lats.ravel(), euro_lons.ravel()])
    rh_euro = interp(pts).reshape(ny, nx)

    print(f'  GFS RH (Europe): min={rh_euro.min():.1f} max={rh_euro.max():.1f} %')
    return 0, rh_euro

# ---------------------------------------------------------------
# GRAF Europe file reading
# ---------------------------------------------------------------

def GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe):
    """Read European GRAF hourly precipitation from GRIB2."""
    import pygrib

    il = int(clead)
    cyyyymmdd        = cyyyymmddhh[:8]
    chh              = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst   = cyyyymmddhh_fcst[:8]
    chh_fcst         = cyyyymmddhh_fcst[8:10]

    input_dir = GRAFdatadir_europe + cyyyymmdd + '/' + chh + '/'

    for prefix in ('grid.hdo-graf_europe.', 'grid.hdo-graflr_europe.'):
        fname = (prefix
                 + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.'
                 + cyyyymmdd + 'T' + chh + '0000Z.'
                 + 'PT' + clead + 'H.EUROPE@4km.APCP.SFC.grb2')
        infile = input_dir + fname
        if os.path.exists(infile):
            break
    else:
        print(f'  Could not find European GRAF file in {input_dir}')
        return (-1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)),
                0, 0, -99., -99., -999., -999., np.empty((0,0)),
                -999., -999., -999., -999.)

    print(infile, True)
    try:
        f = pygrib.open(infile)
        grb = f.select(endStep=il)[0]
        lats, lons = grb.latlons()
        precipitation = np.where(grb.values > 75., 75., grb.values)
        lon_0 = grb.projparams['lon_0']
        lat_0 = grb.projparams['lat_0']
        lat_1 = grb.projparams['lat_1']
        lat_2 = grb.projparams['lat_2']
        f.close()
    except Exception as e:
        print(f'  Error reading {infile}: {e}')
        return (-1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)),
                0, 0, -99., -99., -999., -999., np.empty((0,0)),
                -999., -999., -999., -999.)

    ny, nx = lats.shape
    tzoff = lons * 12 / 180.
    verif_local_time = int(chh_fcst) + tzoff

    return (0, precipitation, lats, lons, ny, nx,
            lats.min(), lats.max(), lons.min(), lons.max(),
            verif_local_time, lon_0, lat_0, lat_1, lat_2)

# ---------------------------------------------------------------
# Shared helpers (unchanged from CONUS version)
# ---------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if 'GRAFdatadir_conus_laptop' in directory:
        GRAFdatadir_europe = directory.get(
            'GRAFdatadir_europe_laptop',
            directory['GRAFdatadir_conus_laptop'].replace('conus', 'europe'))
        GRAFprobsdir = directory['GRAFprobsdir_conus_laptop']
    else:
        GRAFdatadir_europe = directory.get(
            'GRAFdatadir_europe',
            '/data/resnet_data/GRAF/hdo-graf_europe/')
        base_dir = directory.get('resnet_data_directory',
                                 AWS_BASE_PATH or '/data/resnet_data')
        GRAFprobsdir = f'{base_dir}/probs/'

    print(f'  GRAF Europe path: {GRAFdatadir_europe}')
    print(f'  Probs path: {GRAFprobsdir}')
    return GRAFdatadir_europe, GRAFprobsdir


def init_sigma(cyyyymmddhh, clead):
    lc = int(clead)
    if   lc <= 12: return 15.
    elif lc <= 24: return 20.
    elif lc <= 36: return 25.
    elif lc <= 48: return 25.
    elif lc <= 60: return 30.
    else:          return 30.


def read_terrain_characteristics(infile):
    if not os.path.exists(infile):
        print(f'  Could not find terrain file: {infile}')
        sys.exit(1)
    nc = Dataset(infile, 'r')
    terrain  = nc.variables['terrain_height'][:,:]
    t_diff   = nc.variables['terrain_height_local_difference'][:,:]
    dt_dlon  = nc.variables['dterrain_dlon_smoothed'][:,:]
    dt_dlat  = nc.variables['dterrain_dlat_smoothed'][:,:]
    nc.close()
    return terrain, t_diff, dt_dlon, dt_dlat


def generate_features(nchannels, date, clead,
                      ny, nx, precipitation_GRAF,
                      terrain, t_diff, dt_dlon, dt_dlat,
                      verif_local_time, gfs_rh,
                      norm_stats=None, power_transform=1.0):
    def normalize_stats(data, idx):
        if norm_stats is None: return data
        vmin  = float(norm_stats['min'][idx])
        vmax  = float(norm_stats['max'][idx])
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

    return torch.from_numpy(channels[np.newaxis]).to(DEVICE), precipitation_GRAF


def read_pytorch(cyyyymmddhh, clead):
    """Load trained gamma-mixture model (same weights as CONUS)."""
    inference_date_int = int(cyyyymmddhh)
    target_lead = int(clead)
    files = glob.glob(os.path.join(TRAIN_DIR, 'resunet_gamma_mixture_*_best.pth'))

    if not files:
        print(f'  No gamma mixture weights found in {TRAIN_DIR}')
        return None, None, None, 1.0

    valid = []
    for fpath in files:
        m = re.search(r'resunet_gamma_mixture_(\d{10})_(\d+)h_best\.pth',
                      os.path.basename(fpath))
        if m and int(m.group(1)) <= inference_date_int:
            valid.append({'path': fpath,
                          'date': int(m.group(1)),
                          'lead': int(m.group(2))})

    if not valid:
        print('  No valid gamma mixture checkpoints found.')
        return None, None, None, 1.0

    nearest_lead = min({c['lead'] for c in valid},
                       key=lambda x: abs(x - target_lead))
    best = sorted([c for c in valid if c['lead'] == nearest_lead],
                  key=lambda x: x['date'], reverse=True)[0]
    print(f'  Requested lead {target_lead}h -> using {nearest_lead}h weights')
    print(f'  Loading: {best["path"]}')

    model = AttnResUNet(in_channels=7, num_outputs=6)
    try:
        ckpt = torch.load(best['path'], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt
                              else ckpt)
        model.to(DEVICE)
        model.eval()
        norm_stats   = ckpt.get('normalization_stats', None)
        climatology  = ckpt.get('climatology', None)
        power_transform = ckpt.get('power_transform', 1.0)
        if norm_stats:   print('  Normalization stats loaded.')
        if climatology:  print(f'  Climatology: shape_min={climatology["shape_min"]:.4f}')
        if power_transform != 1.0:
            print(f'  Power transform: GRAF^{power_transform}')
        return model, norm_stats, climatology, power_transform
    except Exception as e:
        print(f'  Error loading model: {e}')
        return None, None, None, 1.0


def calc_raw_probabilities(precipitation_GRAF, sigma):
    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}

    def compute_one(kv):
        k, v = kv
        return k, ndimage.gaussian_filter(
            np.where(precipitation_GRAF >= v, 1., 0.), sigma)

    with ThreadPoolExecutor(max_workers=len(thresholds)) as ex:
        return dict(ex.map(compute_one, thresholds.items()))

# -------------------------------------------------------------
# Whole-domain Gamma model inference
# -------------------------------------------------------------

def calc_gamma_probabilities_fulldomain(model, Xpredict_tensor, ny, nx,
                                        shape_min, scale_min):
    """
    Whole-domain inference: pad to a multiple of 16, single forward pass, crop.

    AttnResUNet has 4 MaxPool2d(2) stages so spatial dims must be divisible
    by 2^4 = 16.  Padding uses 'replicate' mode and is applied to the bottom
    and right edges only.
    """
    DIVISOR = 16
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
        logits = torch.clamp(logits, -10, 10)

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
    clip = BOUNDARY_CLIP
    for arr in (p0_np, w_np, a1_np, t1_np, a2_np, t2_np):
        arr[:clip, :] = np.nan;  arr[-clip:, :] = np.nan
        arr[:, :clip] = np.nan;  arr[:, -clip:] = np.nan
    for arr in gamma_probs_np.values():
        arr[:clip, :] = np.nan;  arr[-clip:, :] = np.nan
        arr[:, :clip] = np.nan;  arr[:, -clip:] = np.nan

    return (gamma_probs_np, p0_np, w_np, a1_np, t1_np, a2_np, t2_np)


def write_probabilities_to_netcdf(filename, lats, lons,
                                  raw_probs, gamma_probs,
                                  fraction_zero, weight_params,
                                  shape1_params, scale1_params,
                                  shape2_params, scale2_params):
    ny, nx = lats.shape
    print(f'  Saving to {filename}')
    try:
        nc = Dataset(filename, 'w', format='NETCDF4')
        nc.createDimension('y', ny)
        nc.createDimension('x', nx)

        lv = nc.createVariable('lat', 'f4', ('y','x'), zlib=True, complevel=4)
        lv[:] = lats
        lo = nc.createVariable('lon', 'f4', ('y','x'), zlib=True, complevel=4)
        lo[:] = lons

        INT16_FILL = -32767

        def write_prob_i2(name, data, long_name):
            v = nc.createVariable(name, 'i2', ('y', 'x'),
                                  zlib=True, complevel=4, fill_value=INT16_FILL)
            v.scale_factor = 0.0001; v.add_offset = 0.0
            v[:] = np.ma.masked_invalid(np.clip(data, 0., 1.))
            v.long_name = long_name; v.units = '1'

        keys = ['0p25', '1', '2p5', '5', '10']
        for key in keys:
            if key in raw_probs:
                write_prob_i2(f'raw_p{key}mm_prob', raw_probs[key],
                              f'Raw GRAF probability > {key.replace("p",".")} mm')
            if key in gamma_probs:
                write_prob_i2(f'gamma_p{key}mm_prob', gamma_probs[key],
                              f'Gamma model probability > {key.replace("p",".")} mm')

        write_prob_i2('fraction_zero', fraction_zero,
                      'Probability of zero precipitation')
        write_prob_i2('mixture_weight', weight_params,
                      'Mixture weight for component 1')

        for vname, arr, ln in [
            ('gamma_shape1', shape1_params, 'Gamma component 1 shape (alpha1)'),
            ('gamma_scale1', scale1_params, 'Gamma component 1 scale (theta1)'),
            ('gamma_shape2', shape2_params, 'Gamma component 2 shape (alpha2)'),
            ('gamma_scale2', scale2_params, 'Gamma component 2 scale (theta2)'),
        ]:
            v = nc.createVariable(vname, 'f4', ('y','x'), zlib=True,
                                  complevel=4, least_significant_digit=3)
            v[:] = np.ma.masked_invalid(arr)
            v.long_name = ln

        mixture_mean = (weight_params * shape1_params * scale1_params +
                        (1 - weight_params) * shape2_params * scale2_params)
        v = nc.createVariable('mixture_mean', 'f4', ('y','x'), zlib=True,
                              complevel=4, least_significant_digit=3)
        v[:] = np.ma.masked_invalid(mixture_mean)
        v.long_name = 'Mixture mean precipitation given non-zero (mm)'
        v.units = 'mm'

        nc.description = ('European precipitation probabilities — '
                          '2-Component Gamma Mixture, full-domain inference')
        nc.history = 'Generated by resunet_inference_gamma_mixture_fulldomain_europe.py'
        nc.close()
    except Exception as e:
        print(f'  Error saving NetCDF: {e}')


# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python resunet_inference_gamma_mixture_fulldomain_europe.py '
              '<YYYYMMDDHH> <lead>')
        sys.exit(1)

    import time
    t0 = time.time()

    cyyyymmddhh = sys.argv[1]
    clead       = sys.argv[2]
    sigma       = init_sigma(cyyyymmddhh, clead)

    ny = 723; nx = 666    # European GRAF domain
    nchannels = 7

    config_file = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    print(f'Config: {config_file}')

    GRAFdatadir_europe, GRAFprobsdir = read_config_file(config_file, 'DIRECTORIES')

    # --- Read European GRAF precipitation ---
    (istat_GRAF, precipitation_GRAF, lats, lons, ny, nx,
     latmin, latmax, lonmin, lonmax, verif_local_time,
     lon_0, lat_0, lat_1, lat_2) = \
        GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe)

    # --- Download GFS column-average RH for European domain ---
    istat_GFS, gfs_rh = read_gfs_data_europe(cyyyymmddhh, clead, lats, lons)

    if istat_GRAF == 0 and istat_GFS == 0:

        raw_probs = calc_raw_probabilities(precipitation_GRAF, sigma)

        if ENVIRONMENT == 'aws':
            terrain_file = f'{AWS_BASE_PATH}/terrain/GRAF_Europe_terrain_info.nc'
        else:
            terrain_file = 'GRAF_Europe_terrain_info.nc'

        print(f'Terrain file: {terrain_file}')
        terrain, t_diff, dt_dlon, dt_dlat = read_terrain_characteristics(terrain_file)

        model, norm_stats, climatology, power_transform = \
            read_pytorch(cyyyymmddhh, clead)

        if model and climatology:
            model = model.float()
            t_inf = time.time()

            Xpredict_tensor, _ = generate_features(
                nchannels, cyyyymmddhh, clead,
                ny, nx, precipitation_GRAF,
                terrain, t_diff, dt_dlon, dt_dlat,
                verif_local_time, gfs_rh,
                norm_stats, power_transform)

            (gamma_probs, fraction_zero, weight_params,
             shape1_params, scale1_params,
             shape2_params, scale2_params) = \
                calc_gamma_probabilities_fulldomain(
                    model, Xpredict_tensor, ny, nx,
                    climatology['shape_min'], climatology['scale_min'])

            print(f'\nInference time: {time.time()-t_inf:.1f} s')

            os.makedirs(GRAFprobsdir, exist_ok=True)
            nc_out = (GRAFprobsdir + cyyyymmddhh + '_' + clead
                      + '_probs_europe_gamma_mixture.nc')
            write_probabilities_to_netcdf(
                nc_out, lats, lons, raw_probs, gamma_probs,
                fraction_zero, weight_params,
                shape1_params, scale1_params,
                shape2_params, scale2_params)

            print(f'\nDone. Output: {nc_out}')
            print(f'Total time: {time.time()-t0:.1f} s')
        else:
            print('Model load failed.')
    else:
        if istat_GRAF != 0: print('European GRAF data not found.')
        if istat_GFS  != 0: print('GFS download failed.')
