"""
python resunet_inference_gamma_mixture_optimized_europe.py cyyyymmddhh clead
e.g.,
python resunet_inference_gamma_mixture_optimized_europe.py 2025120412 12

European-domain version of resunet_inference_gamma_mixture_optimized.py.

Key differences from the CONUS version:
  - Target grid: 723 x 666 (European LCC, ~4 km)
  - GRAF file naming: hdo-graf_europe / hdo-graflr_europe, EUROPE@4km.APCP
  - Terrain: GRAF_Europe_terrain_info.nc
  - GFS RH: downloaded on the fly from s3://noaa-gfs-bdp-pds via HTTP
    byte-range requests.  The field is "RH: entire atmosphere (considered
    as a single layer)" — the column-average relative humidity — matching
    exactly what the CONUS gfs_subset_*.nc files store.
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

# ---------------------------------------------------------------
# Device
# ---------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_AMP = False
    print(f"Running on: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    USE_AMP = False
    print(f"Running on: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False
    print(f"Running on: {DEVICE}")

BATCH_SIZE = 32 if torch.cuda.is_available() else 16

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

    The field is "RH: entire atmosphere (considered as a single layer)"
    (GFS GRIB typeOfLevel=atmosphereSingleLayer, paramId=157) — the same
    field stored as 'r' in the CONUS gfs_subset_*.nc training files.

    Returns istat, rh_on_euro_grid (2-D, %).
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

    # Column-average (whole-atmosphere) RH
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

    # Write to temp file and read with pygrib
    with tempfile.NamedTemporaryFile(suffix='.grb2', delete=False) as tmp:
        tmp.write(rh_bytes)
        tmpname = tmp.name

    try:
        f = pygrib.open(tmpname)
        grb_rh = f.read(1)[0]
        lats_gfs, lons_gfs = grb_rh.latlons()   # (721, 1440) global
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

    # Interpolate global 0.25-deg field to European GRAF grid.
    # lats_gfs rows run 90 -> -90 (decreasing); flip to ascending.
    lats_1d = lats_gfs[:, 0]     # (721,) 90 -> -90
    lons_1d = lons_gfs[0, :]     # (1440,) 0 -> 359.75

    lats_asc = lats_1d[::-1]
    rh_asc   = rh_global[::-1, :]

    # The European GRAF domain straddles 0° longitude (-20° to +29°).
    # Wrapping negative lons to 0-360 would make points just west of 0°
    # appear at ~360° while points just east stay near 0°, creating a
    # 360°-wide apparent gap at the meridian that breaks bilinear
    # interpolation.  Fix: roll the GFS grid to -180..179.75° so the
    # European range lies entirely within a contiguous axis with no wrap.
    roll_by   = lons_1d.size // 2                    # 720 steps = 180°
    lons_rolled = np.concatenate(
        [lons_1d[roll_by:] - 360.0, lons_1d[:roll_by]])   # -180 -> 179.75
    rh_rolled   = np.concatenate(
        [rh_asc[:, roll_by:], rh_asc[:, :roll_by]], axis=1)

    interp = RegularGridInterpolator(
        (lats_asc, lons_rolled),
        rh_rolled,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # euro_lons are already in -180..180 convention; use directly.
    ny, nx = euro_lats.shape
    pts = np.column_stack([euro_lats.ravel(), euro_lons.ravel()])
    rh_euro = interp(pts).reshape(ny, nx)

    print(f'  GFS RH (Europe): min={rh_euro.min():.1f} max={rh_euro.max():.1f} %')
    return 0, rh_euro

# ---------------------------------------------------------------
# GRAF Europe file reading
# ---------------------------------------------------------------

def _euro_prefix_and_dir(cyyyymmddhh, GRAFdatadir_europe):
    """
    Return (input_directory, prefix) for European GRAF files.
    Naming switched from hdo-graflr_europe to hdo-graf_europe around
    late March 2024.  We try the new name first and fall back to the old.
    """
    cyyyymmdd = cyyyymmddhh[:8]
    chh       = cyyyymmddhh[8:10]
    base_dir  = GRAFdatadir_europe + cyyyymmdd + '/' + chh + '/'
    return base_dir   # prefix handled inside read function


def GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe):
    """Read European GRAF hourly precipitation from GRIB2."""
    import pygrib

    il = int(clead)
    cyyyymmdd       = cyyyymmddhh[:8]
    chh             = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst  = cyyyymmddhh_fcst[:8]
    chh_fcst        = cyyyymmddhh_fcst[8:10]

    input_dir = GRAFdatadir_europe + cyyyymmdd + '/' + chh + '/'

    # Try new naming first, then fall back to old
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


def define_manhattan(N):
    w = np.maximum(0.0, 1. - 2. * np.abs(np.arange(N) + 0.5 - N / 2) / N)
    manhattan = (0.5 * np.outer(w, w)).astype(np.float32)
    return torch.from_numpy(manhattan).to(DEVICE)


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
    print(f'  Requested lead {target_lead}h → using {nearest_lead}h weights')
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


def calc_gamma_probabilities_optimized(model, Xpredict_tensor,
                                       manhattan_tensor, N, ny, nx,
                                       shape_min, scale_min, batch_size=32):
    """Identical to CONUS version — patch-based batched inference."""
    nchannels = Xpredict_tensor.shape[1]

    fz_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    w_acc   = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    a1_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    t1_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    a2_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    t2_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    sw_acc  = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)

    jc1 = range(N//2, ny - N//2 + 1, N//2)
    ic1 = range(N//2, nx - N//2 + 1, N//2)
    jc2 = range(N//2 + N//4, ny - 3*N//4, N//2)
    ic2 = range(N//2 + N//4, nx - 3*N//4, N//2)
    coords = ([(j, i) for j in jc1 for i in ic1] +
              [(j, i) for j in jc2 for i in ic2])
    npatches = len(coords)
    print(f'Processing {npatches} patches in batches of {batch_size}...')

    for b0 in range(0, npatches, batch_size):
        batch = coords[b0: b0 + batch_size]
        bsz   = len(batch)
        buf   = torch.empty(bsz, nchannels, N, N,
                            device=DEVICE, dtype=torch.float32)
        meta  = []

        for idx, (j, i) in enumerate(batch):
            jmin, jmax = j - N//2, j + N//2
            imin, imax = i - N//2, i + N//2
            patch = Xpredict_tensor[0, :, jmin:jmax, imin:imax]
            h, w  = patch.shape[1], patch.shape[2]
            ph, pw = N - h, N - w
            buf[idx] = (F.pad(patch.unsqueeze(0), (0, pw, 0, ph),
                              mode='replicate')[0]
                        if ph > 0 or pw > 0 else patch)
            meta.append((j, i, jmin, jmax, imin, imax, h, w, ph, pw))

        with torch.no_grad():
            logits = model(buf).float()
            logits = torch.clamp(logits, -10, 10)
            p0  = torch.sigmoid(logits[:, 0])
            w   = torch.sigmoid(logits[:, 1])
            a1  = shape_min + F.softplus(logits[:, 2])
            t1  = scale_min + F.softplus(logits[:, 3])
            a2  = a1 + F.softplus(logits[:, 4]) + 0.5
            t2  = scale_min + F.softplus(logits[:, 5])

        for idx, (j, i, jmin, jmax, imin, imax, h, w_curr, ph, pw) in \
                enumerate(meta):
            p0p = p0[idx];  wp  = w[idx]
            a1p = a1[idx];  t1p = t1[idx]
            a2p = a2[idx];  t2p = t2[idx]
            if ph > 0 or pw > 0:
                p0p = p0p[:h, :w_curr]; wp  = wp[:h, :w_curr]
                a1p = a1p[:h, :w_curr]; t1p = t1p[:h, :w_curr]
                a2p = a2p[:h, :w_curr]; t2p = t2p[:h, :w_curr]
                mh  = manhattan_tensor[:h, :w_curr]
            else:
                mh = manhattan_tensor

            fz_acc[jmin:jmax, imin:imax] += p0p * mh
            w_acc [jmin:jmax, imin:imax] += wp  * mh
            a1_acc[jmin:jmax, imin:imax] += a1p * mh
            t1_acc[jmin:jmax, imin:imax] += t1p * mh
            a2_acc[jmin:jmax, imin:imax] += a2p * mh
            t2_acc[jmin:jmax, imin:imax] += t2p * mh
            sw_acc[jmin:jmax, imin:imax] += mh

        if (b0 // batch_size) % 10 == 0:
            print(f'  {min(b0+batch_size, npatches)}/{npatches} patches...')

    sw_safe = torch.clamp(sw_acc, min=1e-9)
    vm      = sw_acc > 1e-9

    def _fill(t, f): return torch.where(vm, t / sw_safe,
                                         torch.tensor(f, device=DEVICE,
                                                       dtype=torch.float32))
    fz = _fill(fz_acc, 1.0);  wp = _fill(w_acc,  0.5)
    a1 = _fill(a1_acc, 1.0);  t1 = _fill(t1_acc, 1.0)
    a2 = _fill(a2_acc, 2.0);  t2 = _fill(t2_acc, 1.0)

    a1 = torch.nan_to_num(torch.clamp(a1, min=0.1), nan=1.0)
    t1 = torch.nan_to_num(torch.clamp(t1, min=0.01), nan=1.0)
    a2 = torch.nan_to_num(torch.clamp(a2, min=0.1), nan=2.0)
    t2 = torch.nan_to_num(torch.clamp(t2, min=0.01), nan=1.0)
    wp = torch.nan_to_num(torch.clamp(wp, 0., 1.), nan=0.5)
    fz = torch.nan_to_num(fz, nan=0.5)

    print('Computing probabilities from Gamma mixture (GPU-accelerated)...')
    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}

    g1 = Gamma(concentration=a1, rate=1.0/t1, validate_args=False)
    g2 = Gamma(concentration=a2, rate=1.0/t2, validate_args=False)

    gamma_probs = {}
    for key, thr in thresholds.items():
        t_tensor = torch.tensor(thr, device=DEVICE, dtype=torch.float32)
        sf1 = torch.clamp(1.0 - g1.cdf(t_tensor), 0., 1.)
        sf2 = torch.clamp(1.0 - g2.cdf(t_tensor), 0., 1.)
        prob = torch.clamp((1.0 - fz) * (wp * sf1 + (1.0 - wp) * sf2), 0., 1.)
        gamma_probs[key] = torch.nan_to_num(prob, nan=0.0)

    print('Transferring results to CPU...')
    return (
        {k: v.cpu().numpy() for k, v in gamma_probs.items()},
        fz.cpu().numpy(), wp.cpu().numpy(),
        a1.cpu().numpy(), t1.cpu().numpy(),
        a2.cpu().numpy(), t2.cpu().numpy(),
    )


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

        keys = ['0p25', '1', '2p5', '5', '10']
        for key in keys:
            for pfx, src in (('raw_p', raw_probs), ('gamma_p', gamma_probs)):
                if key not in src: continue
                v = nc.createVariable(f'{pfx}{key}mm_prob', 'i2', ('y','x'),
                                      zlib=True, complevel=4)
                v.scale_factor = 0.0001
                v.add_offset   = 0.0
                v[:] = np.clip(src[key], 0., 1.)
                v.long_name = (('Raw GRAF' if pfx.startswith('raw') else 'Gamma model')
                               + f' probability > {key.replace("p",".")} mm')
                v.units = '1'

        for vname, arr, ln in [
            ('fraction_zero',  fraction_zero,  'Probability of zero precipitation'),
            ('mixture_weight', weight_params,   'Mixture weight for component 1'),
        ]:
            v = nc.createVariable(vname, 'i2', ('y','x'), zlib=True, complevel=4)
            v.scale_factor = 0.0001; v.add_offset = 0.0
            v[:] = np.clip(arr, 0., 1.)
            v.long_name = ln; v.units = '1'

        for vname, arr, ln in [
            ('gamma_shape1', shape1_params, 'Gamma component 1 shape (alpha1)'),
            ('gamma_scale1', scale1_params, 'Gamma component 1 scale (theta1)'),
            ('gamma_shape2', shape2_params, 'Gamma component 2 shape (alpha2)'),
            ('gamma_scale2', scale2_params, 'Gamma component 2 scale (theta2)'),
        ]:
            v = nc.createVariable(vname, 'f4', ('y','x'), zlib=True,
                                  complevel=4, least_significant_digit=3)
            v[:] = arr; v.long_name = ln

        mixture_mean = (weight_params * shape1_params * scale1_params +
                        (1 - weight_params) * shape2_params * scale2_params)
        v = nc.createVariable('mixture_mean', 'f4', ('y','x'), zlib=True,
                              complevel=4, least_significant_digit=3)
        v[:] = mixture_mean
        v.long_name = 'Mixture mean precipitation given non-zero (mm)'
        v.units = 'mm'

        nc.description = ('European precipitation probabilities — '
                          '2-Component Gamma Mixture, transfer learning from CONUS')
        nc.history = 'Generated by resunet_inference_gamma_mixture_optimized_europe.py'
        nc.close()
    except Exception as e:
        print(f'  Error saving NetCDF: {e}')


# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python resunet_inference_gamma_mixture_optimized_europe.py '
              '<YYYYMMDDHH> <lead>')
        sys.exit(1)

    import time
    t0 = time.time()

    cyyyymmddhh = sys.argv[1]
    clead       = sys.argv[2]
    sigma       = init_sigma(cyyyymmddhh, clead)

    N  = 96
    ny = 723; nx = 666    # European GRAF domain
    nchannels = 7

    config_file = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    print(f'Config: {config_file}  |  Batch size: {BATCH_SIZE}')

    GRAFdatadir_europe, GRAFprobsdir = \
        read_config_file(config_file, 'DIRECTORIES')

    manhattan = define_manhattan(N)

    # --- Read European GRAF precipitation
    (istat_GRAF, precipitation_GRAF, lats, lons, ny, nx,
     latmin, latmax, lonmin, lonmax, verif_local_time,
     lon_0, lat_0, lat_1, lat_2) = \
        GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe)

    # --- Download GFS column-average RH for European domain
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
                calc_gamma_probabilities_optimized(
                    model, Xpredict_tensor, manhattan,
                    N, ny, nx,
                    climatology['shape_min'], climatology['scale_min'],
                    BATCH_SIZE)

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
