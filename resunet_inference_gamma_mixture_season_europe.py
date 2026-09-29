"""resunet_inference_gamma_mixture_season_europe.py

python resunet_inference_gamma_mixture_season_europe.py cyyyymmddhh clead

European-domain counterpart to resunet_inference_gamma_mixture_season.py:
runs the season-pooled, FiLM lead-pooled, per-pixel-climatology gamma
mixture checkpoint (resunet_gamma_mixture_season_{DJF,MAM,JJA,SON}_best.pth,
in_channels=10, cond_dim=3 -- see that script's docstring for the
per-pixel-solar-hour and precip_climo channel history) over the European
GRAF domain via a single whole-domain forward pass (fully convolutional
model, no patch tiling needed).

Differences from the CONUS season script, mirroring
resunet_inference_gamma_mixture_optimized_europe.py:
  - Target grid: 723 x 666 (European LCC, ~4 km)
  - GRAF file naming: hdo-graf_europe / hdo-graflr_europe, EUROPE@4km.APCP
  - Terrain: GRAF_Europe_terrain_info.nc
  - GFS RH: downloaded on the fly from s3://noaa-gfs-bdp-pds
  - precip_climo: graf_precip_climo_europe.fulldomain_climo (WorldClim-only
    blend on the Europe grid -- see build_precip_climo_europe.py; the CONUS
    PRISM+WorldClim+ERA5 blend doesn't cover this domain at all)
  - Output: <probs_dir>/<cyyyymmddhh>_<lead>_probs_europe_gamma_mixture_season.nc
    (distinct suffix from the non-season Europe script's
    *_probs_europe_gamma_mixture.nc, so old transfer-learning-era outputs/
    plots for the same IC/lead aren't silently overwritten)

Transfer learning: this is the CONUS-trained season+FiLM model applied to
Europe; no European training data is involved.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma

from graf_precip_climo_europe import fulldomain_climo
from graf_season_index import SEASON_MONTHS, make_cond
from resunet_film import AttnResUNetFiLM
from pytorch_train_resunet_biascorrect_season import local_solar_hour_sincos
from resunet_inference_gamma_mixture_optimized_europe import (
    AWS_BASE_PATH, DEVICE, ENVIRONMENT, TRAIN_DIR, USE_AMP,
    GRAF_precip_read_europe, read_gfs_data_europe, calc_raw_probabilities,
    init_sigma, read_terrain_characteristics, read_config_file,
    write_probabilities_to_netcdf,
)

_MONTH_TO_SEASON = {m: season for season, months in SEASON_MONTHS.items() for m in months}
_DEFAULT_CH_ORDER = ['graf', 'terrain_diff', 'gfs_r', 'terdiff_graf', 'graf_rh',
                     'dlon', 'dlat', 'sin_sh', 'cos_sh', 'precip_climo']
DIVISOR = 16  # 2**4 downsampling stages


def read_pytorch_season(cyyyymmddhh):
    """Load the season checkpoint matching this IC date's calendar month
    (identical logic/checkpoint to the CONUS script -- one shared model)."""
    month = int(cyyyymmddhh[4:6])
    season = _MONTH_TO_SEASON[month]
    ckpt_path = os.path.join(TRAIN_DIR, f'resunet_gamma_mixture_season_{season}_best.pth')

    if not os.path.exists(ckpt_path):
        print(f'   No season checkpoint found: {ckpt_path}')
        return None, None, None, None, None, None, None

    print(f'   Season: {season}  Loading: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    in_channels = checkpoint.get('in_channels', len(_DEFAULT_CH_ORDER))
    cond_dim = checkpoint.get('cond_dim', 3)
    model = AttnResUNetFiLM(in_channels=in_channels, num_outputs=6, cond_dim=cond_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    bounds = checkpoint['normalization_bounds']
    ch_order = checkpoint.get('channel_order', _DEFAULT_CH_ORDER)
    climatology = checkpoint.get('climatology')
    power_transform = checkpoint.get('power_transform', 1.0)
    lead_max = checkpoint.get('lead_max', 72)

    return model, bounds, ch_order, climatology, power_transform, lead_max, season


def generate_features_fulldomain(precipitation_GRAF, t_diff, dt_dlon, dt_dlat, gfs_rh,
                                  sin_sh, cos_sh, climo, bounds, ch_order, power_transform=1.0):
    """Identical channel stack/normalization to the CONUS season script."""
    def normalize(data, key):
        lo, hi = bounds[key]
        denom = hi - lo if (hi - lo) > 1e-6 else 1e-6
        return (data - lo) / denom

    if power_transform != 1.0:
        precipitation_GRAF = np.power(np.clip(precipitation_GRAF, 0.0, None), power_transform)

    terdiff_graf = precipitation_GRAF * t_diff
    graf_rh = precipitation_GRAF * gfs_rh

    fields = {
        'graf': precipitation_GRAF, 'terrain_diff': t_diff, 'gfs_r': gfs_rh,
        'terdiff_graf': terdiff_graf, 'graf_rh': graf_rh,
        'dlon': dt_dlon, 'dlat': dt_dlat, 'sin_sh': sin_sh, 'cos_sh': cos_sh,
        'precip_climo': climo,
    }
    channels = np.stack([normalize(fields[k], k) for k in ch_order], axis=0).astype(np.float32)
    return torch.from_numpy(channels[np.newaxis]).to(DEVICE)


def run_gamma_mixture_season_fulldomain(model, Xpredict_tensor, cond, ny, nx):
    """Whole-domain single forward pass; pad to a multiple of 16, crop back."""
    _, _, h, w = Xpredict_tensor.shape
    pad_h = (DIVISOR - h % DIVISOR) % DIVISOR
    pad_w = (DIVISOR - w % DIVISOR) % DIVISOR

    Xpad = F.pad(Xpredict_tensor, (0, pad_w, 0, pad_h), mode='replicate') if (pad_h or pad_w) \
        else Xpredict_tensor

    print(f'Running single forward pass on {Xpad.shape[2]}x{Xpad.shape[3]} domain...')
    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(Xpad, cond)
        else:
            logits = model(Xpad, cond)
        logits = logits.float()[0]  # (6, H, W)

    return logits[:, :ny, :nx]


def calc_gamma_probabilities_fulldomain(logits, shape_min, scale_min):
    """Identical parameter transform/threshold-exceedance calc to the CONUS
    season script."""
    p0 = torch.sigmoid(logits[0])
    w = torch.sigmoid(logits[1])
    alpha1 = shape_min + F.softplus(logits[2])
    theta1 = scale_min + F.softplus(logits[3])
    shape2_offset = F.softplus(logits[4])
    alpha2 = alpha1 + shape2_offset + 0.5
    theta2 = scale_min + F.softplus(logits[5])

    alpha1_safe = torch.clamp(alpha1, min=0.1)
    theta1_safe = torch.clamp(theta1, min=0.01)
    alpha2_safe = torch.clamp(alpha2, min=0.1)
    theta2_safe = torch.clamp(theta2, min=0.01)
    w_safe = torch.clamp(w, min=0.0, max=1.0)

    gamma_dist1 = Gamma(concentration=alpha1_safe, rate=1.0 / theta1_safe, validate_args=False)
    gamma_dist2 = Gamma(concentration=alpha2_safe, rate=1.0 / theta2_safe, validate_args=False)

    print('Computing probabilities from Gamma mixture (GPU-accelerated)...')
    gamma_probs = {}
    for key, threshold in {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}.items():
        threshold_tensor = torch.tensor(threshold, device=DEVICE, dtype=torch.float32)
        cdf1 = torch.clamp(gamma_dist1.cdf(threshold_tensor), 0.0, 1.0)
        cdf2 = torch.clamp(gamma_dist2.cdf(threshold_tensor), 0.0, 1.0)
        mixture_sf = w_safe * (1.0 - cdf1) + (1.0 - w_safe) * (1.0 - cdf2)
        prob_exceed = torch.clamp(torch.nan_to_num((1.0 - p0) * mixture_sf, nan=0.0), 0.0, 1.0)
        gamma_probs[key] = prob_exceed.cpu().numpy()

    return (gamma_probs, p0.cpu().numpy(), w.cpu().numpy(),
            alpha1.cpu().numpy(), theta1.cpu().numpy(),
            alpha2.cpu().numpy(), theta2.cpu().numpy())


def main():
    if len(sys.argv) < 3:
        print('Usage: python resunet_inference_gamma_mixture_season_europe.py <YYYYMMDDHH> <lead>')
        sys.exit(1)

    start_time = time.time()
    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    sigma = init_sigma(cyyyymmddhh, clead)

    config_file_name = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    GRAFdatadir_europe, GRAFprobsdir = read_config_file(config_file_name, 'DIRECTORIES')

    (istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax,
     lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2) = \
        GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe)

    istat_GFS, gfs_rh = read_gfs_data_europe(cyyyymmddhh, clead, lats, lons)

    if istat_GRAF != 0 or istat_GFS != 0:
        if istat_GRAF != 0:
            print('European GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
        return

    raw_probs = calc_raw_probabilities(precipitation_GRAF, sigma)

    terrain_file = (f'{AWS_BASE_PATH}/terrain/GRAF_Europe_terrain_info.nc'
                    if ENVIRONMENT == 'aws' else 'GRAF_Europe_terrain_info.nc')
    terrain, t_diff, dt_dlon, dt_dlat = read_terrain_characteristics(terrain_file)

    model, bounds, ch_order, climatology, power_transform, lead_max, season = \
        read_pytorch_season(cyyyymmddhh)
    if not model or not climatology:
        print('Season model load failed.')
        return

    inference_start = time.time()

    day, cycle, lead = int(cyyyymmddhh[:8]), int(cyyyymmddhh[8:10]), int(clead)
    sin_sh, cos_sh = local_solar_hour_sincos(cycle, lead, lons)
    climo = fulldomain_climo(int(cyyyymmddhh[4:6]))

    Xpredict_tensor = generate_features_fulldomain(
        precipitation_GRAF, t_diff, dt_dlon, dt_dlat, gfs_rh, sin_sh, cos_sh, climo,
        bounds, ch_order, power_transform=power_transform)

    # cond is [sin_doy, cos_doy, lead_norm] only - global, no lon dependence
    # (solar-hour is handled by the per-pixel input channels above).
    cond_full = make_cond(day, cycle, lead, 0.0, lead_max=lead_max)
    cond = torch.from_numpy(cond_full[[0, 1, 4]][np.newaxis]).float().to(DEVICE)

    logits = run_gamma_mixture_season_fulldomain(model, Xpredict_tensor, cond, ny, nx)

    shape_min, scale_min = climatology['shape_min'], climatology['scale_min']
    (gamma_probs, fraction_zero, weight_params, shape1_params,
     scale1_params, shape2_params, scale2_params) = calc_gamma_probabilities_fulldomain(
        logits, shape_min, scale_min)

    inference_time = time.time() - inference_start
    print(f'\nInference time: {inference_time:.2f} seconds')

    os.makedirs(GRAFprobsdir, exist_ok=True)
    nc_filename = (GRAFprobsdir + cyyyymmddhh + '_' + clead
                  + '_probs_europe_gamma_mixture_season.nc')
    write_probabilities_to_netcdf(nc_filename, lats, lons, raw_probs, gamma_probs,
                                  fraction_zero, weight_params, shape1_params,
                                  scale1_params, shape2_params, scale2_params)

    total_time = time.time() - start_time
    print(f'\nInference complete! Output saved to: {nc_filename}')
    print(f'Total time: {total_time:.2f}s  Inference-only: {inference_time:.2f}s')


if __name__ == '__main__':
    main()
