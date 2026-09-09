"""resunet_inference_gamma_mixture_season.py

python resunet_inference_gamma_mixture_season.py cyyyymmddhh clead

Full-domain inference using a season-pooled, FiLM lead-pooled ResUNet
checkpoint (trained by pytorch_train_resunet_gamma_mixture_season.py)
instead of resunet_inference_gamma_mixture_optimized.py's one-model-per-lead
checkpoints. One model per season (DJF/MAM/JJA/SON) now covers every lead
3-72h, selected by the IC date's calendar month; the conditioning vector
[sin_doy, cos_doy, sin_solar_hour, cos_solar_hour, lead/lead_max] is built
once per call (solar-hour term varies by patch center longitude) and passed
to the model alongside each patch batch.

Reuses feature generation, patch-blend (Manhattan) weighting, raw-GRAF
probability computation, and netCDF output unchanged from
resunet_inference_gamma_mixture_optimized.py -- only checkpoint loading and
the model call (now model(x, cond) instead of model(x)) are new.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.distributions import Gamma

from graf_season_index import SEASON_MONTHS, make_cond
from resunet_film import AttnResUNetFiLM
from resunet_inference_gamma_mixture_optimized import (
    AWS_BASE_PATH, BATCH_SIZE, DEVICE, ENVIRONMENT, GFS_DATA_DIR, TRAIN_DIR,
    USE_AMP, GRAF_precip_read, calc_raw_probabilities, define_manhattan,
    generate_features, init_sigma, read_config_file,
    read_terrain_characteristics, write_probabilities_to_netcdf, read_gfs_data,
)

_MONTH_TO_SEASON = {m: season for season, months in SEASON_MONTHS.items() for m in months}
_CH_ORDER = ['graf', 'terrain_diff', 'gfs_r', 'terdiff_graf', 'graf_rh', 'dlon', 'dlat']


def _bounds_to_norm_stats(bounds):
    """Convert the checkpoint's {name: (min, max)} dict to generate_features'
    {'min': [...], 'max': [...]} channel-index-ordered contract."""
    return {'min': [bounds[k][0] for k in _CH_ORDER],
            'max': [bounds[k][1] for k in _CH_ORDER]}


def read_pytorch_season(cyyyymmddhh):
    """Load the season checkpoint matching this IC date's calendar month."""
    month = int(cyyyymmddhh[4:6])
    season = _MONTH_TO_SEASON[month]
    ckpt_path = os.path.join(TRAIN_DIR, f'resunet_gamma_mixture_season_{season}_best.pth')

    if not os.path.exists(ckpt_path):
        print(f'   No season checkpoint found: {ckpt_path}')
        return None, None, None, None, None

    print(f'   Season: {season}  Loading: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    cond_dim = checkpoint.get('cond_dim', 5)
    model = AttnResUNetFiLM(in_channels=7, num_outputs=6, cond_dim=cond_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    norm_stats = _bounds_to_norm_stats(checkpoint['normalization_bounds'])
    climatology = checkpoint.get('climatology')
    power_transform = checkpoint.get('power_transform', 1.0)
    lead_max = checkpoint.get('lead_max', 72)

    return model, norm_stats, climatology, power_transform, lead_max


def calc_gamma_probabilities_season(model, Xpredict_tensor, manhattan_tensor,
                                    N, ny, nx, shape_min, scale_min,
                                    day, cycle, lead, lead_max, lons, batch_size=32):
    """Same batched patch-tiled inference as calc_gamma_probabilities_optimized
    (resunet_inference_gamma_mixture_optimized.py:509-742), except the model
    call takes a per-patch FiLM conditioning vector (cond varies across
    patches only through each patch's center longitude, via the
    solar-hour term)."""
    nchannels = Xpredict_tensor.shape[1]

    fraction_zero_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    weight_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    shape1_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    scale1_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    shape2_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    scale2_accum = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)
    sumweights_all = torch.zeros((ny, nx), dtype=torch.float32, device=DEVICE)

    jcenter1 = range(N // 2, ny - N // 2 + 1, N // 2)
    icenter1 = range(N // 2, nx - N // 2 + 1, N // 2)
    jcenter2 = range(N // 2 + N // 4, ny - 3 * N // 4, N // 2)
    icenter2 = range(N // 2 + N // 4, nx - 3 * N // 4, N // 2)

    patch_coords = ([(j, i) for j in jcenter1 for i in icenter1] +
                    [(j, i) for j in jcenter2 for i in icenter2])

    num_patches = len(patch_coords)
    print(f'Processing {num_patches} patches in batches of {batch_size} (season+FiLM model)...')

    for batch_start in range(0, num_patches, batch_size):
        batch_end = min(batch_start + batch_size, num_patches)
        batch_coords = patch_coords[batch_start:batch_end]
        current_batch_size = len(batch_coords)

        batch_tensor = torch.empty(current_batch_size, nchannels, N, N,
                                   device=DEVICE, dtype=torch.float32)
        cond_batch = torch.empty(current_batch_size, 5, device=DEVICE, dtype=torch.float32)
        batch_metadata = []

        for idx, (j, i) in enumerate(batch_coords):
            jmin, jmax = j - N // 2, j + N // 2
            imin, imax = i - N // 2, i + N // 2

            Xpatch = Xpredict_tensor[0, :, jmin:jmax, imin:imax]
            h_curr, w_curr = Xpatch.shape[1], Xpatch.shape[2]
            pad_h, pad_w = N - h_curr, N - w_curr

            if pad_h > 0 or pad_w > 0:
                batch_tensor[idx] = F.pad(Xpatch.unsqueeze(0),
                                          (0, pad_w, 0, pad_h), mode='replicate')[0]
            else:
                batch_tensor[idx] = Xpatch

            center_lon = float(lons[j, i])
            cond_batch[idx] = torch.from_numpy(
                make_cond(day, cycle, lead, center_lon, lead_max=lead_max))
            batch_metadata.append((j, i, jmin, jmax, imin, imax, h_curr, w_curr, pad_h, pad_w))

        with torch.no_grad():
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(batch_tensor, cond_batch)
            else:
                logits = model(batch_tensor, cond_batch)

            logits = logits.float()
            logits = torch.clamp(logits, min=-10, max=10)

            p0 = torch.sigmoid(logits[:, 0, :, :])
            w = torch.sigmoid(logits[:, 1, :, :])
            alpha1 = shape_min + F.softplus(logits[:, 2, :, :])
            theta1 = scale_min + F.softplus(logits[:, 3, :, :])
            shape2_offset = F.softplus(logits[:, 4, :, :])
            alpha2 = alpha1 + shape2_offset + 0.5
            theta2 = scale_min + F.softplus(logits[:, 5, :, :])

        for idx, (j, i, jmin, jmax, imin, imax, h_curr, w_curr, pad_h, pad_w) in \
                enumerate(batch_metadata):
            p0_patch, w_patch = p0[idx], w[idx]
            alpha1_patch, theta1_patch = alpha1[idx], theta1[idx]
            alpha2_patch, theta2_patch = alpha2[idx], theta2[idx]

            if pad_h > 0 or pad_w > 0:
                p0_patch     = p0_patch[:h_curr, :w_curr]
                w_patch      = w_patch[:h_curr, :w_curr]
                alpha1_patch = alpha1_patch[:h_curr, :w_curr]
                theta1_patch = theta1_patch[:h_curr, :w_curr]
                alpha2_patch = alpha2_patch[:h_curr, :w_curr]
                theta2_patch = theta2_patch[:h_curr, :w_curr]
                mh_weight = manhattan_tensor[:h_curr, :w_curr]
            else:
                mh_weight = manhattan_tensor

            fraction_zero_accum[jmin:jmax, imin:imax] += p0_patch * mh_weight
            weight_accum[jmin:jmax, imin:imax]        += w_patch * mh_weight
            shape1_accum[jmin:jmax, imin:imax]        += alpha1_patch * mh_weight
            scale1_accum[jmin:jmax, imin:imax]        += theta1_patch * mh_weight
            shape2_accum[jmin:jmax, imin:imax]        += alpha2_patch * mh_weight
            scale2_accum[jmin:jmax, imin:imax]        += theta2_patch * mh_weight
            sumweights_all[jmin:jmax, imin:imax]      += mh_weight

        if (batch_start // batch_size) % 10 == 0:
            print(f'  Processed {batch_end}/{num_patches} patches...')

    sumweights_safe = torch.clamp(sumweights_all, min=1e-9)
    valid_mask = sumweights_all > 1e-9

    def scalar(v):
        return torch.tensor(v, device=DEVICE, dtype=torch.float32)

    fraction_zero = torch.where(valid_mask, fraction_zero_accum / sumweights_safe, scalar(1.0))
    weight_params = torch.where(valid_mask, weight_accum / sumweights_safe, scalar(0.5))
    shape1_params = torch.where(valid_mask, shape1_accum / sumweights_safe, scalar(1.0))
    scale1_params = torch.where(valid_mask, scale1_accum / sumweights_safe, scalar(1.0))
    shape2_params = torch.where(valid_mask, shape2_accum / sumweights_safe, scalar(2.0))
    scale2_params = torch.where(valid_mask, scale2_accum / sumweights_safe, scalar(1.0))

    if torch.stack([fraction_zero, weight_params, shape1_params,
                    scale1_params, shape2_params, scale2_params]).isnan().any():
        print('  WARNING: NaN detected in parameters after normalization')
        fraction_zero = torch.nan_to_num(fraction_zero, nan=0.5)
        weight_params = torch.nan_to_num(weight_params, nan=0.5)
        shape1_params = torch.nan_to_num(shape1_params, nan=1.0)
        scale1_params = torch.nan_to_num(scale1_params, nan=1.0)
        shape2_params = torch.nan_to_num(shape2_params, nan=2.0)
        scale2_params = torch.nan_to_num(scale2_params, nan=1.0)

    print('Computing probabilities from Gamma mixture (GPU-accelerated)...')
    gamma_probs = {}
    thresholds = {'0p25': 0.25, '1': 1.0, '2p5': 2.5, '5': 5.0, '10': 10.0}

    alpha1_safe = torch.clamp(shape1_params, min=0.1)
    theta1_safe = torch.clamp(scale1_params, min=0.01)
    alpha2_safe = torch.clamp(shape2_params, min=0.1)
    theta2_safe = torch.clamp(scale2_params, min=0.01)
    w_safe = torch.clamp(weight_params, min=0.0, max=1.0)

    rate1 = 1.0 / theta1_safe
    rate2 = 1.0 / theta2_safe
    gamma_dist1 = Gamma(concentration=alpha1_safe, rate=rate1, validate_args=False)
    gamma_dist2 = Gamma(concentration=alpha2_safe, rate=rate2, validate_args=False)

    for key, threshold in thresholds.items():
        threshold_tensor = torch.tensor(threshold, device=DEVICE, dtype=torch.float32)
        cdf1 = torch.clamp(gamma_dist1.cdf(threshold_tensor), 0.0, 1.0)
        cdf2 = torch.clamp(gamma_dist2.cdf(threshold_tensor), 0.0, 1.0)
        mixture_sf = w_safe * (1.0 - cdf1) + (1.0 - w_safe) * (1.0 - cdf2)
        prob_exceed = torch.clamp(torch.nan_to_num((1.0 - fraction_zero) * mixture_sf, nan=0.0), 0.0, 1.0)
        gamma_probs[key] = prob_exceed

    print('Transferring results to CPU...')
    gamma_probs_cpu = {key: v.cpu().numpy() for key, v in gamma_probs.items()}
    return (gamma_probs_cpu, fraction_zero.cpu().numpy(), weight_params.cpu().numpy(),
            shape1_params.cpu().numpy(), scale1_params.cpu().numpy(),
            shape2_params.cpu().numpy(), scale2_params.cpu().numpy())


def main():
    if len(sys.argv) < 3:
        print('Usage: python resunet_inference_gamma_mixture_season.py <YYYYMMDDHH> <lead>')
        sys.exit(1)

    start_time = time.time()
    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    sigma = init_sigma(cyyyymmddhh, clead)

    N = 96
    ny, nx = 1308, 1524
    nchannels = 7

    config_file_name = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus_laptop = \
        read_config_file(config_file_name, 'DIRECTORIES')
    manhattan = define_manhattan(N)

    (istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax,
     lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2) = \
        GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)

    istat_GFS, gfs_rh = read_gfs_data(cyyyymmddhh, clead, GFS_DATA_DIR, lats, lons)

    if istat_GRAF != 0 or istat_GFS != 0:
        if istat_GRAF != 0:
            print('GRAF forecast data not found.')
        if istat_GFS != 0:
            print('GFS data not found.')
        return

    raw_probs = calc_raw_probabilities(precipitation_GRAF, sigma)

    terrain_file = (f'{AWS_BASE_PATH}/terrain/GRAF_CONUS_terrain_info.nc'
                    if ENVIRONMENT == 'aws' else 'GRAF_CONUS_terrain_info.nc')
    terrain, t_diff, dt_dlon, dt_dlat = read_terrain_characteristics(terrain_file)

    model, norm_stats, climatology, power_transform, lead_max = read_pytorch_season(cyyyymmddhh)
    if not model or not climatology:
        print('Season model load failed.')
        return

    model = model.float()
    inference_start = time.time()

    Xpredict_tensor, _ = generate_features(nchannels, cyyyymmddhh, clead, ny, nx,
                                           precipitation_GRAF, terrain, t_diff, dt_dlon,
                                           dt_dlat, verif_local_time, gfs_rh,
                                           norm_stats, power_transform=power_transform)

    shape_min, scale_min = climatology['shape_min'], climatology['scale_min']
    day, cycle, lead = int(cyyyymmddhh[:8]), int(cyyyymmddhh[8:10]), int(clead)

    (gamma_probs, fraction_zero, weight_params, shape1_params,
     scale1_params, shape2_params, scale2_params) = calc_gamma_probabilities_season(
        model, Xpredict_tensor, manhattan, N, ny, nx, shape_min, scale_min,
        day, cycle, lead, lead_max, lons, BATCH_SIZE)

    inference_time = time.time() - inference_start
    print(f'\nInference time: {inference_time:.2f} seconds')

    probs_out_dir = GRAFprobsdir_conus_laptop
    os.makedirs(probs_out_dir, exist_ok=True)
    nc_filename = probs_out_dir + cyyyymmddhh + '_' + clead + '_probs_gamma_mixture_season.nc'
    write_probabilities_to_netcdf(nc_filename, lats, lons, raw_probs, gamma_probs,
                                  fraction_zero, weight_params, shape1_params,
                                  scale1_params, shape2_params, scale2_params)

    total_time = time.time() - start_time
    print(f'\nInference complete! Output saved to: {nc_filename}')
    print(f'Total time: {total_time:.2f}s  Inference-only: {inference_time:.2f}s')


if __name__ == '__main__':
    main()
