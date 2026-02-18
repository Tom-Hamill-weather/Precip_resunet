"""Diagnose quantile predictions in dry vs wet regions."""
import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset
import sys

# Read GRAF data
graf_file = '../resnet_data/GRAF/20251204/12/grid.hdo-graf_conus.20251205T120000Z.20251204T120000Z.PT24H.CONUS@4km.APCP.SFC.grb2'
import pygrib
grbs = pygrib.open(graf_file)
grb = grbs[1]
graf_precip = grb.values
lats, lons = grb.latlons()
grbs.close()

# Find dry and wet pixels
dry_mask = graf_precip < 0.01  # Essentially zero
wet_mask = graf_precip > 5.0    # Significant precip

# Sample pixels
dry_pixels = np.where(dry_mask)
wet_pixels = np.where(wet_mask)

if len(dry_pixels[0]) == 0:
    print("No dry pixels found!")
    sys.exit(1)

# Sample one dry pixel from interior (avoid edges)
interior_dry = [(j, i) for j, i in zip(dry_pixels[0], dry_pixels[1])
                if 400 < j < 900 and 500 < i < 1000]
if interior_dry:
    dry_j, dry_i = interior_dry[len(interior_dry)//2]
else:
    dry_j, dry_i = dry_pixels[0][0], dry_pixels[1][0]

# Sample one wet pixel
wet_j, wet_i = wet_pixels[0][0], wet_pixels[1][0]

print(f"DRY PIXEL: [{dry_j}, {dry_i}], GRAF = {graf_precip[dry_j, dry_i]:.3f} mm")
print(f"WET PIXEL: [{wet_j}, {wet_i}], GRAF = {graf_precip[wet_j, wet_i]:.3f} mm")
print()

# Load model and predict for these specific pixels
sys.path.insert(0, '/Users/tom.hamill@weather.com/python/resnet')
from resunet_inference_quantiler import (read_pytorch, read_terrain_characteristics,
                                         read_gfs_data, enforce_monotonicity,
                                         init_sigma, QUANTILE_LEVELS, NUM_QUANTILES)

# Read terrain
terrain, t_diff, dt_dlon, dt_dlat = read_terrain_characteristics('GRAF_CONUS_terrain_info.nc')

# Read GFS
istat_gfs, gfs_rh = read_gfs_data('2025120412', '24', '../resnet_data/gfs', lats, lons)

# Load model
model, norm_stats = read_pytorch('2025120412', '24')
model.eval()

# Build features for dry pixel
def make_features(j, i):
    features = np.zeros(7, dtype=np.float32)
    mins = norm_stats['min']
    maxes = norm_stats['max']

    graf = graf_precip[j, i]
    tdiff = t_diff[j, i]
    rh = gfs_rh[j, i]
    dlon = dt_dlon[j, i]
    dlat = dt_dlat[j, i]

    features[0] = (graf - mins[0]) / (maxes[0] - mins[0])
    features[1] = (tdiff - mins[1]) / (maxes[1] - mins[1])
    features[2] = (rh - mins[2]) / (maxes[2] - mins[2])
    features[3] = (graf * tdiff - mins[3]) / (maxes[3] - mins[3])
    features[4] = (graf * rh - mins[4]) / (maxes[4] - mins[4])
    features[5] = (dlon - mins[5]) / (maxes[5] - mins[5])
    features[6] = (dlat - mins[6]) / (maxes[6] - mins[6])

    return features

dry_features = make_features(dry_j, dry_i)
wet_features = make_features(wet_j, wet_i)

print("DRY PIXEL FEATURES (normalized [0,1]):")
print(f"  GRAF: {dry_features[0]:.6f}, RH: {dry_features[2]:.6f}")
print(f"  GRAF×RH: {dry_features[4]:.6f}")
print()

print("WET PIXEL FEATURES (normalized [0,1]):")
print(f"  GRAF: {wet_features[0]:.6f}, RH: {wet_features[2]:.6f}")
print(f"  GRAF×RH: {wet_features[4]:.6f}")
print()

# Run model
with torch.no_grad():
    dry_input = torch.from_numpy(dry_features).float().view(1, 7, 1, 1).to(next(model.parameters()).device)
    dry_raw = model(dry_input)
    dry_quantiles = enforce_monotonicity(dry_raw).cpu().numpy()[0, :, 0, 0]

    wet_input = torch.from_numpy(wet_features).float().view(1, 7, 1, 1).to(next(model.parameters()).device)
    wet_raw = model(wet_input)
    wet_quantiles = enforce_monotonicity(wet_raw).cpu().numpy()[0, :, 0, 0]

print("DRY PIXEL PREDICTED QUANTILES:")
for i in range(NUM_QUANTILES):
    print(f"  q_{QUANTILE_LEVELS[i]:.2f} = {dry_quantiles[i]:7.4f} mm")

# Compute P(X > 0.25mm)
threshold = 0.25
idx = np.searchsorted(dry_quantiles, threshold)
if idx == 0:
    prob_dry = 1.0
elif idx >= len(dry_quantiles):
    prob_dry = 0.0
else:
    q_low, q_high = QUANTILE_LEVELS[idx-1], QUANTILE_LEVELS[idx]
    v_low, v_high = dry_quantiles[idx-1], dry_quantiles[idx]
    if abs(v_high - v_low) < 1e-8:
        pct = q_low
    else:
        pct = q_low + (q_high - q_low) * (threshold - v_low) / (v_high - v_low)
    prob_dry = 1.0 - pct

print(f"\nDRY PIXEL P(X > 0.25mm) = {prob_dry:.4f} ({100*prob_dry:.1f}%)")
print()

print("WET PIXEL PREDICTED QUANTILES:")
for i in range(NUM_QUANTILES):
    print(f"  q_{QUANTILE_LEVELS[i]:.2f} = {wet_quantiles[i]:7.4f} mm")
