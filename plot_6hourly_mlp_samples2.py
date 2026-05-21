
"""
plot_6hourly_mlp_samples2.py — diagnostic CDF / exceedance plots for
a specific IC date, lead time, and geographic location.

Usage:
    python plot_6hourly_mlp_samples2.py <cyyyymmddhh> <clead>

    cyyyymmddhh : initial-condition date/hour  (e.g. 2025120412)
    clead       : 6-hourly lead time in hours  (e.g. 36)

Reads the six consecutive gamma-mixture parameter files
    {probs_dir}/{cyyyymmddhh}_{lt}_probs_gamma_mixture.nc   (lt = clead-5 .. clead)
and extracts the pixel nearest to the dot location used in
make_plots_6hourly_mlp_4panel.py  (lon=-122.25, lat=44.0).

Produces a two-panel figure:
    (a) CDFs  — 6 thin coloured lines (hourly) + 1 thick black MLP line
    (b) Exceedance probability — same structure

Output: {plot_dir}/6h_sample_IC{cyyyymmddhh}_lead{clead}h.png

Tom Hamill, May 2026
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma as sp_gamma
from netCDF4 import Dataset
from configparser import ConfigParser
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match train_6hourly_mlp.py
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR    = os.path.join(SCRIPT_DIR, 'mlp_trainings')

# Dot location from make_plots_6hourly_mlp_4panel.py
DOT_LON = -122.25
DOT_LAT =   44.0

SHAPE_MIN    = 0.1
SCALE_MIN    = 0.01
HIDDEN_SIZES = [72, 144, 72, 36, 12]

FEATURE_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]

# ─────────────────────────────────────────────────────────────────────────────
# Model (duplicated here to avoid import side-effects from training script)
# ─────────────────────────────────────────────────────────────────────────────

class GammaMixtureMLP(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        layer_sizes = [36] + hidden_sizes
        layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            layers += [nn.Linear(in_sz, out_sz), nn.BatchNorm1d(out_sz), nn.ReLU()]
        layers.append(nn.Linear(hidden_sizes[-1], 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw        = self.net(x)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])
        shape2     = self.shape_min + F.softplus(raw[:, 4])
        scale2     = self.scale_min + F.softplus(raw[:, 5])
        # Reorder so component 1 is always the drier one
        swap           = (shape1 * scale1 > shape2 * scale2).float()
        shape1_out     = (1 - swap) * shape1  + swap * shape2
        scale1_out     = (1 - swap) * scale1  + swap * scale2
        shape2_out     = (1 - swap) * shape2  + swap * shape1
        scale2_out     = (1 - swap) * scale2  + swap * scale1
        mix_weight_out = (1 - swap) * mix_weight + swap * (1 - mix_weight)
        return frac_zero, mix_weight_out, shape1_out, scale1_out, shape2_out, scale2_out

# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers (scipy — no gradients needed here)
# ─────────────────────────────────────────────────────────────────────────────

def mixture_cdf(x, fz, mw, a1, b1, a2, b2):
    """Zero-inflated two-component Gamma mixture CDF.  F(0) = fz."""
    g1 = sp_gamma.cdf(x, a1, scale=b1)
    g2 = sp_gamma.cdf(x, a2, scale=b2)
    return fz + (1.0 - fz) * (mw * g1 + (1.0 - mw) * g2)

def mixture_pdf(x, fz, mw, a1, b1, a2, b2):
    """Continuous part of the zero-inflated two-component Gamma mixture PDF."""
    g1 = sp_gamma.pdf(x, a1, scale=b1)
    g2 = sp_gamma.pdf(x, a2, scale=b2)
    return (1.0 - fz) * (mw * g1 + (1.0 - mw) * g2)

# ─────────────────────────────────────────────────────────────────────────────
# Environment / config
# ─────────────────────────────────────────────────────────────────────────────

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            return 'aws', path
    return 'laptop', None


def read_config(config_file, aws_base):
    cfg = ConfigParser()
    cfg.read(config_file)
    d = cfg['DIRECTORIES']
    if 'GRAFdatadir_conus_laptop' in d:
        probs_dir = d['GRAFprobsdir_conus_laptop']
        plot_dir  = d.get('GRAF_plot_dir', probs_dir)
    else:
        base    = d.get('resnet_data_directory', aws_base or '/data/resnet_data')
        probs_dir = f'{base}/probs/'
        plot_dir  = f'{base}/plots/'
    return probs_dir, plot_dir


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — full-domain prob files for a specific IC and lead window
# ─────────────────────────────────────────────────────────────────────────────

PARAM_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]


def load_pixel_from_prob_files(probs_dir, cyyyymmddhh, clead, dot_lon, dot_lat):
    """
    Read 6 consecutive hourly gamma-mixture param files (leads clead-5..clead),
    find the pixel nearest to (dot_lon, dot_lat), and return a 36-element
    feature vector in the same layout as the training data.
    """
    lead_times = list(range(clead - 5, clead + 1))
    stacks = {k: [] for k in PARAM_VARS}
    lat2d = lon2d = None

    for lt in lead_times:
        fname = os.path.join(probs_dir,
                             f'{cyyyymmddhh}_{lt}_probs_gamma_mixture.nc')
        if not os.path.exists(fname):
            raise FileNotFoundError(f'Missing prob file: {fname}')
        with Dataset(fname, 'r') as ds:
            for k in PARAM_VARS:
                stacks[k].append(ds.variables[k][:].data.astype(np.float32))
            if lat2d is None:
                lat2d = ds.variables['lat'][:].data.astype(np.float32)
                lon2d = ds.variables['lon'][:].data.astype(np.float32)

    for k in PARAM_VARS:
        stacks[k] = np.stack(stacks[k], axis=0)   # (6, ny, nx)

    # Find nearest pixel
    dist = (lat2d - dot_lat) ** 2 + (lon2d - dot_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    found_lat = float(lat2d[iy, ix])
    found_lon = float(lon2d[iy, ix])
    print(f'Nearest pixel: ({found_lat:.3f}N, {found_lon:.3f}E)  '
          f'[grid index iy={iy}, ix={ix}]')

    # Build 36-element feature vector: [var0_t0..t5, var1_t0..t5, ...]
    raw_feat = np.concatenate(
        [stacks[k][:, iy, ix] for k in PARAM_VARS]
    ).astype(np.float32)   # (36,)

    return raw_feat

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample(ax_cdf, ax_exc, raw_feat, target, model, feat_mean, feat_std,
                clead, label):
    """Fill both axes for one validation sample."""

    # ── hourly parameters ────────────────────────────────────────────────
    # Features stored as [var0_t0..t5, var1_t0..t5, ...]  (6 per variable)
    hp = {vname: np.array([raw_feat[vi * 6 + t] for t in range(6)])
          for vi, vname in enumerate(FEATURE_VARS)}

    # ── 6-hourly MLP prediction ──────────────────────────────────────────
    feat_norm = (raw_feat - feat_mean) / feat_std
    with torch.no_grad():
        fz6, mw6, s1_6, sc1_6, s2_6, sc2_6 = model(
            torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0))
    fz6, mw6  = fz6.item(),  mw6.item()
    s1_6, sc1_6 = s1_6.item(), sc1_6.item()
    s2_6, sc2_6 = s2_6.item(), sc2_6.item()

    # ── x-axis: up to 99.5th percentile of the 6-h distribution ─────────
    x_scan   = np.linspace(0.01, 250.0, 8000)
    cdf_scan = mixture_cdf(x_scan, fz6, mw6, s1_6, sc1_6, s2_6, sc2_6)
    idx_995  = int(np.searchsorted(cdf_scan, 0.995))
    x_max    = max(float(x_scan[min(idx_995, len(x_scan) - 1)]),
                   (float(target) * 1.1) if target is not None else 0.0,
                   2.0)
    x_max    = min(x_max, 200.0)

    x_cdf = np.linspace(0.0,  x_max, 600)   # includes 0 → F(0) = fz
    x_exc = np.linspace(1e-3, x_max, 600)   # avoid x=0 for exceedance plot

    # ── colour scheme ────────────────────────────────────────────────────
    hour_colors = plt.cm.tab10(np.linspace(0.0, 0.55, 6))
    lead_hours  = [clead - 5 + t for t in range(6)]

    # ── hourly lines (thin) ──────────────────────────────────────────────
    for t in range(6):
        fz_h  = float(hp['fraction_zero'][t])
        mw_h  = float(hp['mixture_weight'][t])
        a1_h  = float(hp['gamma_shape1'][t])
        b1_h  = float(hp['gamma_scale1'][t])
        a2_h  = float(hp['gamma_shape2'][t])
        b2_h  = float(hp['gamma_scale2'][t])
        hlabel = f'h={lead_hours[t]}'
        col    = hour_colors[t]

        ax_cdf.plot(x_cdf, mixture_cdf(x_cdf, fz_h, mw_h, a1_h, b1_h, a2_h, b2_h),
                    color=col, linewidth=1.0, label=hlabel)
        exc = 1.0 - mixture_cdf(x_exc, fz_h, mw_h, a1_h, b1_h, a2_h, b2_h)
        ax_exc.plot(x_exc, np.maximum(exc, 1e-6),
                    color=col, linewidth=1.0, label=hlabel)

    # ── 6-hourly MLP line (thick black) ──────────────────────────────────
    ax_cdf.plot(x_cdf, mixture_cdf(x_cdf, fz6, mw6, s1_6, sc1_6, s2_6, sc2_6),
                color='black', linewidth=2.5, label='6-h MLP')
    exc6 = 1.0 - mixture_cdf(x_exc, fz6, mw6, s1_6, sc1_6, s2_6, sc2_6)
    ax_exc.plot(x_exc, np.maximum(exc6, 1e-6),
                color='black', linewidth=2.5, label='6-h MLP')

    # ── observed MRMS (dashed crimson) — only if target is provided ──────
    if target is not None:
        obs_label = f'MRMS {target:.1f} mm'
        for ax in (ax_cdf, ax_exc):
            ax.axvline(target, color='crimson', linestyle='--',
                       linewidth=1.5, label=obs_label)

    # ── formatting ───────────────────────────────────────────────────────
    ax_cdf.set_title('(a) CDFs')
    ax_exc.set_title('(b) Exceedance probability')

    for ax in (ax_cdf, ax_exc):
        ax.set_xlabel('Precipitation (mm)')
        ax.set_xlim(0.0, x_max)
    ax_exc.legend(fontsize=14, loc='upper right')

    ax_cdf.set_ylim(0.0, 1.05)
    ax_cdf.set_ylabel('Cumulative probability')
    ax_cdf.text(0.98, 0.05, label,
                transform=ax_cdf.transAxes,
                ha='right', va='bottom', fontsize=11,
                color='0.4')

    ax_exc.set_yscale('log')
    ax_exc.set_ylim(1e-2, 1.05)
    ax_exc.set_ylabel('Exceedance probability  1 − F(x)')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print('Usage: python plot_6hourly_mlp_samples2.py <cyyyymmddhh> <clead>')
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead       = int(sys.argv[2])

    if clead < 6:
        print('ERROR: clead must be >= 6 (need 6 consecutive hourly files)')
        sys.exit(1)

    # ── environment / config ─────────────────────────────────────────────
    env, aws_base = detect_environment()
    config_file   = 'config_aws.ini' if env == 'aws' else 'config_laptop.ini'
    print(f'Environment: {env}  |  Config: {config_file}')
    probs_dir, plot_dir = read_config(config_file, aws_base)
    os.makedirs(plot_dir, exist_ok=True)

    # ── load checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.join(TRAIN_DIR, f'6h_mlp_lead{clead}h.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = GammaMixtureMLP(
        hidden_sizes=ckpt.get('hidden_sizes', HIDDEN_SIZES),
        shape_min=ckpt.get('shape_min', SHAPE_MIN),
        scale_min=ckpt.get('scale_min', SCALE_MIN),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    feat_mean = ckpt['feature_mean']
    feat_std  = ckpt['feature_std']

    # ── load pixel at dot location ───────────────────────────────────────
    print(f'Loading prob files: IC={cyyyymmddhh}  leads {clead-5}–{clead} h')
    print(f'Dot location: lon={DOT_LON}, lat={DOT_LAT}')
    raw_feat = load_pixel_from_prob_files(
        probs_dir, cyyyymmddhh, clead, DOT_LON, DOT_LAT)

    # ── plot ─────────────────────────────────────────────────────────────
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 12 * 1.75})

    fig, (ax_cdf, ax_exc) = plt.subplots(1, 2, figsize=(16, 7.05))
    #fig.suptitle(f'IC {cyyyymmddhh}  lead {clead} h  '
    #             f'(lon={DOT_LON}, lat={DOT_LAT})', fontsize=16)

    label = f'IC {cyyyymmddhh},  lead {clead} h'
    plot_sample(ax_cdf, ax_exc, raw_feat, None,
                model, feat_mean, feat_std, clead, label)

    fig.tight_layout()

    outname = os.path.join(plot_dir,
                           f'6h_sample_IC{cyyyymmddhh}_lead{clead}h.png')
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outname}')


if __name__ == '__main__':
    main()
