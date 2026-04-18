"""
plot_6hourly_mlp_samples.py — diagnostic CDF/PDF plots for the 6-hourly MLP

Usage:
    python plot_6hourly_mlp_samples.py <clead> [n_plots]

    clead    : lead time in hours (e.g. 24)
    n_plots  : number of validation samples to plot (default 20)

For each chosen validation sample a two-panel figure is produced:
    (a) CDFs — 6 thin coloured lines (one per hourly forecast, h=clead-5
                to h=clead) plus one thick black line for the 6-hourly
                MLP prediction.
    (b) PDFs — same structure.

A vertical dashed crimson line marks the observed MRMS 6-hourly total.
The suptitle gives the validation sample index.

Output: plots/6h_sample_{idx:05d}_lead{clead}h.png

Tom Hamill, Apr 2026
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma as sp_gamma
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match train_6hourly_mlp.py
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, 'data')
TRAIN_DIR    = os.path.join(SCRIPT_DIR, 'mlp_trainings')
PLOTS_DIR    = os.path.join(SCRIPT_DIR, 'plots')

VAL_FRAC     = 0.20
RANDOM_SEED  = 42
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(clead):
    pattern = os.path.join(DATA_DIR, f'prob_MRMS_samples_*_lead{clead}h.nc')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No data files found: {pattern}')
    feat_list, tgt_list = [], []
    for fpath in files:
        with Dataset(fpath, 'r') as ds:
            blocks = [ds[v][:].data.astype(np.float32) for v in FEATURE_VARS]
            feat_list.append(np.concatenate(blocks, axis=1))   # (N, 36)
            tgt_list.append(ds['target_precip_6h'][:].data.astype(np.float32))
    return np.concatenate(feat_list, axis=0), np.concatenate(tgt_list, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample(ax_cdf, ax_pdf, raw_feat, target, model, feat_mean, feat_std,
                clead, global_idx):
    """Fill both axes for one validation sample."""

    # ── hourly parameters ────────────────────────────────────────────────
    # Features stored as [var0_t0..t5, var1_t0..t5, ...]  (6 per variable)
    hour_params = []
    for vi in range(6):
        hour_params.append({
            vname: float(raw_feat[vi * 6 + t])
            for t, vname in enumerate(FEATURE_VARS)
        })
    # Re-key by variable name → list of 6 values across hours
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
                   float(target) * 1.1, 2.0)
    x_max    = min(x_max, 200.0)

    x_cdf = np.linspace(0.0,  x_max, 600)   # includes 0 → F(0) = fz
    x_pdf = np.linspace(1e-3, x_max, 600)   # avoid log(0) in pdf

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
        label = f'h={lead_hours[t]}'
        col   = hour_colors[t]

        ax_cdf.plot(x_cdf, mixture_cdf(x_cdf, fz_h, mw_h, a1_h, b1_h, a2_h, b2_h),
                    color=col, linewidth=1.0, label=label)
        ax_pdf.plot(x_pdf, mixture_pdf(x_pdf, fz_h, mw_h, a1_h, b1_h, a2_h, b2_h),
                    color=col, linewidth=1.0, label=label)

    # ── 6-hourly MLP line (thick black) ──────────────────────────────────
    ax_cdf.plot(x_cdf, mixture_cdf(x_cdf, fz6, mw6, s1_6, sc1_6, s2_6, sc2_6),
                color='black', linewidth=2.5, label='6-h MLP')
    ax_pdf.plot(x_pdf, mixture_pdf(x_pdf, fz6, mw6, s1_6, sc1_6, s2_6, sc2_6),
                color='black', linewidth=2.5, label='6-h MLP')

    # ── observed MRMS (dashed crimson) ───────────────────────────────────
    obs_label = f'MRMS {target:.1f} mm'
    for ax in (ax_cdf, ax_pdf):
        ax.axvline(target, color='crimson', linestyle='--',
                   linewidth=1.5, label=obs_label)

    # ── formatting ───────────────────────────────────────────────────────
    ax_cdf.set_title('(a) CDFs')
    ax_pdf.set_title('(b) PDFs')

    for ax in (ax_cdf, ax_pdf):
        ax.set_xlabel('Precipitation (mm)')
        ax.set_xlim(0.0, x_max)
        ax.legend(fontsize=7, loc='upper right')

    ax_cdf.set_ylim(0.0, 1.05)
    ax_cdf.set_ylabel('Cumulative probability')
    ax_pdf.set_ylabel('Probability density')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_6hourly_mlp_samples.py <clead> [n_plots]')
        sys.exit(1)

    clead   = int(sys.argv[1])
    n_plots = int(sys.argv[2]) if len(sys.argv) > 2 else 20

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

    feat_mean = ckpt['feature_mean']   # (36,) numpy array
    feat_std  = ckpt['feature_std']

    # ── load data and reconstruct validation split (same seed as training) ─
    raw_features, targets = load_raw_data(clead)
    n       = len(targets)
    n_train = n - int(n * VAL_FRAC)

    rng      = np.random.default_rng(RANDOM_SEED)
    perm     = rng.permutation(n)
    val_idx  = perm[n_train:]           # global indices of validation samples

    # ── choose n_plots samples ───────────────────────────────────────────
    plot_rng  = np.random.default_rng(RANDOM_SEED + 1)
    chosen    = plot_rng.choice(len(val_idx),
                                size=min(n_plots, len(val_idx)),
                                replace=False)
    chosen_global = val_idx[chosen]     # global indices into raw_features

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f'Plotting {len(chosen_global)} validation samples, lead {clead} h')
    print(f'Output directory: {PLOTS_DIR}')

    for plot_num, global_idx in enumerate(chosen_global):
        raw_feat = raw_features[global_idx]
        target   = float(targets[global_idx])

        fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(12, 5))

        plot_sample(ax_cdf, ax_pdf, raw_feat, target,
                    model, feat_mean, feat_std, clead, global_idx)

        fig.suptitle(f'Validation sample {int(global_idx)}  '
                     f'(lead {clead} h)',
                     fontsize=12)
        fig.tight_layout()

        outname = os.path.join(PLOTS_DIR,
                               f'6h_sample_{int(global_idx):05d}_lead{clead}h.png')
        fig.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f'  [{plot_num + 1:3d}/{len(chosen_global)}] '
              f'sample={int(global_idx)}  target={target:.2f} mm  → {outname}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
