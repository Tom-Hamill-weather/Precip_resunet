"""
make_plots_mlp_sensitivity.py  —  MLP serial-correlation figure

Single-panel figure built from the actual training data for a chosen lead time.

For each sample the script computes:
    x  = mean(x_h)   across 6 consecutive hourly inputs        (mm/h)
    sigma(x) = std(x_h) across the same 6 hours                (mm/h)
    where  x_h = E[X_h] = (1 - p0_h) * [w_h * a1_h * t1_h + (1-w_h) * a2_h * t2_h]
    is the expected precipitation of hourly forecast h.  (Y is reserved
    elsewhere in the manuscript for the MRMS target, hence x here.)

Both axes use a square-root transform, which gives natural spacing for
precipitation data without requiring an artificial offset for near-zero values.

The figure shows the binned-median ratio  q_MLP / q_naive  as a function of
(x, sigma(x)), where q_naive is the 90th percentile of the sum of 6
*independent* draws from the hourly distributions (Monte Carlo).  Ratio > 1
indicates that positive serial correlation (persistence) causes the MLP to
assign a heavier tail than independence predicts.

Contour iso-lines are drawn at explicit round-number physical levels and are
computed from the same binned medians as the hexbin coloring, ensuring they
are consistent with the colored background.  Bins are deliberately coarse
(see HEXBIN_GRID / CONTOUR_BINS) to damp sample-count noise.

Usage:
    python make_plots_mlp_sensitivity.py [clead]   (default: 24)

Output:
    my_tex/Figs_6h/mlp_sensitivity_lead{clead}h.png

Tom Hamill, May 2026
"""

import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.special import gammainc
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Configuration
# =========================================================================

QUANTILE       = 0.90    # output quantile to study
MC_DRAWS       = 300     # Monte Carlo draws per sample (naive-independence)
MC_BATCH_SIZE  = 5000    # samples per MC batch
MLP_FWD_BATCH  = 131072  # MLP forward-pass batch size
MIN_MEAN_EY    = 0.01    # drop samples with mean E[Y_h] < this (mm/h)
MIN_Q_NAIVE    = 0.05    # min naive q for ratio to be included
HEXBIN_GRID    = 28      # hexbin gridsize (coarse, to damp sampling noise)
HEXBIN_MINCNT  = 8       # min samples per hexbin for it to be drawn
CONTOUR_BINS   = 22      # rectangular bins for contour computation
CONTOUR_MINCNT = 15      # min samples per contour bin for it to be used
CONTOUR_SIGMA  = 0.8     # light Gaussian smoothing sigma for contours

# Font sizes: current values scaled ×1.2; contour labels ×1.5
F_SUPTITLE  = 19
F_TITLE     = 16
F_LABEL     = 16
F_TICK      = 11
F_CB_LABEL  = 14
F_CB_TICK   = 13
F_ANNOT     = 12
F_CLABEL    = 11   # contour labels: original 7pt × 1.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'my_tex', 'Figs_6h')

SHAPE_MIN    = 0.1
SCALE_MIN    = 0.01
HIDDEN_SIZES = [72, 144, 72, 36, 12]

FEATURE_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]

# =========================================================================
# MLP  (identical to train_6hourly_mlp.py)
# =========================================================================

class GammaMixtureMLP(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        layer_sizes = [36] + hidden_sizes
        layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            layers += [nn.Linear(in_sz, out_sz),
                       nn.BatchNorm1d(out_sz),
                       nn.ReLU()]
        layers.append(nn.Linear(hidden_sizes[-1], 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        fz  = torch.sigmoid(raw[:, 0])
        mw  = torch.sigmoid(raw[:, 1])
        s1  = self.shape_min + F.softplus(raw[:, 2])
        sc1 = self.scale_min + F.softplus(raw[:, 3])
        s2  = self.shape_min + F.softplus(raw[:, 4])
        sc2 = self.scale_min + F.softplus(raw[:, 5])
        swap    = (s1 * sc1 > s2 * sc2).float()
        s1_out  = (1 - swap) * s1  + swap * s2
        sc1_out = (1 - swap) * sc1 + swap * sc2
        s2_out  = (1 - swap) * s2  + swap * s1
        sc2_out = (1 - swap) * sc2 + swap * sc1
        mw_out  = (1 - swap) * mw  + swap * (1 - mw)
        return fz, mw_out, s1_out, sc1_out, s2_out, sc2_out


def load_mlp(clead, device):
    path = os.path.join(SCRIPT_DIR, 'mlp_trainings', f'6h_mlp_lead{clead}h.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GammaMixtureMLP(
        hidden_sizes=ckpt.get('hidden_sizes', HIDDEN_SIZES),
        shape_min=ckpt.get('shape_min', SHAPE_MIN),
        scale_min=ckpt.get('scale_min', SCALE_MIN),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f'Loaded MLP (epoch {ckpt["epoch"]+1}) from {path}')
    return model, ckpt['feature_mean'], ckpt['feature_std']


# =========================================================================
# Data loading
# =========================================================================

def _locate_data_dir():
    for base in ['/data/resnet_data', '/data2/resnet_data']:
        cand = os.path.join(base, 'prob_samples')
        if os.path.isdir(cand):
            return cand
    raise RuntimeError(
        'Cannot locate prob_samples directory under /data/resnet_data '
        'or /data2/resnet_data.')


def load_data(clead):
    data_dir = _locate_data_dir()
    pattern  = os.path.join(data_dir, f'prob_MRMS_samples_*_lead{clead}h.nc')
    files    = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files matching:\n  {pattern}')
    print(f'Loading {len(files)} file(s) for lead {clead} h ...')
    feat_list, tgt_list = [], []
    for fp in files:
        with Dataset(fp, 'r') as ds:
            blocks = [ds[v][:].data.astype(np.float32) for v in FEATURE_VARS]
            feat_list.append(np.concatenate(blocks, axis=1))        # (n, 36)
            tgt_list.append(ds['target_precip_6h'][:].data.astype(np.float32))
    features = np.concatenate(feat_list, axis=0)
    targets  = np.concatenate(tgt_list,  axis=0)
    print(f'  {len(targets):,} total samples  '
          f'(wet fraction {(targets > 0).mean():.3f})')
    return features, targets


# =========================================================================
# Per-hour expected value
#   features columns: 0:6 frac_zero | 6:12 mix_wt | 12:18 shape1 | 18:24 scale1
#                     24:30 shape2   | 30:36 scale2
# =========================================================================

def hourly_expected_value(features):
    """Return (N, 6) array of E[X_h] for each of the 6 input lead-time hours."""
    fz  = features[:, 0:6]
    mw  = features[:, 6:12]
    s1  = features[:, 12:18];  sc1 = features[:, 18:24]
    s2  = features[:, 24:30];  sc2 = features[:, 30:36]
    mu_cond = mw * s1 * sc1 + (1.0 - mw) * s2 * sc2
    return (1.0 - fz) * mu_cond   # (N, 6)


# =========================================================================
# MLP forward pass (batched)
# =========================================================================

def run_mlp_batched(model, feat_mean, feat_std, features, device):
    safe_std  = np.where(feat_std < 1e-8, 1.0, feat_std)
    norm_feat = (features - feat_mean) / safe_std
    N    = len(norm_feat)
    outs = [[] for _ in range(6)]
    with torch.no_grad():
        for s in range(0, N, MLP_FWD_BATCH):
            xb = torch.tensor(norm_feat[s:s + MLP_FWD_BATCH],
                              dtype=torch.float32, device=device)
            for i, t in enumerate(model(xb)):
                outs[i].append(t.cpu().numpy())
    return [np.concatenate(o) for o in outs]   # 6 arrays, each shape (N,)


# =========================================================================
# 6-h MLP quantile: vectorized bisection on the ZI two-component Gamma CDF
#
#   F(y) = p0 + (1-p0) * [ w * gammainc(a1, y/t1) + (1-w) * gammainc(a2, y/t2) ]
# =========================================================================

def mixture_quantile(fz, mw, s1, sc1, s2, sc2, q, n_iter=52, y_max=500.0):
    """Return the q-th quantile for each sample via vectorized bisection."""
    N      = len(fz)
    result = np.zeros(N, dtype=np.float32)
    active = (fz < q)
    if not active.any():
        return result

    _fz  = fz[active].astype(np.float64)
    _mw  = mw[active].astype(np.float64)
    _s1  = s1[active].astype(np.float64)
    _sc1 = sc1[active].astype(np.float64)
    _s2  = s2[active].astype(np.float64)
    _sc2 = sc2[active].astype(np.float64)

    lo = np.full(active.sum(), 1e-9)
    hi = np.full(active.sum(), float(y_max))

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        cdf = (_fz
               + (1.0 - _fz) * (_mw  * gammainc(_s1, mid / _sc1)
                                 + (1.0 - _mw) * gammainc(_s2, mid / _sc2)))
        lo = np.where(cdf < q,  mid, lo)
        hi = np.where(cdf >= q, mid, hi)

    result[active] = 0.5 * (lo + hi)
    return result


# =========================================================================
# Naive-independence quantile: Monte Carlo sum of 6 independent ZI-Gamma draws
# =========================================================================

def naive_independence_quantile(features, q=QUANTILE,
                                 n_draws=MC_DRAWS, batch=MC_BATCH_SIZE, seed=42):
    """
    For each sample, draw n_draws realisations from each of the 6 hourly
    zero-inflated Gamma mixture distributions, sum across hours, and return
    the empirical q-th quantile.  Loops over hours inside each batch to keep
    peak memory around batch * n_draws * 8 bytes (~12 MB at default settings).
    """
    rng = np.random.default_rng(seed)

    fz  = features[:, 0:6].astype(np.float64)
    mw  = features[:, 6:12].astype(np.float64)
    s1  = features[:, 12:18].astype(np.float64)
    sc1 = features[:, 18:24].astype(np.float64)
    s2  = features[:, 24:30].astype(np.float64)
    sc2 = features[:, 30:36].astype(np.float64)

    N       = len(features)
    q_out   = np.empty(N, dtype=np.float32)
    n_batch = (N + batch - 1) // batch

    for b, start in enumerate(range(0, N, batch)):
        end = min(start + batch, N)
        B   = end - start

        if b % 20 == 0:
            print(f'  MC batch {b+1}/{n_batch}  ({end:,}/{N:,})')

        sums = np.zeros((B, n_draws), dtype=np.float64)

        for h in range(6):
            fz_h  = fz[start:end,  h]
            mw_h  = mw[start:end,  h]
            s1_h  = s1[start:end,  h]
            sc1_h = sc1[start:end, h]
            s2_h  = s2[start:end,  h]
            sc2_h = sc2[start:end, h]

            is_wet = rng.random((B, n_draws)) > fz_h[:, None]
            use_c1 = rng.random((B, n_draws)) < mw_h[:, None]

            # (B, 1) shape parameter broadcasts to (B, n_draws)
            g1 = rng.standard_gamma(s1_h[:, None],  (B, n_draws)) * sc1_h[:, None]
            g2 = rng.standard_gamma(s2_h[:, None],  (B, n_draws)) * sc2_h[:, None]

            sums += is_wet * np.where(use_c1, g1, g2)

        q_out[start:end] = np.quantile(sums, q, axis=1)

    return q_out


# =========================================================================
# Axis-tick helper: sqrt-space axes labeled with physical mm/h values
# =========================================================================

def _set_sqrt_ticks(ax, xlim, ylim):
    """Label sqrt-transformed axes with the underlying physical (mm/h) values."""
    phys = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    def _make(lo, hi):
        ts = [np.sqrt(v) for v in phys if lo - 0.03 <= np.sqrt(v) <= hi + 0.03]
        ls = [f'{v:g}' for v in phys if lo - 0.03 <= np.sqrt(v) <= hi + 0.03]
        return ts, ls
    xt, xl = _make(*xlim)
    yt, yl = _make(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter(xl))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(yl))
    ax.tick_params(labelsize=F_TICK)


# =========================================================================
# Consistent contour overlay
#
# Bins (tx, ty) → statistic(z_phys) on a rectangular grid using the same
# statistic as the calling panel's hexbin, fills empty bins with
# nearest-neighbour interpolation to avoid edge artifacts from Gaussian
# smoothing, then draws contours at the specified physical levels.  Because
# the hexbin and the contour use the same per-bin statistic, the iso-lines
# are consistent with the background colour field.
# =========================================================================

def _add_contours(ax, tx, ty, z_phys, xlim, ylim, levels, colors,
                  n_bins=CONTOUR_BINS, sigma=CONTOUR_SIGMA, fmt='%.3g',
                  statistic='median'):
    stat, xedge, yedge, _ = binned_statistic_2d(
        tx, ty, z_phys, statistic=statistic, bins=n_bins, range=[xlim, ylim])
    cnt,  *_ = binned_statistic_2d(
        tx, ty, z_phys, statistic='count',  bins=n_bins, range=[xlim, ylim])

    bad = (cnt < CONTOUR_MINCNT) | ~np.isfinite(stat)

    # Fill NaN bins via nearest-neighbour before smoothing to prevent
    # the constant-fill artifact that shifts contours away from the data.
    if bad.any() and (~bad).sum() > 10:
        ix  = np.arange(n_bins)
        Xi, Yi = np.meshgrid(ix, ix, indexing='ij')
        good = ~bad
        nn   = NearestNDInterpolator(
            np.column_stack([Xi[good], Yi[good]]), stat[good])
        filled = nn(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(n_bins, n_bins)
    else:
        filled = np.where(bad, np.nanmedian(stat[~bad]) if (~bad).any() else 0.0, stat)

    smoothed    = gaussian_filter(filled, sigma=sigma)
    smoothed_ma = np.ma.array(smoothed, mask=bad)

    xc = 0.5 * (xedge[:-1] + xedge[1:])
    yc = 0.5 * (yedge[:-1] + yedge[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

    valid_levels = [lv for lv in levels
                    if smoothed_ma.min() <= lv <= smoothed_ma.max()]
    if not valid_levels:
        return

    try:
        cs = ax.contour(Xc, Yc, smoothed_ma,
                        levels=valid_levels, colors=colors,
                        linewidths=1.1, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=F_CLABEL, fmt=fmt)
    except Exception:
        pass


# =========================================================================
# Figure
# =========================================================================

def make_figure(x, y, q_mlp, q_naive, clead):
    eps_q = 1e-6

    # Square-root transform for both axes (handles y=0 naturally)
    sx = np.sqrt(x)
    sy = np.sqrt(y)

    ok    = np.isfinite(sx) & np.isfinite(sy) & (q_mlp > 0)
    ratio = np.where(q_naive > MIN_Q_NAIVE,
                     q_mlp / (q_naive + eps_q), np.nan)
    ok    = ok & np.isfinite(ratio) & (ratio > 0)

    xlim = (np.sqrt(np.percentile(x[ok],  0.5)),
            np.sqrt(np.percentile(x[ok], 99.5)))
    ylim = (np.sqrt(np.percentile(y[ok],  0.5)),
            np.sqrt(np.percentile(y[ok], 99.5)))
    ext  = [xlim[0], xlim[1], ylim[0], ylim[1]]

    qlabel = f'q{int(QUANTILE * 100)}'

    fig = plt.figure(figsize=(7.5, 6.2))
    ax  = fig.add_axes([0.11, 0.12, 0.72, 0.78])
    cax = fig.add_axes([0.86, 0.12, 0.03, 0.78])
    fig.suptitle(
        f'6-h MLP vs. naive-independence {qlabel}  — lead {clead} h',
        fontsize=F_SUPTITLE, y=0.97)

    r_vals = ratio[ok]
    vmin_r = max(np.nanpercentile(r_vals,  2), 0.15)
    vmax_r = min(np.nanpercentile(r_vals, 98), 5.0)
    vmin_r = min(vmin_r, 0.85)   # keep centre at 1.0 in range
    vmax_r = max(vmax_r, 1.15)

    hb = ax.hexbin(
        sx[ok], sy[ok], C=r_vals,
        reduce_C_function=np.median,
        gridsize=HEXBIN_GRID, extent=ext, mincnt=HEXBIN_MINCNT,
        cmap='RdBu_r', linewidths=0.15,
        norm=TwoSlopeNorm(vmin=vmin_r, vcenter=1.0, vmax=vmax_r),
    )

    cb = fig.colorbar(hb, cax=cax)
    r_ticks = [t for t in [0.25, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
               if vmin_r * 0.9 <= t <= vmax_r * 1.1]
    cb.set_ticks(r_ticks)
    cb.set_ticklabels([f'{t:g}' for t in r_ticks], fontsize=F_CB_TICK)
    cb.set_label(f'MLP {qlabel} / naive-independence {qlabel}', fontsize=F_CB_LABEL)

    # Contour at physically meaningful ratio levels
    contour_levels = [lv for lv in [0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0]
                      if vmin_r <= lv <= vmax_r]
    _add_contours(ax, sx[ok], sy[ok], r_vals,
                  xlim, ylim, contour_levels, colors='black', fmt='%g')

    ax.set_xlabel('Mean hourly  E[X]  (mm/h)', fontsize=F_LABEL)
    ax.set_ylabel(r'$\sigma(x)$  (mm/h)', fontsize=F_LABEL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _set_sqrt_ticks(ax, xlim, ylim)

    # ------------------------------------------------------------------ #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'mlp_sensitivity_lead{clead}h.png')
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)
    return out_path


# =========================================================================
# Main
# =========================================================================

def main():
    clead  = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Lead {clead} h  |  quantile q{int(QUANTILE*100)}  |  device {device}')

    features, _ = load_data(clead)

    ey = hourly_expected_value(features)   # (N, 6)
    x  = ey.mean(axis=1)
    y  = ey.std(axis=1)

    keep = x >= MIN_MEAN_EY
    print(f'Retaining {keep.sum():,}/{len(keep):,} samples '
          f'(mean E[Y] >= {MIN_MEAN_EY} mm/h)')
    features = features[keep]
    x = x[keep]
    y = y[keep]

    if len(x) < 500:
        print('WARNING: fewer than 500 samples after filter — figure may look sparse.')

    model, feat_mean, feat_std = load_mlp(clead, device)
    print('Running MLP forward pass ...')
    fz6, mw6, s1_6, sc1_6, s2_6, sc2_6 = run_mlp_batched(
        model, feat_mean, feat_std, features, device)

    print(f'Computing MLP q{int(QUANTILE*100)} analytically ...')
    q_mlp = mixture_quantile(fz6, mw6, s1_6, sc1_6, s2_6, sc2_6, QUANTILE)
    print(f'  q{int(QUANTILE*100)} range: '
          f'{np.percentile(q_mlp,  1):.2f} – '
          f'{np.percentile(q_mlp, 99):.2f} mm  (1st–99th pct)')

    print(f'Computing naive-independence q{int(QUANTILE*100)} '
          f'({MC_DRAWS} MC draws per sample) ...')
    q_naive = naive_independence_quantile(features)
    print(f'  naive q{int(QUANTILE*100)} range: '
          f'{np.percentile(q_naive,  1):.2f} – '
          f'{np.percentile(q_naive, 99):.2f} mm')

    valid   = (q_naive > MIN_Q_NAIVE) & (q_mlp > 0)
    ratio_v = q_mlp[valid] / (q_naive[valid] + 1e-6)
    print(f'  Ratio MLP/naive — median {np.median(ratio_v):.3f}  '
          f'p10={np.percentile(ratio_v, 10):.3f}  '
          f'p90={np.percentile(ratio_v, 90):.3f}')

    print('Generating figure ...')
    out_path = make_figure(x, y, q_mlp, q_naive, clead)
    print(f'\nDone.  Figure written to:\n  {out_path}')


if __name__ == '__main__':
    main()
