"""
make_plots_mlp_kurtosis.py  —  EXPLORATORY  (not yet part of the manuscript)

Companion to make_plots_mlp_sensitivity.py.  Tests the hypothesis that
treating the six hourly forecasts as independent (as in the naive-independence
baseline) makes their 6-h sum more Gaussian (lower kurtosis, via a
Central-Limit-Theorem argument), whereas the MLP's single fitted 6-h
distribution should retain more skewness/kurtosis inherited from serial
correlation.

Both the MLP's 6-h distribution and each hourly input distribution are
zero-inflated two-component Gamma mixtures, so their moments are analytic.
This script:

  1. Computes the raw moments (1st-4th) of the MLP's predicted 6-h mixture
     directly, converts to cumulants, and computes Pearson's kurtosis
     mu4 / mu2^2.
  2. Computes the same cumulants for each of the 6 *hourly* input mixtures,
     and sums the cumulants across hours — cumulants add exactly for sums of
     independent random variables, so this gives the EXACT kurtosis of the
     naive-independence 6-h sum with no Monte Carlo sampling required (a
     sampling-based estimate of the 4th moment would be far too noisy).
  3. Plots the binned-median ratio kurtosis_MLP / kurtosis_naive as a
     function of (x, sigma(x)), same axes as make_plots_mlp_sensitivity.py.

Ratio > 1 supports the CLT hypothesis: the MLP's output is more leptokurtic
(heavier-tailed / more peaked) than independence would predict.

Usage:
    python make_plots_mlp_kurtosis.py [clead]   (default: 24)

Output:
    mlp_kurtosis_ratio_lead{clead}h.png   (repo root — exploratory, not in my_tex)

Tom Hamill, May 2026
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

import torch

import make_plots_mlp_sensitivity as base

MIN_VAR   = 0.1     # mm^2 floor on both MLP and naive variance, to exclude
                    # near-degenerate (nearly-always-dry) samples where the
                    # kurtosis ratio is numerically unstable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Analytic moments of a zero-inflated two-component Gamma mixture
# =========================================================================

def _gamma_raw_moment(alpha, theta, k):
    """E[G^k] for Gamma(alpha, theta):  theta^k * alpha(alpha+1)...(alpha+k-1)."""
    factor = np.ones_like(alpha)
    for j in range(k):
        factor = factor * (alpha + j)
    return theta**k * factor


def _zi_mixture_raw_moments(p0, w, a1, t1, a2, t2, k_max=4):
    """Raw moments 1..k_max of Y = 0 w.p. p0, else a two-component Gamma mixture."""
    moms = []
    for k in range(1, k_max + 1):
        g1k = _gamma_raw_moment(a1, t1, k)
        g2k = _gamma_raw_moment(a2, t2, k)
        moms.append((1.0 - p0) * (w * g1k + (1.0 - w) * g2k))
    return moms   # [m1, m2, m3, m4]


def _raw_to_cumulants(m):
    m1, m2, m3, m4 = m
    k1 = m1
    k2 = m2 - m1**2
    k3 = m3 - 3*m1*m2 + 2*m1**3
    k4 = m4 - 4*m1*m3 - 3*m2**2 + 12*m1**2*m2 - 6*m1**4
    return k1, k2, k3, k4


def _kurtosis_from_cumulants(k2, k4):
    """Pearson's kurtosis mu4/mu2^2 (Gaussian = 3), given cumulants of a
    single distribution OR summed cumulants of independent contributions."""
    mu4 = k4 + 3.0 * k2**2
    return mu4 / np.maximum(k2**2, 1e-300), k2


def mlp_kurtosis(fz, mw, s1, sc1, s2, sc2):
    """Kurtosis of the MLP's own predicted 6-h ZI-Gamma-mixture distribution."""
    m = _zi_mixture_raw_moments(fz, mw, s1, sc1, s2, sc2)
    k1, k2, k3, k4 = _raw_to_cumulants(m)
    return _kurtosis_from_cumulants(k2, k4)


def naive_sum_kurtosis(features):
    """Exact kurtosis of the sum of 6 independent (non-identical) hourly
    ZI-Gamma-mixture distributions, via cumulant addition (no Monte Carlo)."""
    fz  = features[:, 0:6].astype(np.float64)
    mw  = features[:, 6:12].astype(np.float64)
    s1  = features[:, 12:18].astype(np.float64)
    sc1 = features[:, 18:24].astype(np.float64)
    s2  = features[:, 24:30].astype(np.float64)
    sc2 = features[:, 30:36].astype(np.float64)

    k2_sum = np.zeros(len(features))
    k4_sum = np.zeros(len(features))
    for h in range(6):
        m = _zi_mixture_raw_moments(fz[:, h], mw[:, h], s1[:, h],
                                     sc1[:, h], s2[:, h], sc2[:, h])
        _, k2, _, k4 = _raw_to_cumulants(m)
        k2_sum += k2
        k4_sum += k4
    return _kurtosis_from_cumulants(k2_sum, k4_sum)


# =========================================================================
# Figure
# =========================================================================

def make_figure(x, y, ratio, ok, clead):
    sx, sy = np.sqrt(x), np.sqrt(y)

    xlim = (np.sqrt(np.percentile(x[ok],  0.5)),
            np.sqrt(np.percentile(x[ok], 99.5)))
    ylim = (np.sqrt(np.percentile(y[ok],  0.5)),
            np.sqrt(np.percentile(y[ok], 99.5)))
    ext  = [xlim[0], xlim[1], ylim[0], ylim[1]]

    log_ratio = np.log2(ratio[ok])
    vmax_l = min(np.percentile(np.abs(log_ratio), 98), 4.0)
    vmin_l = -vmax_l

    fig = plt.figure(figsize=(7.5, 6.2))
    ax  = fig.add_axes([0.11, 0.12, 0.72, 0.78])
    cax = fig.add_axes([0.86, 0.12, 0.03, 0.78])
    fig.suptitle(
        f'EXPLORATORY — MLP vs. naive-independence kurtosis  — lead {clead} h',
        fontsize=base.F_SUPTITLE * 0.85, y=0.97)

    hb = ax.hexbin(
        sx[ok], sy[ok], C=log_ratio,
        reduce_C_function=np.median,
        gridsize=base.HEXBIN_GRID, extent=ext, mincnt=base.HEXBIN_MINCNT,
        cmap='RdBu_r', linewidths=0.15,
        norm=TwoSlopeNorm(vmin=vmin_l, vcenter=0.0, vmax=vmax_l),
    )

    cb = fig.colorbar(hb, cax=cax)
    ratio_ticks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    tick_locs   = [np.log2(t) for t in ratio_ticks if vmin_l <= np.log2(t) <= vmax_l]
    tick_labs   = [f'{t:g}' for t in ratio_ticks if vmin_l <= np.log2(t) <= vmax_l]
    cb.set_ticks(tick_locs)
    cb.set_ticklabels(tick_labs, fontsize=base.F_CB_TICK)
    cb.set_label('Kurtosis ratio: MLP / naive-independence', fontsize=base.F_CB_LABEL)

    contour_levels = [lv for lv in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
                      if vmin_l <= np.log2(lv) <= vmax_l]
    base._add_contours(ax, sx[ok], sy[ok], ratio[ok],
                        xlim, ylim, contour_levels, colors='black', fmt='%g')

    ax.set_xlabel('Mean hourly  E[X]  (mm/h)', fontsize=base.F_LABEL)
    ax.set_ylabel(r'$\sigma(x)$  (mm/h)', fontsize=base.F_LABEL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    base._set_sqrt_ticks(ax, xlim, ylim)

    out_path = os.path.join(SCRIPT_DIR, f'mlp_kurtosis_ratio_lead{clead}h.png')
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
    print(f'Lead {clead} h  |  device {device}')

    features, _ = base.load_data(clead)

    ey = base.hourly_expected_value(features)
    x  = ey.mean(axis=1)
    y  = ey.std(axis=1)

    keep = x >= base.MIN_MEAN_EY
    print(f'Retaining {keep.sum():,}/{len(keep):,} samples '
          f'(mean E[X] >= {base.MIN_MEAN_EY} mm/h)')
    features, x, y = features[keep], x[keep], y[keep]

    model, feat_mean, feat_std = base.load_mlp(clead, device)
    print('Running MLP forward pass ...')
    fz6, mw6, s1_6, sc1_6, s2_6, sc2_6 = base.run_mlp_batched(
        model, feat_mean, feat_std, features, device)
    fz6, mw6 = fz6.astype(np.float64), mw6.astype(np.float64)
    s1_6, sc1_6 = s1_6.astype(np.float64), sc1_6.astype(np.float64)
    s2_6, sc2_6 = s2_6.astype(np.float64), sc2_6.astype(np.float64)

    print('Computing analytic kurtosis (MLP distribution) ...')
    kurt_mlp, var_mlp = mlp_kurtosis(fz6, mw6, s1_6, sc1_6, s2_6, sc2_6)

    print('Computing analytic kurtosis (naive-independence sum via cumulants) ...')
    kurt_naive, var_naive = naive_sum_kurtosis(features)

    ratio = kurt_mlp / kurt_naive
    ok = (np.isfinite(ratio) & (ratio > 0)
          & (var_mlp > MIN_VAR) & (var_naive > MIN_VAR))
    print(f'  Retained {ok.sum():,}/{len(ok):,} samples after variance floor '
          f'({100*ok.mean():.1f}%)')
    print(f'  Ratio MLP/naive kurtosis — median {np.median(ratio[ok]):.3f}  '
          f'p10={np.percentile(ratio[ok], 10):.3f}  '
          f'p90={np.percentile(ratio[ok], 90):.3f}')

    print('Generating figure ...')
    out_path = make_figure(x, y, ratio, ok, clead)
    print(f'\nDone.  Figure written to:\n  {out_path}')


if __name__ == '__main__':
    main()
