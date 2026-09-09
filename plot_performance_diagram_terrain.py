#!/usr/bin/env python3
"""
plot_performance_diagram_terrain.py <threshold_mm>

e.g.,
python plot_performance_diagram_terrain.py 0.25

Two-panel performance diagram (POD vs. Success Ratio, shaded CSI contours,
dashed frequency-bias lines -- see ~/fronts/evaluation/performance_diagrams.py
for the reference implementation this is adapted from).

Panels: Top 10% roughest terrain (left), Bottom 90% terrain (right).
Curves: Attention ResUNet (gamma mixture, RoyalBlue) and Smoothed GRAF raw
(Red) -- same colors used throughout this repo's other verification plots --
for lead times of 6, 24, and 48 h (distinguished by line style) plotted
together on the same axes.

POD/SR are built by treating each of the 11 saved probability bins as a
forecast-probability cutoff and cumulating contingency-table counts from
that bin to the top (bin 10 = forecast probability 1.0). This is the same
per-bin contingency table used for the reliability diagrams, just read
cumulatively instead of per-bin -- no new data collection is needed beyond
the contab_*_top10/bottom90 arrays already saved by
reliability_resunet_mixture.py.
"""

import os
import sys
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

RELIA_DIR  = '/data/resnet_data/relia'
DATE_RANGE = '2025030100_to_2025123118'
LEADS = ['6', '24', '48']
LEAD_LINESTYLES = {'6': '-', '24': '--', '48': ':'}

COLOR_RAW   = 'Red'
COLOR_GAMMA = 'RoyalBlue'

FB_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
CSI_LEVELS = np.linspace(0, 1, 11)
CMAP = 'Blues'
MIN_COUNT = 1000

# Highest forecast-probability bin (%) to plot, by threshold (mm). Bins above
# this are dropped -- for 5 mm the >=70% bins are so rarely used that they
# produce an erratic, non-monotonic hook in POD/SR space (see discussion in
# conversation). Thresholds not listed here plot the full 0-100% range.
MAX_PROB_PCT_BY_THRESH = {5.0: 50.0}


def pod_sr_from_contab(contab, min_count=MIN_COUNT):
    """
    contab: (ncats, 2) array, column 0 = nonevent counts, column 1 = event
    counts, per forecast-probability bin (bin k <-> probability k/(ncats-1)).

    Returns (pod, sr), each length ncats, built by cumulating from bin k to
    the top bin so that index k represents the contingency table you'd get
    by calling "forecast yes" for every bin >= k. Bins where the cumulative
    number of forecasts (tp+fp) is below min_count are left as nan, since
    the resulting point is dominated by sampling noise rather than signal
    (this shows up as an erratic hook back toward the origin at the
    high-confidence tail when a bin is rarely or never used).
    """
    ncats = contab.shape[0]
    nonevent = contab[:, 0].astype(float)
    event = contab[:, 1].astype(float)

    pod = np.full(ncats, np.nan)
    sr = np.full(ncats, np.nan)
    for k in range(ncats):
        tp = event[k:].sum()
        fp = nonevent[k:].sum()
        fn = event[:k].sum()
        if (tp + fp) < min_count:
            continue
        if (tp + fn) > 0:
            pod[k] = tp / (tp + fn)
        sr[k] = tp / (tp + fp)
    return pod, sr


def draw_performance_background(ax):
    sr_matrix, pod_matrix = np.meshgrid(np.linspace(0.001, 1, 101),
                                         np.linspace(0.001, 1, 101))
    csi_matrix = 1.0 / ((1.0 / sr_matrix) + (1.0 / pod_matrix) - 1.0)
    fb_matrix = pod_matrix * (sr_matrix ** -1)

    csi_contour = ax.contourf(sr_matrix, pod_matrix, csi_matrix, CSI_LEVELS,
                               cmap=CMAP)
    cs = ax.contour(sr_matrix, pod_matrix, fb_matrix, FB_LEVELS,
                     colors='black', linewidths=0.5, linestyles='--')
    ax.clabel(cs, FB_LEVELS, fontsize=8.4)
    return csi_contour


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_performance_diagram_terrain.py <threshold_mm>")
        print("Example: python plot_performance_diagram_terrain.py 0.25")
        sys.exit(1)

    threshold_mm = float(sys.argv[1])

    data_by_lead = {}
    thresh_used = None
    for clead in LEADS:
        fname = os.path.join(RELIA_DIR,
            f'relia_GRAF_ResUNet_Mixture_q0.5_{DATE_RANGE}_lead{clead}h.cPick')
        with open(fname, 'rb') as fh:
            data = cPickle.load(fh)

        pthresholds = data['pthresholds']
        ithresh = int(np.argmin(np.abs(np.array(pthresholds) - threshold_mm)))
        if abs(pthresholds[ithresh] - threshold_mm) > 0.01:
            print(f"WARNING: requested threshold {threshold_mm} mm not found; "
                  f"using nearest available threshold {pthresholds[ithresh]} mm")
        thresh_used = pthresholds[ithresh]
        data_by_lead[clead] = (data, ithresh)

    regions = [
        ('top10',    '(a) Top 10% roughest terrain'),
        ('bottom90', '(b) Bottom 90% terrain'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.11, top=0.87, wspace=0.18)

    max_prob_pct = MAX_PROB_PCT_BY_THRESH.get(thresh_used, 100.0)

    csi_contour = None
    for ax, (region_key, panel_title) in zip(axes, regions):
        csi_contour = draw_performance_background(ax)

        for clead in LEADS:
            data, ithresh = data_by_lead[clead]
            ls = LEAD_LINESTYLES[clead]
            probability = data['probability']

            contab_raw = data[f'contab_raw_{region_key}'][ithresh]
            contab_gamma = data[f'contab_gamma_{region_key}'][ithresh]

            pod_raw, sr_raw = pod_sr_from_contab(contab_raw)
            pod_gamma, sr_gamma = pod_sr_from_contab(contab_gamma)

            drop = probability > max_prob_pct
            pod_raw[drop] = np.nan
            sr_raw[drop] = np.nan
            pod_gamma[drop] = np.nan
            sr_gamma[drop] = np.nan

            ax.plot(sr_raw, pod_raw, 'o', linestyle=ls, color=COLOR_RAW,
                    linewidth=2, markersize=5,
                    label=f'Smoothed GRAF, {clead}h', zorder=5)
            ax.plot(sr_gamma, pod_gamma, 'o', linestyle=ls, color=COLOR_GAMMA,
                    linewidth=2, markersize=5,
                    label=f'Attention ResUNet, {clead}h', zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=13.2)
        ax.set_ylabel('Probability of Detection (POD)', fontsize=13.2)
        ax.set_title(panel_title, fontsize=15.6)
        ax.tick_params(labelsize=10.8)

    legend_loc = 'upper right' if thresh_used >= 5.0 else 'lower left'
    axes[0].legend(loc=legend_loc, fontsize=10.8, framealpha=0.85)

    cax = fig.add_axes([0.92, 0.11, 0.02, 0.76])
    cbar = fig.colorbar(csi_contour, cax=cax)
    cbar.set_label('Critical Success Index (CSI)', fontsize=12)
    cbar.ax.tick_params(labelsize=9.6)

    fig.suptitle(
        rf'Forecast performance diagram, P(obs $\geq$ {thresh_used} mm)',
        fontsize=16.8)

    leads_tag = '-'.join(LEADS)
    outfile = f'PerfDiagram_terrain_{DATE_RANGE}_{thresh_used}mm_leads{leads_tag}h.png'
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f'Saved: {outfile}')
