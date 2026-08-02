#!/usr/bin/env python3
"""
performance_diagram_6hourly_mlp_3panel.py <clead>

e.g.,
    python performance_diagram_6hourly_mlp_3panel.py 12

3-panel performance diagram (POD vs. Success Ratio, shaded CSI contours,
dashed frequency-bias lines) comparing the six-hourly MLP against the
independence-assumption ensemble control, for the three verification
thresholds (0.25, 2.5, 10.0 mm) used throughout the six-hourly MLP
evaluation.  Panels mirror the layout/labeling of
reliability_6hourly_mlp_3panel.py's 3-panel reliability figure.

Reuses the cumulative-contingency-table approach and CSI/FB background from
plot_performance_diagram_terrain.py, and the cached per-threshold contab /
contab_control arrays already written by reliability_6hourly_mlp_3panel.py
(no new data collection -- this only reads the existing .cPick cache, so it
requires that script to have been run first for the requested lead time).

Not yet wired into method_6h.tex -- standalone figure for review.

Tom Hamill, Jul 2026
"""

import os
import sys
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

from reliability_6hourly_mlp_3panel import get_paths, build_test_datelist
from plot_performance_diagram_terrain import pod_sr_from_contab, draw_performance_background

COLOR_CONTROL = 'red'
COLOR_MLP     = 'RoyalBlue'

PANEL_LABELS = [
    r'(a) $\geq$ 0.25 mm/6h',
    r'(b) $\geq$ 2.5 mm/6h',
    r'(c) $\geq$ 10 mm/6h',
]

CSI_LEVELS = np.linspace(0, 1, 11)
CMAP = 'Blues'


def main():
    if len(sys.argv) != 2:
        print('Usage: python performance_diagram_6hourly_mlp_3panel.py <clead>')
        sys.exit(1)

    clead = int(sys.argv[1])

    _, _, relia_dir, _ = get_paths()

    cyyyymmddhh_list = build_test_datelist()
    date_range = f'{cyyyymmddhh_list[0]}_to_{cyyyymmddhh_list[-1]}'
    pick_fname = os.path.join(
        relia_dir, f'relia_6h_MLP_3panel_q0.6_{date_range}_lead{clead}h.cPick')

    if not os.path.exists(pick_fname):
        print(f'ERROR: cached statistics not found: {pick_fname}')
        print(f'  Run:  python reliability_6hourly_mlp_3panel.py {clead}')
        sys.exit(1)

    with open(pick_fname, 'rb') as fh:
        d = cPickle.load(fh)

    if 'contab_control' not in d:
        print(f'ERROR: {pick_fname} lacks contab_control; rerun '
              f'reliability_6hourly_mlp_3panel.py {clead} to regenerate the cache.')
        sys.exit(1)

    pthresholds = d['pthresholds']
    contab_mlp     = d['contab']
    contab_control = d['contab_control']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.12, top=0.85, wspace=0.25)

    csi_contour = None
    for idx, (ax, thresh, panel_label) in enumerate(
            zip(axes, pthresholds, PANEL_LABELS)):

        csi_contour = draw_performance_background(ax)

        pod_control, sr_control = pod_sr_from_contab(contab_control[idx])
        pod_mlp,     sr_mlp     = pod_sr_from_contab(contab_mlp[idx])

        ax.plot(sr_control, pod_control, 'o-', color=COLOR_CONTROL,
                linewidth=2, markersize=5, label='Indep. ensemble', zorder=5)
        ax.plot(sr_mlp, pod_mlp, 'o-', color=COLOR_MLP,
                linewidth=2, markersize=5, label='MLP', zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=13.2)
        ax.set_ylabel('Probability of Detection (POD)', fontsize=13.2)
        ax.set_title(panel_label, fontsize=16)
        ax.tick_params(labelsize=10.8)

    axes[0].legend(loc='lower left', fontsize=11, framealpha=0.85)

    cax = fig.add_axes([0.92, 0.12, 0.015, 0.73])
    cbar = fig.colorbar(csi_contour, cax=cax)
    cbar.set_label('Critical Success Index (CSI)', fontsize=12)
    cbar.ax.tick_params(labelsize=9.6)

    fig.suptitle(
        f'Six-hourly MLP vs. independence-assumption control, {clead}-h lead time',
        fontsize=19.2)

    outfile = f'PerfDiagram_6h_MLP_MRMS_3panel_{date_range}_{clead}h.png'
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    main()
