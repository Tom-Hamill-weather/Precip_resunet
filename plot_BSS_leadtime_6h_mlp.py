#!/usr/bin/env python3
"""
plot_BSS_leadtime_6h_mlp.py

Six-panel plot of Brier Skill Score vs. lead time for the six-hourly MLP
vs. the independence-assumption control.
Rows: 0.25, 2.5, 10.0 mm thresholds.
Columns: Top 10% roughest terrain, Bottom 90% terrain
(terrain_roughness_graf.py, sigma=14 grid points -- same masks used for the
hourly ResUNet's BSS-vs-lead-time figure, plot_BSS_leadtime.py).
Data read from the per-lead cPickles written by
reliability_6hourly_mlp_terrain.py.

Tom Hamill, Aug 2026
"""

import os
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

from reliability_6hourly_mlp_3panel import get_paths, build_test_datelist

LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
PTHRESHOLDS = [0.25, 2.5, 10.0]
THRESH_LABELS = ['> 0.25 mm', '> 2.5 mm', '> 10 mm']

COLOR_CONTROL = 'red'
COLOR_MLP     = 'RoyalBlue'
LW = 2


def main():
    _, _, relia_dir, _ = get_paths()
    cyyyymmddhh_list = build_test_datelist()
    date_range = f'{cyyyymmddhh_list[0]}_to_{cyyyymmddhh_list[-1]}'

    data = {}
    for lead in LEAD_TIMES:
        fname = os.path.join(
            relia_dir, f'relia_6h_MLP_terrain_q0.6_{date_range}_lead{lead}h.cPick')
        if not os.path.exists(fname):
            print(f'ERROR: missing {fname}')
            print(f'  Run:  python reliability_6hourly_mlp_terrain.py {lead}')
            return
        with open(fname, 'rb') as fh:
            data[lead] = cPickle.load(fh)

    def collect(region, method):
        return np.array([[data[lead]['BSS'][region][method][ti]
                          for ti in range(len(PTHRESHOLDS))]
                         for lead in LEAD_TIMES])

    BSS_mlp_top10        = collect('top10', 'mlp')
    BSS_control_top10    = collect('top10', 'control')
    BSS_mlp_bottom90     = collect('bottom90', 'mlp')
    BSS_control_bottom90 = collect('bottom90', 'control')

    fig, axes = plt.subplots(3, 2, sharey='row', figsize=(6, 7))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91,
                        bottom=0.08, hspace=0.42, wspace=0.07)

    panel_ids = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
    col_names = ['Top 10% roughest terrain', 'Bottom 90% terrain']

    for row in range(3):
        for col in range(2):
            ax = axes[row, col]

            title = (f'{panel_ids[row][col]} {col_names[col]}, '
                     f'{THRESH_LABELS[row]}')
            ax.set_title(title, fontsize=9.5, loc='center')

            if col == 0:
                y_control = BSS_control_top10[:, row]
                y_mlp     = BSS_mlp_top10[:, row]
            else:
                y_control = BSS_control_bottom90[:, row]
                y_mlp     = BSS_mlp_bottom90[:, row]

            ax.axhline(0, color='k', linewidth=1.5, zorder=2)

            ax.plot(LEAD_TIMES, y_control, 'o-', color=COLOR_CONTROL,
                    linewidth=LW, label='Indep. ensemble')
            ax.plot(LEAD_TIMES, y_mlp, 'o-', color=COLOR_MLP,
                    linewidth=LW, label='6-h MLP')

            ax.set_xticks(LEAD_TIMES)
            ax.set_xlim(3, 51)
            ax.tick_params(labelsize=8.8)
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

            if row == 2:
                ax.set_xlabel('Lead time (h)', fontsize=9.9)
            if col == 0:
                ax.set_ylabel('Brier Skill Score', fontsize=9.9)
            if col == 1:
                ax.tick_params(labelleft=False)

            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Six-hourly MLP BSS vs. Lead Time  —  ' + date_range,
                 fontsize=12.5, y=0.975)

    outfile = f'BSS_leadtime_6h_MLP_q0.6_{date_range}.png'
    plt.savefig(outfile, dpi=150)
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    main()
