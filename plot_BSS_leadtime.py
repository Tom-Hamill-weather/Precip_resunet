#!/usr/bin/env python3
"""
plot_BSS_leadtime.py

Six-panel plot of Brier Skill Score vs. lead time.
Rows: 0.25, 1.0, 5.0 mm thresholds.
Columns: CONUS, Western US (west of -105 lon).
Data read from cPickled reliability files (q0.5).
"""

import os
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

RELIA_DIR  = '/data/resnet_data/relia'
DATE_RANGE = '2025030100_to_2025123112'
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]

# Indices into pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
PLOT_THRESHOLDS = [0.25, 1.0, 5.0]
THRESH_IDX      = [0, 1, 3]
THRESH_LABELS   = ['> 0.25 mm', '> 1 mm', '> 5 mm']

# ── load one cPickle per lead time ────────────────────────────────────────

data = {}
for lead in LEAD_TIMES:
    fname = os.path.join(RELIA_DIR,
        f'relia_GRAF_ResUNet_Mixture_q0.5_{DATE_RANGE}_lead{lead}h.cPick')
    with open(fname, 'rb') as fh:
        data[lead] = cPickle.load(fh)

# Build (n_leads, n_thresholds_plotted) arrays
def collect(key):
    return np.array([[data[lead][key][ti] for ti in THRESH_IDX]
                     for lead in LEAD_TIMES])

BSS_raw        = collect('BSS_raw')        # (5, 3)
BSS_gamma      = collect('BSS_gamma')
BSS_raw_west   = collect('BSS_raw_west')
BSS_gamma_west = collect('BSS_gamma_west')

# ── plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, sharey='row', figsize=(6, 7))
fig.subplots_adjust(left=0.13, right=0.97, top=0.91,
                    bottom=0.08, hspace=0.42, wspace=0.07)

COLOR_RAW   = 'Red'
COLOR_GAMMA = 'RoyalBlue'
LW = 2

panel_ids = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
col_names = ['CONUS', 'Western US']

for row in range(3):
    for col in range(2):
        ax = axes[row, col]

        title = (f'{panel_ids[row][col]} {col_names[col]}, '
                 f'{THRESH_LABELS[row]}')
        ax.set_title(title, fontsize=9.5, loc='center')

        if col == 0:
            y_raw   = BSS_raw[:, row]
            y_gamma = BSS_gamma[:, row]
        else:
            y_raw   = BSS_raw_west[:, row]
            y_gamma = BSS_gamma_west[:, row]

        # BSS = 0 reference
        ax.axhline(0, color='k', linewidth=1.5, zorder=2)

        ax.plot(LEAD_TIMES, y_raw,   'o-', color=COLOR_RAW,
                linewidth=LW, label='Smoothed GRAF')
        ax.plot(LEAD_TIMES, y_gamma, 'o-', color=COLOR_GAMMA,
                linewidth=LW, label='Attention ResUNet')

        ax.set_xticks(LEAD_TIMES)
        ax.set_xlim(3, 51)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

        if row == 2:
            ax.set_xlabel('Lead time (h)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Brier Skill Score', fontsize=9)
        if col == 1:
            ax.tick_params(labelleft=False)

        # legend in top-right panel only
        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc='lower left')

fig.suptitle('Brier Skill Score vs. Lead Time  —  Mar–Dec 2025',
             fontsize=11, y=0.975)

outfile = f'BSS_leadtime_q0.5_{DATE_RANGE}.png'
plt.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')
