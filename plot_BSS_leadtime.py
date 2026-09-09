#!/usr/bin/env python3
"""
plot_BSS_leadtime.py

Six-panel plot of Brier Skill Score vs. lead time.
Rows: 0.25, 1.0, 5.0 mm thresholds.
Columns: Top 10% roughest terrain, Bottom 90% terrain
(terrain_roughness_graf.py, sigma=14 grid points, naive 90th-percentile
candidate set -- see GRAF_TERRAIN_BSS_ADAPTATION_GUIDE.md).
Data read from cPickled reliability files (q0.5).
"""

import os
import glob
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

RELIA_DIR  = '/data/resnet_data/relia'
DATE_RANGE = '2025030100_to_2025123118'
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
SHOW_NOTERRAIN_DOT = False

# Lead time at which the no-terrain ablation dots are added (column-1 panels,
# i.e. the slot the "Western US" column used to occupy before the terrain-
# roughness columns replaced it -- the dot itself is still the West-of-105
# BSS_gamma from reliability_resunet_mixture_noterrain.py, which was not
# re-run with a top10/bottom90 split; see GRAF_TERRAIN_BSS_ADAPTATION_GUIDE.md)
NOTERRAIN_LEAD = 24

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

BSS_raw_top10      = collect('BSS_raw_top10')        # (5, 3)
BSS_gamma_top10    = collect('BSS_gamma_top10')
BSS_raw_bottom90   = collect('BSS_raw_bottom90')
BSS_gamma_bottom90 = collect('BSS_gamma_bottom90')

# ── load the NO-TERRAIN ablation file (single lead, West-of-105 dots) ──────
# Produced by reliability_resunet_mixture_noterrain.py at NOTERRAIN_LEAD.
# Still West-of-105-stratified (that script wasn't ported to top10/bottom90);
# plotted below in the column-1 slot alongside the Bottom-90%-terrain curves.
# Filename date-range may differ from DATE_RANGE, so glob for it.
noterrain_west = None  # (n_thresholds_plotted,) BSS for no-terrain model, west
nt_glob = os.path.join(
    RELIA_DIR,
    f'relia_GRAF_ResUNet_Mixture_noterrain_q0.5_*_lead{NOTERRAIN_LEAD}h.cPick')
nt_files = sorted(glob.glob(nt_glob)) if SHOW_NOTERRAIN_DOT else []
if nt_files:
    with open(nt_files[-1], 'rb') as fh:
        nt = cPickle.load(fh)
    noterrain_west = np.array([nt['BSS_gamma_west'][ti] for ti in THRESH_IDX])
    print(f'Loaded no-terrain BSS from: {nt_files[-1]}')
    print(f'  Western US BSS_gamma (no terrain) at {NOTERRAIN_LEAD}h: {noterrain_west}')
else:
    print(f'No no-terrain file found (pattern: {nt_glob}); skipping black dots.')

# ── plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, sharey='row', figsize=(6, 7))
fig.subplots_adjust(left=0.13, right=0.97, top=0.91,
                    bottom=0.08, hspace=0.42, wspace=0.07)

COLOR_RAW   = 'Red'
COLOR_GAMMA = 'RoyalBlue'
LW = 2

panel_ids = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
col_names = ['Top 10% roughest terrain', 'Bottom 90% terrain']

for row in range(3):
    for col in range(2):
        ax = axes[row, col]

        title = (f'{panel_ids[row][col]} {col_names[col]}, '
                 f'{THRESH_LABELS[row]}')
        ax.set_title(title, fontsize=9.5, loc='center')

        if col == 0:
            y_raw   = BSS_raw_top10[:, row]
            y_gamma = BSS_gamma_top10[:, row]
        else:
            y_raw   = BSS_raw_bottom90[:, row]
            y_gamma = BSS_gamma_bottom90[:, row]

        # BSS = 0 reference
        ax.axhline(0, color='k', linewidth=1.5, zorder=2)

        ax.plot(LEAD_TIMES, y_raw,   'o-', color=COLOR_RAW,
                linewidth=LW, label='Smoothed GRAF')
        ax.plot(LEAD_TIMES, y_gamma, 'o-', color=COLOR_GAMMA,
                linewidth=LW, label='Attention ResUNet')

        # No-terrain ablation: black dot at NOTERRAIN_LEAD, column-1 panels only
        # (West-of-105 BSS_gamma, not Bottom-90%-terrain -- see note above)
        if col == 1 and noterrain_west is not None:
            ax.plot(NOTERRAIN_LEAD, noterrain_west[row], 'o',
                    color='black', markersize=7, zorder=6,
                    label='Attention ResUNet w/o terrain')

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

        # legend in top-right panel only
        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc='lower left')

fig.suptitle('Brier Skill Score vs. Lead Time  —  Mar–Dec 2025',
             fontsize=13.2, y=0.975)

outfile = f'BSS_leadtime_q0.5_{DATE_RANGE}.png'
plt.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')
