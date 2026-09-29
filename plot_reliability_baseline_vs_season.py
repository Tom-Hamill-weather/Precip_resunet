"""
python plot_reliability_baseline_vs_season.py clead [date0_to_date1]

Three-panel reliability diagram (0.25, 1.0, 5.0 mm) comparing the
Attention ResUNet's own postprocessed forecasts across two training
regimes:
  - "Original training" = model_tag 'baseline' (one checkpoint per
    calendar month x 3-h lead, as described in the original manuscript)
  - "Current training"  = model_tag 'season' (season-pooled + FiLM,
    per-pixel solar-hour + precip-climo channels)
Only the postprocessed (gamma-mixture) forecast is shown; the raw/
neighborhood-smoothed-GRAF reference used in plot_reliability_3panel.py
is intentionally omitted.

Arguments:
    clead         : lead time in hours (e.g. 6, 12, 24)
    date0_to_date1: (optional) explicit '<date0>_to_<date1>' tag; if
                    omitted, defaults to '2025010100_to_2025063018' (the
                    2025h1 Jan/Apr/Jun sample, cached for both model
                    tags already).
"""

import sys, os
import _pickle as cPickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

RELIA_DIR = '/data/resnet_data/relia'
PLOT_THRESHOLDS = [0.25, 1.0, 5.0]

clead = sys.argv[1]
date_tag = sys.argv[2] if len(sys.argv) > 2 else '2025010100_to_2025063018'

baseline_file = os.path.join(
    RELIA_DIR, f'relia_GRAF_ResUNet_Mixture_q0.5_{date_tag}_lead{clead}h.cPick')
season_file = os.path.join(
    RELIA_DIR, f'relia_GRAF_ResUNet_Mixture_Season_q0.5_{date_tag}_lead{clead}h.cPick')

for f in (baseline_file, season_file):
    if not os.path.exists(f):
        print(f'Missing reliability file: {f}')
        sys.exit(1)

with open(baseline_file, 'rb') as fh:
    d_baseline = cPickle.load(fh)
with open(season_file, 'rb') as fh:
    d_season = cPickle.load(fh)

pthresholds = d_baseline['pthresholds']
probability = d_baseline['probability']

thresh_idx = []
for t in PLOT_THRESHOLDS:
    matches_t = [i for i, pt in enumerate(pthresholds) if abs(pt - t) < 1e-6]
    if not matches_t:
        print(f'Threshold {t} mm not found (available: {pthresholds})')
        sys.exit(1)
    thresh_idx.append(matches_t[0])

date0, date1 = date_tag.split('_to_')

pan_size = 6.5
fig, axes = plt.subplots(1, 3, figsize=(pan_size * 3 + 1.0, pan_size + 1.0),
    gridspec_kw={'wspace': 0.3})
fig.suptitle(f'{clead}-h forecast reliability: original vs. current training',
             fontsize=24, y=0.98)

for col, (ithresh, thresh) in enumerate(zip(thresh_idx, PLOT_THRESHOLDS)):
    ax = axes[col]

    bss_baseline = d_baseline['BSS_gamma'][ithresh]
    bss_season = d_season['BSS_gamma'][ithresh]
    cbss_baseline = f'{bss_baseline:.2f}' if not np.isnan(bss_baseline) else 'N/A'
    cbss_season = f'{bss_season:.2f}' if not np.isnan(bss_season) else 'N/A'

    label_baseline = f'Original training\nBSS = {cbss_baseline}  n={d_baseline["ngood"]}'
    label_season = f'Current training\nBSS = {cbss_season}  n={d_season["ngood"]}'

    ax.plot([0, 100], [0, 100], '--', color='k', lw=1)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.set_aspect('equal')
    ax.set_xlabel('Forecast probability (%)', fontsize=18)
    ax.set_ylabel('Observed relative frequency (%)', fontsize=18)
    ax.tick_params(labelsize=13)
    panel_letter = 'abc'[col]
    ax.set_title(f'({panel_letter}) ' + r'P(obs $\geq$ ' + str(thresh) + ' mm)', fontsize=21)

    for imodel, (d, color, label) in enumerate([
            (d_baseline, 'DarkOrange', label_baseline),
            (d_season, 'RoyalBlue', label_season),
    ]):
        relia = d['relia_gamma'][ithresh]
        frequse = d['frequse_gamma'][ithresh]
        relia_ma = ma.masked_where(relia < -99., relia)
        ax.plot(probability, 100. * relia_ma, 'o-',
                color=color, linewidth=3, markersize=9, label=label)

        if imodel == 0:
            a2 = ax.inset_axes([0.09, 0.68, 0.44, 0.27])
            a2.bar(probability - 1.5, frequse, width=1.5, bottom=1e-5,
                   log=True, color=color, edgecolor='None', align='center')
            a2.set_xlim(-5, 105)
            a2.set_ylim(1e-5, 1.)
            a2.set_title('Frequency of usage', fontsize=10)
            a2.set_xlabel('Forecast probability', fontsize=11)
            a2.tick_params(labelsize=8)
            a2.hlines([1e-4, 1e-3, 1e-2, 0.1], 0, 100,
                      linestyles='dashed', colors='gray', lw=0.5)
        else:
            a2.bar(probability, frequse, width=1.5, bottom=1e-5,
                   log=True, color=color, edgecolor='None', align='center')

    ax.legend(loc='lower right', fontsize=12)

plt.subplots_adjust(top=0.88, bottom=0.1, left=0.06, right=0.98, wspace=0.3)

outfile = os.path.join(
    RELIA_DIR, f'Relia_3panel_baseline_vs_season_{date0}_to_{date1}_lead{clead}h.png')
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(f'Saved to {outfile}')
