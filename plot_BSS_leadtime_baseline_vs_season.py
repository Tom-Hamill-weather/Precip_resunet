#!/usr/bin/env python3
"""
plot_BSS_leadtime_baseline_vs_season.py [date_range]

Six-panel plot of Brier Skill Score vs. lead time, comparing the
Attention ResUNet's OWN forecasts across two training regimes:
  - "Original training" = model_tag 'baseline' (one checkpoint per
    calendar month x 3-h lead, as described in the original manuscript)
  - "Current training"  = model_tag 'season' (season-pooled + FiLM,
    per-pixel solar-hour + precip-climo channels)
Both curves are BSS_gamma (the postprocessed ResUNet probabilities);
the raw/neighborhood-smoothed-GRAF curve (BSS_raw) used in
plot_BSS_leadtime.py is intentionally omitted -- this figure is about
what changed in the ResUNet's own output across the two trainings, not
about the ResUNet vs. a raw-GRAF reference.

Rows: 0.25, 1.0, 5.0 mm thresholds.
Columns: Top 10% roughest terrain, Bottom 90% terrain.
Data read from cPickled reliability files (q0.5) written by
reliability_resunet_mixture.py for model_tag in {baseline, season}.

Shaded bands around each curve are 90% Hodges-Lehmann confidence
intervals on that model's own per-day BSS_gamma distribution -- a
rank-based nonparametric CI obtained by inverting the (one-sample)
Wilcoxon signed-rank statistic (see compute_bss_ci_baseline_vs_season.py),
NOT a bootstrap resample.

Usage:
    python plot_BSS_leadtime_baseline_vs_season.py [date_range]
    date_range defaults to '2026010100_to_2026083118' (the full daily
    2026 Jan-Aug record, backfilled for both model tags). Requires the
    matching CI pickle from compute_bss_ci_baseline_vs_season.py to
    already exist for the same date_range.
"""

import os
import sys
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

RELIA_DIR  = '/data/resnet_data/relia'
DATE_RANGE = sys.argv[1] if len(sys.argv) > 1 else '2026010100_to_2026083118'
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]

# Indices into pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
PLOT_THRESHOLDS = [0.25, 1.0, 5.0]
THRESH_IDX      = [0, 1, 3]
THRESH_LABELS   = ['> 0.25 mm', '> 1 mm', '> 5 mm']

MODEL_TAGS = {
    'baseline': 'ResUNet_Mixture',
    'season':   'ResUNet_Mixture_Season',
}

# ── load one cPickle per lead time, per model tag ──────────────────────────

data = {tag: {} for tag in MODEL_TAGS}
for tag, out_model_name in MODEL_TAGS.items():
    for lead in LEAD_TIMES:
        fname = os.path.join(RELIA_DIR,
            f'relia_GRAF_{out_model_name}_q0.5_{DATE_RANGE}_lead{lead}h.cPick')
        if not os.path.exists(fname):
            print(f'Missing: {fname}')
            sys.exit(1)
        with open(fname, 'rb') as fh:
            data[tag][lead] = cPickle.load(fh)


def collect(tag, key):
    return np.array([[data[tag][lead][key][ti] for ti in THRESH_IDX]
                     for lead in LEAD_TIMES])

BSS_top10    = {tag: collect(tag, 'BSS_gamma_top10')    for tag in MODEL_TAGS}
BSS_bottom90 = {tag: collect(tag, 'BSS_gamma_bottom90')  for tag in MODEL_TAGS}

# ── load Hodges-Lehmann CI bands (per model_tag, lead, region, threshold) ──

ci_fname = os.path.join(RELIA_DIR, f'bss_ci_baseline_vs_season_{DATE_RANGE}.cPick')
if not os.path.exists(ci_fname):
    print(f'Missing CI file: {ci_fname} -- run compute_bss_ci_baseline_vs_season.py first')
    sys.exit(1)
with open(ci_fname, 'rb') as fh:
    ci_data = cPickle.load(fh)

def ci_band(tag, region, thresh, y):
    """Hodges-Lehmann band, recentered on the plotted (pooled, ratio-of-
    sums) BSS score y. The HL estimate itself targets the median of the
    per-day BSS_gamma ratios, which is a different -- generally biased
    relative to the pooled ratio via Jensen's inequality on a per-day
    ratio statistic -- quantity than the plotted pooled score; using its
    raw [ci_lo, ci_hi] would produce a band not centered on the curve it
    shades. The asymmetric half-widths (hl_est - ci_lo), (ci_hi - hl_est)
    still capture the rank-based sampling uncertainty and are shifted
    onto the actual plotted value instead."""
    hl = np.array([ci_data[(tag, lead, region, thresh)]['hl_est'] for lead in LEAD_TIMES])
    lo = np.array([ci_data[(tag, lead, region, thresh)]['ci_lo'] for lead in LEAD_TIMES])
    hi = np.array([ci_data[(tag, lead, region, thresh)]['ci_hi'] for lead in LEAD_TIMES])
    return y - (hl - lo), y + (hi - hl)

# ── plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, sharey='row', figsize=(6, 7))
fig.subplots_adjust(left=0.13, right=0.97, top=0.91,
                    bottom=0.08, hspace=0.42, wspace=0.07)

COLOR_BASELINE = 'DarkOrange'
COLOR_SEASON   = 'RoyalBlue'
LW = 2

panel_ids = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
col_names = ['Top 10% roughest terrain', 'Bottom 90% terrain']

for row in range(3):
    for col in range(2):
        ax = axes[row, col]

        title = (f'{panel_ids[row][col]} {col_names[col]}, '
                 f'{THRESH_LABELS[row]}')
        ax.set_title(title, fontsize=9.5, loc='center')

        BSS = BSS_top10 if col == 0 else BSS_bottom90
        region = 'top10' if col == 0 else 'bottom90'
        thresh = PLOT_THRESHOLDS[row]
        y_baseline = BSS['baseline'][:, row]
        y_season   = BSS['season'][:, row]

        ax.axhline(0, color='k', linewidth=1.5, zorder=2)

        lo_b, hi_b = ci_band('baseline', region, thresh, y_baseline)
        lo_s, hi_s = ci_band('season', region, thresh, y_season)
        ax.fill_between(LEAD_TIMES, lo_b, hi_b, color=COLOR_BASELINE,
                         alpha=0.22, lw=0, zorder=1)
        ax.fill_between(LEAD_TIMES, lo_s, hi_s, color=COLOR_SEASON,
                         alpha=0.22, lw=0, zorder=1)

        ax.plot(LEAD_TIMES, y_baseline, 'o-', color=COLOR_BASELINE,
                linewidth=LW, label='Original training', zorder=3)
        ax.plot(LEAD_TIMES, y_season, 'o-', color=COLOR_SEASON,
                linewidth=LW, label='Current training', zorder=3)

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

date0, date1 = DATE_RANGE.split('_to_')
fig.suptitle('Attention ResUNet Brier Skill Score vs. Lead Time',
             fontsize=12.5, y=0.985)

outfile = f'BSS_leadtime_baseline_vs_season_q0.5_{DATE_RANGE}.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f'Saved: {outfile}')
