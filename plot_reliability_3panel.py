"""
python plot_reliability_3panel.py clead [relia_file]

Three-panel reliability diagram for 0.25, 1.0, and 5.0 mm thresholds,
read from a pre-saved cPickle file produced by reliability_resunet_mixture.py.

Arguments:
    clead      : lead time in hours (e.g. 6, 12, 24)
    relia_file : (optional) explicit path to cPickle file; if omitted the
                 most recent matching file in /data/resnet_data/relia/ is used
"""

import sys, os, glob
import _pickle as cPickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
RELIA_DIR = '/data/resnet_data/relia'
PLOT_THRESHOLDS = [0.25, 1.0, 5.0]   # must be in pthresholds of the saved file
# ---------------------------------------------------------------

clead = sys.argv[1]

if len(sys.argv) >= 3:
    relia_file = sys.argv[2]
else:
    # Find latest file matching this lead time (prefer _to_2025123118_, fall back)
    pattern = os.path.join(RELIA_DIR,
        f'relia_GRAF_ResUNet_Mixture_q0.5_*_lead{clead}h.cPick')
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f'No reliability file found for lead={clead}h in {RELIA_DIR}')
        sys.exit(1)
    relia_file = matches[-1]   # alphabetically latest = most recent date range

print(f'Loading {relia_file}')
with open(relia_file, 'rb') as fh:
    d = cPickle.load(fh)

pthresholds  = d['pthresholds']
probability  = d['probability']        # (ncats,) bin centres, 0-100
relia_raw    = d['relia_raw']          # (nthresh, ncats)
relia_gamma  = d['relia_gamma']
frequse_raw  = d['frequse_raw']
frequse_gamma= d['frequse_gamma']
BSS_raw      = d['BSS_raw']            # (nthresh,)
BSS_gamma    = d['BSS_gamma']
ngood        = d['ngood']

# Map requested thresholds to row indices
thresh_idx = []
for t in PLOT_THRESHOLDS:
    matches_t = [i for i, pt in enumerate(pthresholds) if abs(pt - t) < 1e-6]
    if not matches_t:
        print(f'Threshold {t} mm not found in file (available: {pthresholds})')
        sys.exit(1)
    thresh_idx.append(matches_t[0])

# ---------------------------------------------------------------
# Extract date range label from filename
fname = os.path.basename(relia_file)
# e.g. relia_GRAF_ResUNet_Mixture_q0.5_2025030100_to_2025123118_lead6h.cPick
try:
    parts = fname.replace('.cPick','').split('_')
    date0 = parts[5]   # 2025030100
    date1 = parts[7]   # 2025123118
    date_label = f'{date0} to {date1}'
except Exception:
    date_label = ''

# ---------------------------------------------------------------

# Each panel 5"×5" square, with space for ylabel on left and suptitle on top
pan_size = 5.0
fig, axes = plt.subplots(1, 3, figsize=(pan_size * 3 + 1.0, pan_size))
fig.suptitle(f'{clead}-h forecast reliability',
             fontsize=18)

for col, (ithresh, thresh) in enumerate(zip(thresh_idx, PLOT_THRESHOLDS)):
    ax = axes[col]

    bss_r = BSS_raw[ithresh]
    bss_g = BSS_gamma[ithresh]
    cbss_r = f'{bss_r:.2f}' if not np.isnan(bss_r) else 'N/A'
    cbss_g = f'{bss_g:.2f}' if not np.isnan(bss_g) else 'N/A'

    label_raw   = f'Smoothed GRAF raw,  BSS = {cbss_r}'
    label_gamma = f'Attention ResUNet,  BSS = {cbss_g}'

    ax.plot([0, 100], [0, 100], '--', color='k', lw=1)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.set_aspect('equal')
    ax.set_xlabel('Forecast probability (%)', fontsize=14)
    ax.set_ylabel('Observed relative frequency (%)', fontsize=14)
    ax.tick_params(labelsize=13)
    panel_letter = 'abc'[col]
    ax.set_title(f'({panel_letter}) ' + r'P(obs $\geq$ ' + str(thresh) + ' mm)', fontsize=17)

    for imodel, (relia, frequse, color, label) in enumerate([
            (relia_raw[ithresh],   frequse_raw[ithresh],   'Red',       label_raw),
            (relia_gamma[ithresh], frequse_gamma[ithresh], 'RoyalBlue', label_gamma),
    ]):
        relia_ma = ma.masked_where(relia < -99., relia)
        ax.plot(probability, 100. * relia_ma, 'o-',
                color=color, linewidth=2, label=label)

        # Frequency-of-use inset: upper-left, no y-label to avoid overlap with main axis
        if imodel == 0:
            a2 = ax.inset_axes([0.09, 0.68, 0.44, 0.27])
            a2.bar(probability - 1.5, frequse, width=1.5, bottom=1e-5,
                   log=True, color=color, edgecolor='None', align='center')
            a2.set_xlim(-5, 105)
            a2.set_ylim(1e-5, 1.)
            a2.set_title('Frequency of usage', fontsize=9)
            a2.set_xlabel('Forecast probability', fontsize=8)
            a2.tick_params(labelsize=7)
            a2.hlines([1e-4, 1e-3, 1e-2, 0.1], 0, 100,
                      linestyles='dashed', colors='gray', lw=0.5)
        else:
            a2.bar(probability, frequse, width=1.5, bottom=1e-5,
                   log=True, color=color, edgecolor='None', align='center')

    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()

outfile = os.path.join(RELIA_DIR,
    f'Relia_3panel_{date0}_to_{date1}_lead{clead}h.png')
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(f'Saved to {outfile}')
