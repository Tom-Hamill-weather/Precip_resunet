"""
python plot_patch_scatter_dec2025.py

Scatter (2D density) plots of GRAF forecast vs MRMS observed precipitation
from December 2025 training patches, for 6h and 12h lead times.
Only pixels where GRAF > 2 mm and MRMS quality > 0.5 are shown.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys
sys.path.insert(0, '/home/thamill/resnet')
from data_loader_utils import load_training_data

PATCH_DIR = '/data/resnet_data/patch_data'
SPLITS    = ['train', 'test', 'predict']
INIT_DATE = '2025120100'
GRAF_THRESH = 2.0    # mm  — only plot pixels where GRAF exceeds this
QUAL_THRESH = 0.5    # MRMS quality threshold

# ---------------------------------------------------------------

def load_lead(lead_str):
    """Load and concatenate all splits for a given lead time string (e.g. '6h')."""
    graf_all, mrms_all, qual_all = [], [], []
    for split in SPLITS:
        fname = f'GRAF_Unet_data_{split}_{INIT_DATE}_{lead_str}.nc'
        fpath = os.path.join(PATCH_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  Missing: {fpath}')
            continue
        print(f'  Loading {fname} ...', end=' ')
        d = load_training_data(fpath)
        graf_all.append(d['GRAF'].ravel())
        mrms_all.append(d['MRMS'].ravel())
        qual_all.append(d['MRMS_qual'].ravel())
        print(f'{d["GRAF"].shape[0]} patches')
    if not graf_all:
        return None, None
    graf = np.concatenate(graf_all)
    mrms = np.concatenate(mrms_all)
    qual = np.concatenate(qual_all)

    # Apply quality mask and GRAF threshold
    mask = (qual > QUAL_THRESH) & (graf > GRAF_THRESH) & (mrms >= 0.)
    print(f'  {mask.sum():,} pixels after filtering '
          f'(GRAF>{GRAF_THRESH}mm, qual>{QUAL_THRESH})')
    return graf[mask], mrms[mask]

# ---------------------------------------------------------------

print('Loading 6h patches ...')
graf_6,  mrms_6  = load_lead('6h')
print('Loading 12h patches ...')
graf_12, mrms_12 = load_lead('12h')

# ---------------------------------------------------------------
# Plot

MAXVAL = 10.    # axis limit
BINS   = 100    # number of bins per axis

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle('Dec 2025 patch data: GRAF forecast vs MRMS observed\n'
             f'(pixels with GRAF > {GRAF_THRESH} mm, good MRMS quality)',
             fontsize=13)

def panel(ax, graf, mrms, title):
    if graf is None:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return

    # Square-root transform for display
    sg = np.sqrt(graf)
    sm = np.sqrt(mrms)
    smaxval = np.sqrt(MAXVAL)

    h, xedges, yedges = np.histogram2d(
        sg, sm,
        bins=BINS,
        range=[[0, smaxval], [0, smaxval]]
    )

    # Mask empty bins so they show white
    h = np.ma.masked_where(h == 0, h)

    im = ax.pcolormesh(xedges, yedges, h.T,
                       norm=mcolors.LogNorm(vmin=1, vmax=h.max()),
                       cmap='plasma')
    plt.colorbar(im, ax=ax, label='Pixel count')

    # 1:1 line in sqrt space
    ax.plot([0, smaxval], [0, smaxval], 'k--', lw=1, label='1:1')

    # Linear regression in sqrt space
    coeffs = np.polyfit(sg, sm, 1)
    xfit = np.array([0., smaxval])
    ax.plot(xfit, np.polyval(coeffs, xfit), 'w-', lw=1.5,
            label=f'fit: slope={coeffs[0]:.2f}')

    ax.set_xlim(0, smaxval)
    ax.set_ylim(0, smaxval)

    # Replace sqrt-space tick labels with original mm values
    ticks_mm = [0, 1, 2, 5, 10, 20, 40]
    ticks_mm = [t for t in ticks_mm if t <= MAXVAL]
    tick_pos = np.sqrt(ticks_mm)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(t) for t in ticks_mm])
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(t) for t in ticks_mm])

    ax.set_xlabel('GRAF forecast precipitation (mm)', fontsize=11)
    ax.set_ylabel('MRMS observed precipitation (mm)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')

    # Stats computed on original mm values
    bias = np.mean(mrms - graf)
    rmse = np.sqrt(np.mean((mrms - graf)**2))
    corr = np.corrcoef(graf, mrms)[0, 1]
    stats = (f'n = {len(graf):,}\n'
             f'bias = {bias:+.2f} mm\n'
             f'RMSE = {rmse:.2f} mm\n'
             f'r = {corr:.3f}')
    ax.text(0.97, 0.05, stats, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

panel(axes[0], graf_6,  mrms_6,
      f'(a) 6-h lead, Dec 2025 init ({INIT_DATE})')
panel(axes[1], graf_12, mrms_12,
      f'(b) 12-h lead, Dec 2025 init ({INIT_DATE})')

plt.tight_layout()
outfile = f'/data/resnet_data/plots/patch_scatter_6h_vs_12h_{INIT_DATE}.png'
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(f'\nSaved to {outfile}')
