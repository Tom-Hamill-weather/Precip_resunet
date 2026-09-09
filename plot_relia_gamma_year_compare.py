"""
python plot_relia_gamma_year_compare.py

Plots the Attention ResUNet (gamma) reliability curve for matched
Jan/Apr/Jun 2025 vs. 2026 (baseline model, 24h lead) on the same axes,
one PNG per threshold -- the year-over-year companion to
reliability_resunet_mixture.py's per-year raw-vs-gamma plots. Reuses the
cPickle files already produced by that script; no new inference/reading.
"""
import pickle as cPickle
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RELIA_DIR = '/data/resnet_data/relia'
FILE_PATTERN = {
    2025: RELIA_DIR + '/relia_GRAF_ResUNet_Mixture_q0.5_2025010100_to_2025063018_lead{}h.cPick',
    2026: RELIA_DIR + '/relia_GRAF_ResUNet_Mixture_q0.5_2026010100_to_2026063018_lead{}h.cPick',
}
COLORS = {2025: 'RoyalBlue', 2026: 'Red'}
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]

for clead in LEADS:
    data = {}
    for year, pattern in FILE_PATTERN.items():
        with open(pattern.format(clead), 'rb') as f:
            data[year] = cPickle.load(f)

    pthresholds = data[2025]['pthresholds']
    probability = data[2025]['probability']

    for ithresh, thresh in enumerate(pthresholds):
        ctthresh = str(thresh) + 'mm'
        ctitle = str(clead) + r'-h Attention ResUNet reliability, P(obs $\geq$ ' + ctthresh + ')'

        fig = plt.figure(figsize=(5., 5.))
        a1 = fig.add_axes([.13, .1, .83, .8])
        a1.set_title(ctitle, fontsize=13 * 0.95)
        a1.plot([0, 100], [0, 100], '--', color='k')
        a1.set_ylabel('Observed relative frequency (%)', fontsize=12)
        a1.set_xlabel('Forecast probability (%)', fontsize=12)
        a1.set_ylim(-1, 101)
        a1.set_xlim(-1, 101)

        a2 = fig.add_axes([.26, .63, .34, .18])
        a2.set_title('Frequency of usage', fontsize=9)
        a2.set_xlabel('Forecast probability', fontsize=7)
        a2.set_ylabel('Forecast frequency', fontsize=7)
        a2.set_xlim(-5, 105)
        a2.set_ylim(1e-5, 1.)
        a2.hlines([1e-4, 0.001, .01, .1], 0, 100, linestyles='dashed', colors='gray', lw=0.5)

        for iy, year in enumerate((2025, 2026)):
            d = data[year]
            relia = d['relia_gamma'][ithresh]
            frequse = d['frequse_gamma'][ithresh]
            bss = d['BSS_gamma'][ithresh]
            color = COLORS[year]
            cbss = "%.3f" % bss if not np.isnan(bss) else "N/A"
            label = f'{year} (Jan/Apr/Jun), BSS = {cbss}'

            relia_ma = ma.masked_where(relia < -99., relia)
            a1.plot(probability, 100. * relia_ma, 'o-', color=color,
                    linewidth=2, label=label)

            offset = -1.5 if iy == 0 else 1.5
            a2.bar(probability + offset, frequse, width=1.5, bottom=1e-5,
                   log=True, color=color, edgecolor='None', align='center')

        a1.legend(loc=4, fontsize='small')
        outfile = f'Relia_ResUNet_year_compare_2025_vs_2026_{ctthresh}_{clead}h.png'
        plt.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f'Saved {outfile}')
