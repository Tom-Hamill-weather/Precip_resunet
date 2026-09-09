"""
python plot_hourly_graf_mrms_cdf_comparison.py clead [mmdd_begin] [mmdd_end]

e.g.,

python plot_hourly_graf_mrms_cdf_comparison.py 24
python plot_hourly_graf_mrms_cdf_comparison.py 24 0401 0731

Reads histogram counts saved by save_hourly_graf_mrms_cdf_data.py (for the
same clead and month-day window, default Jan01-Jun30) and plots pooled
empirical CDFs and a Q-Q plot of hourly GRAF forecast precip vs
contemporaneous MRMS observed precip (quality > 0.5 pixels only), for the
given window in 2025 vs 2026.
"""

import sys
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------

def cdf_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return np.cumsum(counts) / total

# ---------------------------------------------------------------------

def make_plot(data, clead, window_label, plot_filename):

    bin_edges = data['bin_edges']
    bin_right_edges = bin_edges[1:-1]           # finite thresholds 0.25 ... 25.00
    bin_right_edges = np.append(bin_right_edges, 25.0)  # overflow bin plotted at 25.0

    fig, axarr = plt.subplots(1, 2, figsize=(11, 8), sharey=True)

    for ax, year in zip(axarr, ['2025', '2026']):
        counts_graf = data[year]['counts_graf']
        counts_mrms = data[year]['counts_mrms']
        cdf_graf = cdf_from_counts(counts_graf)
        cdf_mrms = cdf_from_counts(counts_mrms)
        n_graf = counts_graf.sum()
        n_mrms = counts_mrms.sum()

        ax.plot(bin_right_edges, cdf_graf, color='red', lw=2, label='GRAF')
        ax.plot(bin_right_edges, cdf_mrms, color='blue', lw=2, label='MRMS')
        ax.set_title(f'{year} {window_label}, lead {clead}h\n'
                     f'(n_GRAF={n_graf:.2e}, n_MRMS={n_mrms:.2e}, '
                     f"inits used={data[year]['nused']}/{data[year]['ntotal']})",
                     fontsize=9)
        ax.set_xlabel('Precipitation (mm)')
        ax.set_xlim(0, 25)
        ax.set_ylim(0.94, 1.001)
        ax.grid(True, lw=0.3)
        ax.legend(loc='lower right', fontsize=9)

    axarr[0].set_ylabel('Cumulative probability')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f'Saved {plot_filename}')

# ---------------------------------------------------------------------

def make_qq_plot(data, clead, window_label, plot_filename):

    """ Q-Q plot of MRMS quantile vs GRAF quantile, both years overlaid.
        Quantiles are obtained by inverting the pooled empirical CDFs at
        a common set of probability levels, concentrated toward the
        upper (wet) tail where the two distributions actually differ. """

    bin_edges = data['bin_edges']
    bin_right_edges = bin_edges[1:-1]
    bin_right_edges = np.append(bin_right_edges, 25.0)

    p_levels = np.sort(1.0 - np.geomspace(1e-4, 0.06, 300))

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = {'2025': 'darkorange', '2026': 'purple'}
    for year in ['2025', '2026']:
        cdf_graf = cdf_from_counts(data[year]['counts_graf'])
        cdf_mrms = cdf_from_counts(data[year]['counts_mrms'])

        q_graf = np.interp(p_levels, cdf_graf, bin_right_edges)
        q_mrms = np.interp(p_levels, cdf_mrms, bin_right_edges)

        ax.plot(q_graf, q_mrms, color=colors[year], lw=2, label=year)

    lims = [0, 25]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('GRAF quantile (mm)')
    ax.set_ylabel('MRMS quantile (mm)')
    ax.set_title(f'Q-Q plot: GRAF vs MRMS, {window_label}, lead {clead}h\n'
                 f'(quantile levels p={p_levels[0]:.4f} to {p_levels[-1]:.4f})',
                 fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, lw=0.3)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f'Saved {plot_filename}')

# =======================================================================

def main():

    clead = sys.argv[1]
    mmdd_begin = sys.argv[2] if len(sys.argv) > 2 else '0101'
    mmdd_end = sys.argv[3] if len(sys.argv) > 3 else '0630'
    window_label = f'{mmdd_begin}-{mmdd_end}'

    infile = f'CDF_GRAF_vs_MRMS_data_lead{clead}h_{mmdd_begin}-{mmdd_end}.cPick'
    with open(infile, 'rb') as f:
        data = cPickle.load(f)

    plot_filename = f'CDF_GRAF_vs_MRMS_2025_2026_lead{clead}h_{window_label}.png'
    make_plot(data, clead, window_label, plot_filename)

    qq_filename = f'QQ_GRAF_vs_MRMS_2025_2026_lead{clead}h_{window_label}.png'
    make_qq_plot(data, clead, window_label, qq_filename)

if __name__ == '__main__':
    main()
