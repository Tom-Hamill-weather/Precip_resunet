"""
compute_bss_ci_baseline_vs_season.py [date_range]

Computes, for the baseline (original-training) and season (current-
training) Attention ResUNet, a per-lead, per-threshold, per-terrain-
stratum Hodges-Lehmann point estimate and 90% confidence interval on
the model's own per-day BSS_gamma distribution -- a rank-based
nonparametric CI derived by inverting the (one-sample) Wilcoxon
signed-rank statistic on the per-day BSS_gamma values, NOT a bootstrap
resample. This is the same statistical family (Wilcoxon signed-rank)
used for the paired MLP-vs-control significance test in the 6-hourly
MLP manuscript (reliability_6hourly_mlp_3panel.py) -- here applied as
a one-sample location CI on each curve separately, since the BSS-vs-
lead figure plots two independent model curves rather than a single
paired difference.

Per-day BSS_gamma = 1 - bs_gamma_day / bs_climo_day, read directly from
the daily_contab cache written by reliability_resunet_mixture.py (no
re-reading of raw probability/MRMS data, no re-running of inference).

Usage:
    python compute_bss_ci_baseline_vs_season.py 2026010100_to_2026083118
"""

import os
import sys
import numpy as np
import _pickle as cPickle
from dateutils import daterange

RELIA_DIR = '/data/resnet_data/relia'
DAILY_CACHE_DIR = os.path.join(RELIA_DIR, 'daily_contab')
DATE_RANGE = sys.argv[1] if len(sys.argv) > 1 else '2026010100_to_2026083118'
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
PLOT_THRESHOLDS = [0.25, 1.0, 5.0]
THRESH_IDX = [0, 1, 3]   # indices into pthresholds = [0.25, 1.0, 2.5, 5.0, 10.0]
REGIONS = ['top10', 'bottom90']
CI_LEVEL = 90.0

MODEL_CACHE_TAG = {'baseline': '', 'season': '_season'}

date0, date1 = DATE_RANGE.split('_to_')
cyyyymmddhh_list = daterange(date0, date1, 6)


def hodges_lehmann_ci(x, ci=90.0):
    """Rank-based (Wilcoxon signed-rank inversion) point estimate + CI on
    the median of a one-dimensional sample, via the Walsh averages --
    no resampling. Uses the large-n normal approximation to the
    Wilcoxon signed-rank null distribution to pick which order
    statistic of the sorted Walsh averages bounds the CI (Hollander &
    Wolfe, Nonparametric Statistical Methods, 2nd ed., Sec. 3.3)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2.0)
    M = len(walsh)
    hl_est = float(np.median(walsh))

    from scipy.stats import norm
    alpha = 1.0 - ci / 100.0
    z = norm.ppf(1.0 - alpha / 2.0)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(round(mu - z * sigma))
    k = max(k, 0)
    k = min(k, M // 2)
    lo = float(walsh[k])
    hi = float(walsh[M - 1 - k])
    return hl_est, lo, hi, n


def load_daily_bss(tag, lead, region, ithresh):
    """Per-day BSS_gamma array for one (model_tag, lead, region, threshold)."""
    cache_tag = MODEL_CACHE_TAG[tag]
    vals = []
    for date in cyyyymmddhh_list:
        fn = os.path.join(DAILY_CACHE_DIR, f'{date}_lead{lead}h{cache_tag}.cPick')
        if not os.path.exists(fn):
            continue
        with open(fn, 'rb') as fh:
            cached = cPickle.load(fh)
        rs = cached.get('region_stats')
        if rs is None or region not in rs:
            continue
        bs_raw_arr, bs_gamma_arr, bs_climo_arr, ns_arr = rs[region]
        bs_climo = bs_climo_arr[ithresh]
        if bs_climo <= 0:
            continue
        vals.append(1.0 - bs_gamma_arr[ithresh] / bs_climo)
    return np.array(vals)


results = {}
for tag in MODEL_CACHE_TAG:
    for lead in LEAD_TIMES:
        for region in REGIONS:
            for ti, ithresh in enumerate(THRESH_IDX):
                daily = load_daily_bss(tag, lead, region, ithresh)
                hl_est, lo, hi, n = hodges_lehmann_ci(daily, ci=CI_LEVEL)
                key = (tag, lead, region, PLOT_THRESHOLDS[ti])
                results[key] = dict(hl_est=hl_est, ci_lo=lo, ci_hi=hi, n=n)
                print(f'{tag:8s} lead={lead:2d}h {region:9s} '
                      f'{PLOT_THRESHOLDS[ti]:5.2f}mm  n={n:4d}  '
                      f'HL={hl_est:.3f}  [{lo:.3f}, {hi:.3f}]')

outfile = os.path.join(RELIA_DIR, f'bss_ci_baseline_vs_season_{DATE_RANGE}.cPick')
with open(outfile, 'wb') as fh:
    cPickle.dump(results, fh)
print(f'\nSaved: {outfile}')
