"""
bootstrap_ci_6hourly_mlp.py -- bootstrap confidence intervals + seed-spread
comparison for 6-hourly MLP reliability/BSS results.

Motivation: after 8+ rounds of architecture/loss/feature comparisons against
the same day12-end-2025 test window, Tom asked whether the GRU's apparent
"near-perfect" 10mm calibration is a real, reproducible effect or an
artifact of how long the search ran. This script answers the "is it
reproducible" half by (a) bootstrap-resampling test *dates* (not grid
points -- dates are the right resampling unit since grid points within a
date are spatially correlated, same reasoning as the paired-day
significance test already in reliability_6hourly_mlp_3panel.py) to get a CI
on BSS and mean|reliability deviation| for each checkpoint, and (b)
comparing seed-to-seed spread (several independently-trained checkpoints of
the *same* recipe) against a single seed's bootstrap CI width -- if seeds
disagree by more than one seed's own resampling uncertainty, the recipe's
apparent quality is seed-sensitive, not just estimation noise.

Usage:
    python bootstrap_ci_6hourly_mlp.py <clead> <variant1> [<variant2> ...] [--nboot=1000]

    Each <variant> is the same freeform suffix string
    reliability_6hourly_mlp_3panel.py accepts, e.g. "hurdle_texture_gru_seed999"
    or "texture_gru_seed2026" -- must have already been evaluated (its
    percache file must exist).

Output: printed table (point estimate + 95% CI per variant per threshold),
plus a pickled summary dict at
    <relia_dir>/bootstrap_ci_summary_lead{clead}h.cPick
merging in results from any previous run on the same clead (so seeds/
variants can be added incrementally across separate invocations without
losing earlier results).

Tom Hamill / Claude, Aug 2026
"""

import os
import sys
import numpy as np
import _pickle as cPickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirror reliability_6hourly_mlp_3panel.py's environment detection exactly --
# duplicated rather than imported to keep this a standalone, dependency-light
# analysis script (no torch/netCDF4/matplotlib needed here).
def detect_environment():
    aws_base = '/data/resnet_data'
    if os.path.isdir(aws_base):
        return 'aws', aws_base
    return 'laptop', None


ENVIRONMENT, AWS_BASE_PATH = detect_environment()


def get_relia_dir():
    if ENVIRONMENT == 'aws':
        return os.path.join(AWS_BASE_PATH, 'relia')
    return os.path.expanduser('~/python/resnet_data/relia')


PTHRESHOLDS = [0.25, 2.5, 10.0]
NCATS = 11
PROBABILITY = np.arange(NCATS) * 100.0 / float(NCATS - 1)
INTERIOR_LO, INTERIOR_HI = 10.0, 90.0  # interior-bin window for mean|dev|, matches prior ad-hoc analyses this session


def compute_relia(contab, ncats):
    """Identical to reliability_6hourly_mlp_3panel.py's compute_relia."""
    frequse = np.zeros(ncats, dtype=float)
    relia = np.full(ncats, -99.99)
    total = float(np.sum(contab))
    for icat in range(ncats):
        n = np.sum(contab[icat, :])
        frequse[icat] = n / total if total > 0 else 0.0
        if n > 5:
            relia[icat] = float(contab[icat, 1]) / n
    return frequse, relia


def mean_abs_deviation(relia, probability):
    """Mean |forecast probability - observed frequency| over interior bins
    (10-90%), skipping bins with too few samples (relia == -99.99).
    Matches the ad-hoc metric reported throughout this session's chat, now
    made reproducible as code."""
    mask = (probability >= INTERIOR_LO) & (probability <= INTERIOR_HI) & (relia > -99.0)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(relia[mask] * 100.0 - probability[mask])))


def aggregate_dates(percache, date_subset, ithresh, key='mlp'):
    """Sum contab/BS/nsamps/nobs_* over a (possibly repeated, for bootstrap)
    list of dates for one threshold index. Returns the 5 summed quantities."""
    contab = np.zeros((NCATS, 2), dtype=np.int64)
    bs_sum = 0.0
    ns_sum = 0.0
    nex_sum = 0.0
    ntot_sum = 0.0
    dates_dict = percache['dates']
    for cdate in date_subset:
        cached = dates_dict.get(cdate)
        if cached is None:
            continue
        ctab, bs, ns, nex, ntot = cached[key][ithresh]
        contab += ctab
        bs_sum += bs
        ns_sum += ns
        nex_sum += nex
        ntot_sum += ntot
    return contab, bs_sum, ns_sum, nex_sum, ntot_sum


def bss_and_meandev(contab, bs_sum, ns_sum, nex_sum, ntot_sum):
    if ns_sum <= 0 or ntot_sum <= 0:
        return np.nan, np.nan
    bs_mean = bs_sum / ns_sum
    climo_freq = nex_sum / ntot_sum
    bs_climo = climo_freq * (1.0 - climo_freq)
    bss = 1.0 - bs_mean / bs_climo if bs_climo > 0 else np.nan
    _, relia = compute_relia(contab, NCATS)
    mdev = mean_abs_deviation(relia, PROBABILITY)
    return bss, mdev


def load_percache(clead, variant, relia_dir):
    variant_suffix = f'_{variant}' if variant else ''
    fname = os.path.join(relia_dir, f'relia_6h_MLP_3panel_percache_lead{clead}h{variant_suffix}.cPick')
    if not os.path.exists(fname):
        print(f'  WARNING: percache not found, skipping: {fname}')
        return None
    with open(fname, 'rb') as fh:
        return cPickle.load(fh)


def bootstrap_one(percache, nboot, rng):
    """Bootstrap-resample dates (with replacement) nboot times; return
    per-threshold arrays of shape (nboot,) for BSS and mean|dev|, plus the
    single point estimate (no resampling) for each."""
    dates = sorted(percache['dates'].keys())
    ndates = len(dates)
    nthresh = len(PTHRESHOLDS)

    point_bss = np.full(nthresh, np.nan)
    point_mdev = np.full(nthresh, np.nan)
    for ithresh in range(nthresh):
        contab, bs, ns, nex, ntot = aggregate_dates(percache, dates, ithresh)
        point_bss[ithresh], point_mdev[ithresh] = bss_and_meandev(contab, bs, ns, nex, ntot)

    boot_bss = np.full((nboot, nthresh), np.nan)
    boot_mdev = np.full((nboot, nthresh), np.nan)
    for iboot in range(nboot):
        resample_idx = rng.integers(0, ndates, size=ndates)
        resample_dates = [dates[i] for i in resample_idx]
        for ithresh in range(nthresh):
            contab, bs, ns, nex, ntot = aggregate_dates(percache, resample_dates, ithresh)
            boot_bss[iboot, ithresh], boot_mdev[iboot, ithresh] = bss_and_meandev(contab, bs, ns, nex, ntot)

    return {
        'ndates': ndates,
        'point_bss': point_bss,
        'point_mdev': point_mdev,
        'boot_bss': boot_bss,
        'boot_mdev': boot_mdev,
    }


def ci95(arr):
    lo = np.nanpercentile(arr, 2.5)
    hi = np.nanpercentile(arr, 97.5)
    return lo, hi


def print_variant_table(variant, result):
    print(f'\n--- {variant} (n_dates={result["ndates"]}) ---')
    for ithresh, thresh in enumerate(PTHRESHOLDS):
        bss_pt = result['point_bss'][ithresh]
        mdev_pt = result['point_mdev'][ithresh]
        bss_lo, bss_hi = ci95(result['boot_bss'][:, ithresh])
        mdev_lo, mdev_hi = ci95(result['boot_mdev'][:, ithresh])
        print(f'  {thresh:5.2f} mm | BSS={bss_pt:.3f} [{bss_lo:.3f}, {bss_hi:.3f}]  '
              f'mean|dev|={mdev_pt:.2f} [{mdev_lo:.2f}, {mdev_hi:.2f}]')


def compare_seed_spread(variants_by_group, results):
    """For each group of same-recipe-different-seed variants, compare the
    spread of point estimates across seeds against the width of a single
    seed's own bootstrap CI. If seed spread >> one seed's CI width, the
    recipe's apparent quality is seed-sensitive."""
    for group_name, variant_list in variants_by_group.items():
        present = [v for v in variant_list if v in results]
        if len(present) < 2:
            continue
        print(f'\n=== Seed spread check: {group_name} ({len(present)} seeds: {present}) ===')
        for ithresh, thresh in enumerate(PTHRESHOLDS):
            bss_vals = np.array([results[v]['point_bss'][ithresh] for v in present])
            mdev_vals = np.array([results[v]['point_mdev'][ithresh] for v in present])
            bss_range = np.nanmax(bss_vals) - np.nanmin(bss_vals)
            mdev_range = np.nanmax(mdev_vals) - np.nanmin(mdev_vals)
            ci_widths_bss = [ci95(results[v]['boot_bss'][:, ithresh]) for v in present]
            ci_widths_mdev = [ci95(results[v]['boot_mdev'][:, ithresh]) for v in present]
            avg_ci_width_bss = np.mean([hi - lo for lo, hi in ci_widths_bss])
            avg_ci_width_mdev = np.mean([hi - lo for lo, hi in ci_widths_mdev])
            flag_bss = 'SEED-SENSITIVE' if bss_range > avg_ci_width_bss else 'stable'
            flag_mdev = 'SEED-SENSITIVE' if mdev_range > avg_ci_width_mdev else 'stable'
            print(f'  {thresh:5.2f} mm | BSS: seed-range={bss_range:.4f} vs '
                  f'avg-1-seed-CI-width={avg_ci_width_bss:.4f}  [{flag_bss}]')
            print(f'             | mean|dev|: seed-range={mdev_range:.2f} vs '
                  f'avg-1-seed-CI-width={avg_ci_width_mdev:.2f}  [{flag_mdev}]')


def main():
    if len(sys.argv) < 3:
        print('Usage: python bootstrap_ci_6hourly_mlp.py <clead> <variant1> [<variant2> ...] [--nboot=N]')
        sys.exit(1)

    clead = int(sys.argv[1])
    nboot = 1000
    variants = []
    for a in sys.argv[2:]:
        if a.startswith('--nboot='):
            nboot = int(a.split('=', 1)[1])
        else:
            variants.append(a)

    relia_dir = get_relia_dir()
    rng = np.random.default_rng(20260809)  # fixed seed for the resampling itself -- reproducible CI, not a model seed

    print(f'Bootstrap CI: clead={clead}h  nboot={nboot}  variants={variants}')

    summary_fname = os.path.join(relia_dir, f'bootstrap_ci_summary_lead{clead}h.cPick')
    if os.path.exists(summary_fname):
        with open(summary_fname, 'rb') as fh:
            all_results = cPickle.load(fh)
        print(f'Loaded existing summary with {len(all_results)} variant(s): {list(all_results.keys())}')
    else:
        all_results = {}

    for variant in variants:
        print(f'\nProcessing variant: {variant}')
        percache = load_percache(clead, variant, relia_dir)
        if percache is None:
            continue
        result = bootstrap_one(percache, nboot, rng)
        all_results[variant] = result
        print_variant_table(variant, result)

    with open(summary_fname, 'wb') as fh:
        cPickle.dump(all_results, fh)
    print(f'\nSaved bootstrap summary ({len(all_results)} variants total) -> {summary_fname}')

    # Seed-spread groups: strip a trailing _seed{N} to find same-recipe groups
    # among everything in all_results (not just this invocation's variants),
    # so spread checks stay complete as more seeds accumulate across runs.
    import re
    groups = {}
    for v in all_results:
        base = re.sub(r'_seed\d+$', '', v)
        groups.setdefault(base, []).append(v)
    compare_seed_spread(groups, all_results)


if __name__ == '__main__':
    main()
