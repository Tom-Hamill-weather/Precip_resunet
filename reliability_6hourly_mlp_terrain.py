#!/usr/bin/env python3
"""
reliability_6hourly_mlp_terrain.py — terrain-roughness-stratified BSS for the
six-hourly MLP vs. the independence-assumption control, one lead time per run.

Usage:
    python reliability_6hourly_mlp_terrain.py <clead>

Mirrors reliability_6hourly_mlp_3panel.py's data-collection loop (same MLP,
same control, same out-of-sample test date list, same MRMS quality
threshold Q>0.6) but additionally stratifies every per-date contingency
table into two terrain-roughness regions using the top10_mask/bottom90_mask
from terrain_roughness_mask_graf.nc (see terrain_roughness_graf.py), the
same masks used for the hourly ResUNet's BSS-vs-lead-time figure
(plot_BSS_leadtime.py / reliability_resunet_mixture.py).

Region restriction is applied by setting MRMS quality to -1 outside the
region before calling compute_contab_BS, so the existing quality>0.6 check
in that function does the masking -- no changes to compute_contab_BS itself.

Output: one cPickle per lead time,
    {relia_dir}/relia_6h_MLP_terrain_q0.6_{date_start}_to_{date_end}_lead{clead}h.cPick
containing per-threshold, per-region BSS/contab for both the MLP and the
independence-assumption control.

Tom Hamill, Aug 2026
"""

import os
import sys
import numpy as np
import _pickle as cPickle
import torch
from netCDF4 import Dataset

from reliability_6hourly_mlp_3panel import (
    get_paths, load_mlp, read_prob_params_6h, read_control_probs_6h,
    read_mrms_6h, apply_mlp_fulldomain, exceedance_prob, compute_contab_BS,
    julian_features, build_test_datelist, load_local_std_grid,
)

TERRAIN_MASK_NC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'terrain_roughness_mask_graf.nc')

PTHRESHOLDS = [0.25, 2.5, 10.0]
NCATS = 11
REGIONS = ['unstrat', 'top10', 'bottom90']


def load_terrain_masks():
    with Dataset(TERRAIN_MASK_NC, 'r') as ds:
        top10 = np.asarray(ds.variables['top10_mask'][:], dtype=bool)
        bottom90 = np.asarray(ds.variables['bottom90_mask'][:], dtype=bool)
    return {'unstrat': None, 'top10': top10, 'bottom90': bottom90}


def region_quality(mean_qual, region_mask):
    """Quality array with -1 (fails Q>0.6) outside the region, so an
    existing compute_contab_BS call restricted to this quality array is
    automatically restricted to the region too."""
    if region_mask is None:
        return mean_qual
    return np.where(region_mask, mean_qual, -1.0)


def main():
    if len(sys.argv) not in (2, 3):
        print('Usage: python reliability_6hourly_mlp_terrain.py <clead> [film]')
        sys.exit(1)

    clead   = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) == 3 else None
    if clead < 6:
        print('ERROR: clead must be >= 6')
        sys.exit(1)

    print(f'reliability_6hourly_mlp_terrain.py  clead={clead}h'
          f'{"  variant=" + variant if variant else ""}')

    probs_dir, mrms_dir, relia_dir, control_dir = get_paths()
    os.makedirs(relia_dir, exist_ok=True)

    region_masks = load_terrain_masks()
    nthresholds = len(PTHRESHOLDS)

    cyyyymmddhh_list = build_test_datelist()
    date_start = cyyyymmddhh_list[0]
    date_end   = cyyyymmddhh_list[-1]

    variant_suffix = f'_{variant}' if variant else ''
    pick_fname = os.path.join(
        relia_dir,
        f'relia_6h_MLP_terrain_q0.6_{date_start}_to_{date_end}_lead{clead}h{variant_suffix}.cPick')

    if os.path.exists(pick_fname):
        print(f'Cache already exists, skipping: {pick_fname}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Torch device: {device}')
    model, feat_mean, feat_std = load_mlp(clead, device, variant=variant)
    local_std_grid = load_local_std_grid() if len(feat_mean) >= 39 else None

    ndates = len(cyyyymmddhh_list)
    print(f'Total init times to process: {ndates}')

    # contab/BS accumulators: [region][method] -> arrays over thresholds
    contab = {r: {'mlp': np.zeros((nthresholds, NCATS, 2), dtype=np.int64),
                  'control': np.zeros((nthresholds, NCATS, 2), dtype=np.int64)}
              for r in REGIONS}
    BS_sum = {r: {'mlp': np.zeros(nthresholds), 'control': np.zeros(nthresholds)}
              for r in REGIONS}
    nsamps_sum = {r: {'mlp': np.zeros(nthresholds), 'control': np.zeros(nthresholds)}
                  for r in REGIONS}
    nobs_exceed_sum = {r: {'mlp': np.zeros(nthresholds), 'control': np.zeros(nthresholds)}
                       for r in REGIONS}
    nobs_total_sum = {r: {'mlp': np.zeros(nthresholds), 'control': np.zeros(nthresholds)}
                      for r in REGIONS}
    ngood = 0

    for idate, cdate in enumerate(cyyyymmddhh_list):
        params_6h, lat, lon = read_prob_params_6h(probs_dir, cdate, clead)
        prob_ok = params_6h is not None

        precip_6h, mean_qual, mrms_istat = read_mrms_6h(mrms_dir, cdate, clead)
        mrms_ok = mrms_istat == 0

        control_probs = read_control_probs_6h(control_dir, cdate, clead, PTHRESHOLDS)
        control_ok = control_probs is not None

        ps = 'ok' if prob_ok else 'missing'
        ms = 'ok' if mrms_ok else 'missing'
        cs = 'ok' if control_ok else 'missing'
        print(f'{idate+1:4d}/{ndates}  init={cdate}  params={ps}  mrms={ms}  control={cs}')

        if not prob_ok or not mrms_ok or not control_ok:
            continue

        ny, nx = precip_6h.shape
        ngood += 1

        cos_doy, sin_doy = julian_features(cdate)
        fz, mw, s1, sc1, s2, sc2 = apply_mlp_fulldomain(
            model, feat_mean, feat_std, params_6h, ny, nx, device, cos_doy, sin_doy,
            local_std=local_std_grid)

        for ithresh, thresh in enumerate(PTHRESHOLDS):
            prob_mlp = exceedance_prob(fz, mw, s1, sc1, s2, sc2, thresh)
            prob_ctl = control_probs[thresh]

            for region in REGIONS:
                qual_region = region_quality(mean_qual, region_masks[region])

                ctab_m, bs_m, ns_m, nex_m, ntot_m = compute_contab_BS(
                    ny, nx, prob_mlp, precip_6h, qual_region, NCATS, thresh)
                contab[region]['mlp'][ithresh]          += ctab_m
                BS_sum[region]['mlp'][ithresh]          += bs_m
                nsamps_sum[region]['mlp'][ithresh]      += ns_m
                nobs_exceed_sum[region]['mlp'][ithresh] += nex_m
                nobs_total_sum[region]['mlp'][ithresh]  += ntot_m

                ctab_c, bs_c, ns_c, nex_c, ntot_c = compute_contab_BS(
                    ny, nx, prob_ctl, precip_6h, qual_region, NCATS, thresh)
                contab[region]['control'][ithresh]          += ctab_c
                BS_sum[region]['control'][ithresh]          += bs_c
                nsamps_sum[region]['control'][ithresh]      += ns_c
                nobs_exceed_sum[region]['control'][ithresh] += nex_c
                nobs_total_sum[region]['control'][ithresh]  += ntot_c

    if ngood == 0:
        print('\nERROR: No dates with complete data found.')
        sys.exit(1)

    print(f'\n{ngood}/{ndates} init times had complete data.')

    BSS = {r: {'mlp': np.full(nthresholds, np.nan),
               'control': np.full(nthresholds, np.nan)} for r in REGIONS}
    BS_climo_out = {r: np.full(nthresholds, np.nan) for r in REGIONS}
    climo_freq_out = {r: np.full(nthresholds, np.nan) for r in REGIONS}

    for region in REGIONS:
        for ithresh, thresh in enumerate(PTHRESHOLDS):
            climo_freq = (nobs_exceed_sum[region]['mlp'][ithresh]
                          / nobs_total_sum[region]['mlp'][ithresh]
                          if nobs_total_sum[region]['mlp'][ithresh] > 0 else np.nan)
            BS_climo = climo_freq * (1.0 - climo_freq) if not np.isnan(climo_freq) else np.nan
            climo_freq_out[region][ithresh] = climo_freq
            BS_climo_out[region][ithresh] = BS_climo

            for method in ('mlp', 'control'):
                ns = nsamps_sum[region][method][ithresh]
                bs_mean = BS_sum[region][method][ithresh] / ns if ns > 0 else np.nan
                bss = (1.0 - bs_mean / BS_climo
                       if (not np.isnan(BS_climo) and BS_climo > 0
                           and not np.isnan(bs_mean)) else np.nan)
                BSS[region][method][ithresh] = bss

            print(f'  region={region:9s} thresh={thresh:5.2f}mm  climo={climo_freq:.4f}  '
                  f'BSS_mlp={BSS[region]["mlp"][ithresh]:.3f}  '
                  f'BSS_control={BSS[region]["control"][ithresh]:.3f}')

    out_dict = {
        'clead': clead,
        'pthresholds': PTHRESHOLDS,
        'ngood': ngood,
        'ndates': ndates,
        'regions': REGIONS,
        'BSS': BSS,
        'BS_climo': BS_climo_out,
        'climo_freq': climo_freq_out,
        'contab': contab,
        'nsamps': nsamps_sum,
    }
    with open(pick_fname, 'wb') as fh:
        cPickle.dump(out_dict, fh)
    print(f'Saved statistics to {pick_fname}')


if __name__ == '__main__':
    main()
