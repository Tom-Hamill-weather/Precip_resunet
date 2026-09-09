"""
reliability_6hourly_mlp_3panel_bycycle.py -- stratify the existing 6h MLP
3-panel reliability diagram by GRAF init cycle (00/06/12/18Z), to test
whether the pooled underconfidence signal seen after the 06/18Z verification
extension is a real per-cycle miscalibration or a pooling artifact (cycles
individually calibrated but offset from each other).

Reuses the per-date cache written by reliability_6hourly_mlp_3panel.py
(relia_6h_MLP_3panel_percache_lead{clead}h.cPick) -- no new inference or
MRMS/control reads, just re-aggregation of already-cached per-date
contingency tables, split by cdate[-2:].

Usage:
    python reliability_6hourly_mlp_3panel_bycycle.py <clead>
"""

import os
import sys
import numpy as np
import _pickle as cPickle

from reliability_6hourly_mlp_3panel import (
    get_paths, compute_relia, plot_3panel,
)

CYCLES = ['00', '06', '12', '18']


def main():
    if len(sys.argv) != 2:
        print('Usage: python reliability_6hourly_mlp_3panel_bycycle.py <clead>')
        sys.exit(1)
    clead = int(sys.argv[1])

    _, _, relia_dir, _ = get_paths()
    percache_fname = os.path.join(
        relia_dir, f'relia_6h_MLP_3panel_percache_lead{clead}h.cPick')
    with open(percache_fname, 'rb') as fh:
        percache = cPickle.load(fh)
    dates = percache['dates']
    print(f'Loaded per-date cache: {len(dates)} dates, {percache_fname}')

    pthresholds = [0.25, 2.5, 10.0]
    nthresholds = len(pthresholds)
    ncats = 11
    probability = np.arange(ncats) * 100.0 / float(ncats - 1)

    for cyc in CYCLES:
        cyc_dates = [d for d in dates if d[-2:] == cyc]
        if not cyc_dates:
            print(f'cycle={cyc}Z: no cached dates, skipping')
            continue

        contab         = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        BS_sum         = np.zeros(nthresholds)
        nsamps_sum     = np.zeros(nthresholds)
        nobs_exceed_sum = np.zeros(nthresholds)
        nobs_total_sum  = np.zeros(nthresholds)

        contab_c        = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        BS_sum_c         = np.zeros(nthresholds)
        nsamps_sum_c      = np.zeros(nthresholds)

        for cdate in cyc_dates:
            r = dates[cdate]
            for ithresh in range(nthresholds):
                ctab, bs, ns, nex, ntot = r['mlp'][ithresh]
                contab[ithresh]          += ctab
                BS_sum[ithresh]          += bs
                nsamps_sum[ithresh]      += ns
                nobs_exceed_sum[ithresh] += nex
                nobs_total_sum[ithresh]  += ntot

                ctab_ci, bs_ci, ns_ci, _, _ = r['control'][ithresh]
                contab_c[ithresh]     += ctab_ci
                BS_sum_c[ithresh]     += bs_ci
                nsamps_sum_c[ithresh] += ns_ci

        relia_arr           = np.full((nthresholds, ncats), -99.99)
        frequse_arr         = np.zeros((nthresholds, ncats))
        BSS_arr              = np.full(nthresholds, np.nan)
        relia_control_arr    = np.full((nthresholds, ncats), -99.99)
        frequse_control_arr  = np.zeros((nthresholds, ncats))
        BSS_control_arr      = np.full(nthresholds, np.nan)

        print(f'\n=== cycle={cyc}Z  ({len(cyc_dates)} dates) ===')
        for ithresh, thresh in enumerate(pthresholds):
            if nsamps_sum[ithresh] == 0:
                continue
            BS_mean    = BS_sum[ithresh] / nsamps_sum[ithresh]
            climo_freq = (nobs_exceed_sum[ithresh] / nobs_total_sum[ithresh]
                          if nobs_total_sum[ithresh] > 0 else np.nan)
            BS_climo   = climo_freq * (1.0 - climo_freq) if not np.isnan(climo_freq) else np.nan
            BSS        = (1.0 - BS_mean / BS_climo
                          if (not np.isnan(BS_climo) and BS_climo > 0) else np.nan)
            frequse, relia = compute_relia(contab[ithresh], ncats)
            relia_arr[ithresh]   = relia
            frequse_arr[ithresh] = frequse
            BSS_arr[ithresh]     = BSS

            BS_mean_c = BS_sum_c[ithresh] / nsamps_sum_c[ithresh] if nsamps_sum_c[ithresh] > 0 else np.nan
            BSS_c     = (1.0 - BS_mean_c / BS_climo
                        if (not np.isnan(BS_climo) and BS_climo > 0 and not np.isnan(BS_mean_c)) else np.nan)
            frequse_c, relia_c = compute_relia(contab_c[ithresh], ncats)
            relia_control_arr[ithresh]   = relia_c
            frequse_control_arr[ithresh] = frequse_c
            BSS_control_arr[ithresh]     = BSS_c

            cbss = f'{BSS:.3f}' if not np.isnan(BSS) else 'N/A'
            print(f'  thresh={thresh:5.2f}mm  climo={climo_freq:.4f}  '
                  f'BSS_mlp={cbss}  BSS_control={BSS_c:.3f}')
            print(f'    relia(mlp)    = {100.*np.ma.masked_where(relia<-99.,relia)}')
            print(f'    relia(control)= {100.*np.ma.masked_where(relia_c<-99.,relia_c)}')
            print(f'    fcst prob bins= {probability}')

        out_png = f'Relia_6h_MLP_MRMS_3panel_cycle{cyc}Z_lead{clead}h.png'
        plot_3panel(probability, relia_arr, frequse_arr, BSS_arr,
                    relia_control_arr, frequse_control_arr, BSS_control_arr,
                    pthresholds, clead, f'cycle{cyc}Z', f'lead{clead}h', out_png)


if __name__ == '__main__':
    main()
