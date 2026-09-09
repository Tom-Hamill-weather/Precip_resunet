"""
python control_resunet_inference_2026_may_jul_aug.py [n_workers]

Backfills baseline-model probability files for May/Jul/Aug 2026, every
6-hourly cycle (00/06/12/18Z), leads 6-48h every 6h -- the "monthly by
month and lead time dependent training" model (resunet_inference_gamma_
mixture_optimized.py), not the season+FiLM pooled model. Requires the
month-specific training checkpoints (resunet_gamma_mixture_2026{05,07,08}
0100_{lead}h_best.pth) to already exist -- run the patch-pool build +
training pass for these (date, lead) pairs first.

Skips any (date, lead) whose output file already exists, so it's safe to
re-run after a partial/interrupted backfill.

Pattern copied from control_resunet_inference_2026_comparison.py (same
ThreadPoolExecutor + per-worker CPU thread-capping convention), restricted
to the baseline model only.
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutils import daterange

n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8

_threads_per_worker = str(max(1, os.cpu_count() // n_workers))
_worker_env = dict(os.environ,
                   OMP_NUM_THREADS=_threads_per_worker,
                   MKL_NUM_THREADS=_threads_per_worker,
                   OPENBLAS_NUM_THREADS=_threads_per_worker,
                   NUMEXPR_NUM_THREADS=_threads_per_worker)

PROBS_DIR = '/data/resnet_data/probs/'
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
SCRIPT = 'resunet_inference_gamma_mixture_optimized.py'
SUFFIX = '_probs_gamma_mixture.nc'

may = daterange('2026050100', '2026053118', 6)
jul = daterange('2026070100', '2026073118', 6)
aug = daterange('2026080100', '2026083118', 6)
date_list = may + jul + aug

jobs = []
for date in date_list:
    for lead in LEADS:
        outfile = os.path.join(PROBS_DIR, f'{date}_{lead}{SUFFIX}')
        if os.path.exists(outfile):
            continue
        jobs.append((date, lead))

print(f'{len(date_list)} init times x {len(LEADS)} leads '
      f'= {len(date_list) * len(LEADS)} total, '
      f'{len(jobs)} still needed (rest already on disk), {n_workers} workers')


def run_one(date, lead):
    cmd = [sys.executable, SCRIPT, date, str(lead)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_worker_env)
    return date, lead, result.returncode, result.stderr[-500:]


with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(run_one, *job) for job in jobs]
    ndone = 0
    nfail = 0
    for future in as_completed(futures):
        date, lead, rc, err = future.result()
        ndone += 1
        if rc != 0:
            nfail += 1
            print(f'  FAILED {date} lead={lead}h (rc={rc}): {err}')
        if ndone % 50 == 0 or ndone == len(jobs):
            print(f'  {ndone}/{len(jobs)} complete ({nfail} failed)')

print(f'Done. {nfail} failures out of {len(jobs)}.')
