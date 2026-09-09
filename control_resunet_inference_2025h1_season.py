"""
python control_resunet_inference_2025h1_season.py [n_workers]

Backfills seasonal+FiLM model probability files for Jan/Apr/Jun 2025 --
the year-earlier counterpart to control_resunet_inference_2026_comparison.py's
2026h1 season backfill, needed so plot_relia_gamma_year_compare.py can do a
true 2025-vs-2026 comparison under the season model. Baseline-model probs
for these dates already exist on disk, so only the season model is run here.

Lead times are the 8 values plot_BSS_leadtime.py plots (6,12,...,48h).

Skips any (date, lead) whose season output file already exists, so it's
safe to re-run after a partial/interrupted backfill.
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

jan = daterange('2025010100', '2025013118', 6)
apr = daterange('2025040100', '2025043018', 6)
jun = daterange('2025060100', '2025063018', 6)
date_list = jan + apr + jun

SCRIPT = 'resunet_inference_gamma_mixture_season.py'
SUFFIX = '_probs_gamma_mixture_season.nc'

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
