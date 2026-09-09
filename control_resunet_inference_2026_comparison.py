"""
python control_resunet_inference_2026_comparison.py [n_workers]

Backfills probability files for the H1-2026 seasonal-vs-monthly comparison:
runs both resunet_inference_gamma_mixture_optimized.py (the per-lead/
per-month-retrained "baseline" model) and resunet_inference_gamma_mixture_
season.py (the season-pooled + FiLM lead-pooled model) for every (date,
lead) pair needed by reliability_resunet_mixture.py's date_set='2026h1'
sample.

Date sample mirrors the existing baseline evaluation convention (one
representative month per season, 6-hourly cycles) but drawn from the
independent Jan-Jun 2026 data: Jan/Apr/Jun as DJF/MAM/JJA proxies. No SON
proxy exists in H1 2026.

Lead times are the 8 values plot_BSS_leadtime.py actually plots
(6,12,...,48h) -- both models have checkpoints/support at these leads, so
no need to run the full 1-48h integer sweep the older control script does.

Skips any (date, lead, model) whose output file already exists, so it's
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

jan = daterange('2026010100', '2026013118', 6)
apr = daterange('2026040100', '2026043018', 6)
jun = daterange('2026060100', '2026063018', 6)
date_list = jan + apr + jun

MODELS = {
    'baseline': ('resunet_inference_gamma_mixture_optimized.py', '_probs_gamma_mixture.nc'),
    'season':   ('resunet_inference_gamma_mixture_season.py',    '_probs_gamma_mixture_season.nc'),
}

jobs = []
for date in date_list:
    for lead in LEADS:
        for model_tag, (script, suffix) in MODELS.items():
            outfile = os.path.join(PROBS_DIR, f'{date}_{lead}{suffix}')
            if os.path.exists(outfile):
                continue
            jobs.append((date, lead, model_tag, script))

print(f'{len(date_list)} init times x {len(LEADS)} leads x {len(MODELS)} models '
      f'= {len(date_list) * len(LEADS) * len(MODELS)} total, '
      f'{len(jobs)} still needed (rest already on disk), {n_workers} workers')


def run_one(date, lead, model_tag, script):
    cmd = ['python', script, date, str(lead)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_worker_env)
    return date, lead, model_tag, result.returncode, result.stderr[-500:]


with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(run_one, *job) for job in jobs]
    ndone = 0
    nfail = 0
    for future in as_completed(futures):
        date, lead, model_tag, rc, err = future.result()
        ndone += 1
        if rc != 0:
            nfail += 1
            print(f'  FAILED {date} lead={lead}h model={model_tag} (rc={rc}): {err}')
        if ndone % 50 == 0 or ndone == len(jobs):
            print(f'  {ndone}/{len(jobs)} complete ({nfail} failed)')

print(f'Done. {nfail} failures out of {len(jobs)}.')
