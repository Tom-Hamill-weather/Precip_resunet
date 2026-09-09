"""
python control_save_graf_texture_features.py cyyyymmddhh_begin cyyyymmddhh_end clead [n_workers]

Runs save_graf_texture_features.py for every 6-hourly init time in the date
range (all 4 GRAF cycles), for one lead time. Structural clone of
control_resunet_inference_gamma_mixture.py: each init time is a separate
process (isolates a crash/OOM in one GRIB read from the rest of the
backfill); n_workers of them run concurrently. This job is CPU/IO-bound
(pygrib reads + scipy.ndimage filters), not GPU-bound, so it can run
alongside GPU training/inference without contention.

Tom Hamill, Aug 2026
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutils import daterange

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end   = sys.argv[2]
clead             = sys.argv[3]
n_workers         = int(sys.argv[4]) if len(sys.argv) > 4 else 8

# Same oversubscription guard as control_resunet_inference_gamma_mixture.py --
# scipy.ndimage/numpy otherwise default to using every core per process.
_threads_per_worker = str(max(1, os.cpu_count() // n_workers))
_worker_env = dict(os.environ,
                   OMP_NUM_THREADS=_threads_per_worker,
                   MKL_NUM_THREADS=_threads_per_worker,
                   OPENBLAS_NUM_THREADS=_threads_per_worker,
                   NUMEXPR_NUM_THREADS=_threads_per_worker)

date_list = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 6)   # all 4 cycles

print(f'{len(date_list)} init times ({cyyyymmddhh_begin} to {cyyyymmddhh_end}, '
      f'6-h stride), clead={clead}h, {n_workers} workers')


def run_one(date):
    cmd = ['python', 'save_graf_texture_features.py', date, date, clead]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_worker_env)
    return date, result.returncode, result.stderr[-500:]


with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(run_one, date) for date in date_list]
    ndone = 0
    nfailed = 0
    for future in as_completed(futures):
        date, rc, err = future.result()
        ndone += 1
        if rc != 0:
            nfailed += 1
            print(f'  FAILED {date} (rc={rc}): {err}')
        if ndone % 50 == 0 or ndone == len(date_list):
            print(f'  {ndone}/{len(date_list)} complete ({nfailed} failed)')

print(f'\nDone: {ndone - nfailed}/{ndone} succeeded.')
