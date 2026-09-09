"""
python control_resunet_inference_gamma_mixture.py cyyyymmddhh_begin cyyyymmddhh_end [n_workers] [day_stride]

Runs resunet_inference_gamma_mixture_optimized.py for every (date, lead) pair
in the date range, for lead times 1-48h.  Both 00Z and 12Z cycles are used
for each day kept.  day_stride subsamples calendar days (relative to
cyyyymmddhh_begin) to cut cost, e.g. day_stride=3 keeps every 3rd day
(days 1, 4, 7, ... of the range) and drops the rest.  Default day_stride=1
(every day).

Each (date, lead) call is a separate process (fresh model/data load), so
most of its wall time is CPU/IO-bound overhead rather than GPU compute;
running n_workers of them concurrently overlaps that overhead across calls
sharing the GPU.  Default n_workers=4.
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutils import daterange

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end   = sys.argv[2]
n_workers         = int(sys.argv[3]) if len(sys.argv) > 3 else 4
day_stride        = int(sys.argv[4]) if len(sys.argv) > 4 else 1

# Each subprocess otherwise defaults to using every CPU core for
# torch/numpy ops; with n_workers>1 that oversubscribes the machine and
# workers thrash each other.  Cap per-process threads so total demand
# roughly matches the core count.
_threads_per_worker = str(max(1, os.cpu_count() // n_workers))
_worker_env = dict(os.environ,
                   OMP_NUM_THREADS=_threads_per_worker,
                   MKL_NUM_THREADS=_threads_per_worker,
                   OPENBLAS_NUM_THREADS=_threads_per_worker,
                   NUMEXPR_NUM_THREADS=_threads_per_worker)

all_days  = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 24)   # one entry/day, same hour as begin
kept_days = all_days[::day_stride]

date_list = []
for d in kept_days:
    yyyymmdd = d[:8]
    date_list.append(yyyymmdd + '00')
    date_list.append(yyyymmdd + '12')

jobs = [(date, ilead) for date in date_list for ilead in range(1, 49)]

print(f'{len(kept_days)}/{len(all_days)} days kept (stride={day_stride}), '
      f'{len(date_list)} init times (00Z/12Z), 48 leads each -> '
      f'{len(jobs)} inference calls, {n_workers} workers')


def run_one(date, ilead):
    cmd = ['python', 'resunet_inference_gamma_mixture_optimized.py', date, str(ilead)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_worker_env)
    return date, ilead, result.returncode, result.stderr[-500:]


with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(run_one, date, ilead) for date, ilead in jobs]
    ndone = 0
    for future in as_completed(futures):
        date, ilead, rc, err = future.result()
        ndone += 1
        if rc != 0:
            print(f'  FAILED {date} lead={ilead}h (rc={rc}): {err}')
        if ndone % 50 == 0 or ndone == len(jobs):
            print(f'  {ndone}/{len(jobs)} complete')
