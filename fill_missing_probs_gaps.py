"""
python fill_missing_probs_gaps.py [n_workers]

Fills in missing (date, lead) gamma-mixture probability files in probs/ for
the day 1-7 (MLP training) and day 10-end (MLP test) windows of all 12
months of 2025, 00Z/12Z only.  Skips any (date, lead) pair whose output
file already exists, so it's safe to re-run after a partial/interrupted
run.  Mirrors control_resunet_inference_gamma_mixture.py's concurrency
and per-worker CPU thread-capping.

Tom Hamill, Jul 2026
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PROBS_DIR = '/data/resnet_data/probs'

MONTHS_2025 = [
    (1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
    (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31),
]
TRAIN_DAY_END  = 7    # days 1-7 train
TEST_DAY_START = 10   # days 10-end test (days 8-9 are a gap, no data needed)

n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8

_threads_per_worker = str(max(1, os.cpu_count() // n_workers))
_worker_env = dict(os.environ,
                   OMP_NUM_THREADS=_threads_per_worker,
                   MKL_NUM_THREADS=_threads_per_worker,
                   OPENBLAS_NUM_THREADS=_threads_per_worker,
                   NUMEXPR_NUM_THREADS=_threads_per_worker)


def needed_dates():
    dates = []
    for mm, ndays in MONTHS_2025:
        days = list(range(1, TRAIN_DAY_END + 1)) + list(range(TEST_DAY_START, ndays + 1))
        for dd in days:
            for hh in ('00', '12'):
                dates.append(f'2025{mm:02d}{dd:02d}{hh}')
    return dates


def missing_jobs():
    jobs = []
    for date in needed_dates():
        for lead in range(1, 49):
            fname = os.path.join(PROBS_DIR, f'{date}_{lead}_probs_gamma_mixture.nc')
            if not os.path.exists(fname):
                jobs.append((date, lead))
    return jobs


def run_one(date, ilead):
    cmd = ['python', 'resunet_inference_gamma_mixture_optimized.py', date, str(ilead)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_worker_env)
    return date, ilead, result.returncode, result.stderr[-500:]


def main():
    jobs = missing_jobs()
    print(f'{len(jobs)} missing (date, lead) pairs to fill, {n_workers} workers')

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_one, date, ilead) for date, ilead in jobs]
        ndone = 0
        for future in as_completed(futures):
            date, ilead, rc, err = future.result()
            ndone += 1
            if rc != 0:
                print(f'  FAILED {date} lead={ilead}h (rc={rc}): {err}')
            if ndone % 100 == 0 or ndone == len(jobs):
                print(f'  {ndone}/{len(jobs)} complete')


if __name__ == '__main__':
    main()
