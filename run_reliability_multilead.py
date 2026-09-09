"""
python run_reliability_multilead.py [n_workers]

Runs reliability_resunet_mixture.py for every 6h lead 6-48h, for both the
2025h1 and 2026h1 (matched Jan/Apr/Jun) date sets, baseline model_tag --
extending the single-lead (24h) year-over-year reliability/BSS comparison
to all lead times. Skips any (lead, date_set) whose output cPickle already
exists.
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

RELIA_DIR = '/data/resnet_data/relia'
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
DATE_SETS = {
    '2025h1': ('2025010100', '2025063018'),
    '2026h1': ('2026010100', '2026063018'),
}

jobs = []
for lead in LEADS:
    for date_set, (d0, d1) in DATE_SETS.items():
        outfile = os.path.join(
            RELIA_DIR, f'relia_GRAF_ResUNet_Mixture_q0.5_{d0}_to_{d1}_lead{lead}h.cPick')
        if os.path.exists(outfile):
            continue
        jobs.append((lead, date_set))

print(f'{len(LEADS)} leads x {len(DATE_SETS)} date_sets = {len(LEADS)*len(DATE_SETS)} total, '
      f'{len(jobs)} still needed, {n_workers} workers')


def run_one(lead, date_set):
    cmd = ['python', 'reliability_resunet_mixture.py', str(lead), 'baseline', date_set]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return lead, date_set, result.returncode, result.stdout[-1500:], result.stderr[-1500:]


with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(run_one, *job) for job in jobs]
    for future in as_completed(futures):
        lead, date_set, rc, out, err = future.result()
        status = 'OK' if rc == 0 else 'FAILED'
        print(f'--- lead={lead}h date_set={date_set}: {status} ---')
        if rc != 0:
            print(err)

print('Done.')
