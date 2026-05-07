"""
control_train_resunet_gamma_mixture.py

For each combination of IC date and lead time:
  1. Save patches via save_patched_GRAF_MRMS_GFS2.py
  2. Train via pytorch_train_resunet_gamma_mixture.py

  - IC dates: 2025010100, 2025020100, 2025040100, 2025050100,
              2025070100, 2025080100, 2025100100, 2025110100
  - Lead times: 3, 6, 9, ..., 48 hours (every 3 hours)

Usage:
    python control_train_resunet_gamma_mixture.py
    python control_train_resunet_gamma_mixture.py 2025070100 9   # restart from date/lead

Tom Hamill, April 2026
"""

import subprocess
import sys

IC_DATES = [
    '2025010100',
    '2025020100',
    '2025040100',
    '2025050100',
    '2025070100',
    '2025080100',
    '2025100100',
    '2025110100',
]

LEAD_TIMES = list(range(3, 49, 3))  # 3, 6, 9, ..., 48


def run(cmd):
    print(f'Running: {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    start_date = sys.argv[1] if len(sys.argv) > 1 else None
    start_lead = int(sys.argv[2]) if len(sys.argv) > 2 else None
    skipping = start_date is not None

    total = len(IC_DATES) * len(LEAD_TIMES)
    count = 0
    for date in IC_DATES:
        for lead in LEAD_TIMES:
            count += 1
            if skipping:
                if date == start_date and lead == start_lead:
                    skipping = False
                else:
                    print(f'Skipping {date} {lead}h', flush=True)
                    continue

            print(f'\n[{count}/{total}] date={date}  lead={lead}h', flush=True)

            patch_cmd = ['python', 'save_patched_GRAF_MRMS_GFS2.py', date, str(lead)]
            rc = run(patch_cmd)
            if rc != 0:
                print(f'ERROR: patch saving failed for date={date} lead={lead}h '
                      f'(exit code {rc})', file=sys.stderr)
                sys.exit(rc)

            train_cmd = ['python', 'pytorch_train_resunet_gamma_mixture.py', date, str(lead)]
            rc = run(train_cmd)
            if rc != 0:
                print(f'ERROR: training failed for date={date} lead={lead}h '
                      f'(exit code {rc})', file=sys.stderr)
                sys.exit(rc)

    print(f'\nAll {total} patch+training runs completed successfully.')


if __name__ == '__main__':
    main()
