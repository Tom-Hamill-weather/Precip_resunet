"""
run_crosslead_resume.py -- resumes the cross-lead check from
run_multiseed_validation_overnight.py after discovering a second data
prerequisite (training-sample netCDFs at lead24h/48h lack the
texture-derived fields needed for --texture training; only the eval-side
texture .nc files had been regenerated before the first run).

Reuses run_multiseed_validation_overnight.py's tested run_step/train_variant/
eval_variant functions directly (import + monkeypatch STATUS_PATH/_status_lines
so this writes to its own status file rather than clobbering the original
run's history in OVERNIGHT_STATUS.md).

Sequence per lead in CROSS_LEADS:
  1. Regenerate prob_MRMS_samples_*_lead{clead}h.nc via sample_6hourly_prob_mrms.py
     (now unblocked -- the graf_texture .nc files it reads were already
     generated in the first run).
  2. Train texture_gru_seed999 (the nominal winning variant from the lead12h
     multi-seed/bootstrap comparison -- see project_6h_mlp_multiseed_validation.md).
  3. Evaluate it.
  4. Bootstrap CI on the result.

Tom Hamill / Claude, Aug 2026
"""

import os
import subprocess
import run_multiseed_validation_overnight as base

REPO_DIR = base.REPO_DIR
STATUS_PATH_CROSSLEAD = os.path.join(REPO_DIR, 'OVERNIGHT_STATUS_CROSSLEAD.md')
base.STATUS_PATH = STATUS_PATH_CROSSLEAD
base._status_lines = []

CROSS_LEADS = [24, 48]
WINNER_LOSS = None          # texture_gru == plain NLL, no loss-variant CLI arg
WINNER_SEED = 999


def regen_samples(clead, timeout_sec=6000):
    out_dir = '/data/resnet_data/prob_samples'
    check_month_file = os.path.join(out_dir, f'prob_MRMS_samples_202501_lead{clead}h.nc')
    if os.path.exists(check_month_file):
        from netCDF4 import Dataset
        with Dataset(check_month_file) as ds:
            if 'sample_wet_area_fraction' in ds.variables:
                base.status(f'SKIP sample regen for lead{clead}h (texture fields already present)')
                return True
    cmd = ['python', 'sample_6hourly_prob_mrms.py', '2025010100', '2025123118', str(clead)]
    ok, rc = base.run_step(cmd, f'sample_regen_lead{clead}h.log', timeout_sec)
    return ok


def main():
    base.status('=== Cross-lead resume driver starting ===')
    base.status(f'PID={os.getpid()}  cross leads={CROSS_LEADS}  winner=texture_gru (plain NLL)  seed={WINNER_SEED}')

    for clead in CROSS_LEADS:
        if not regen_samples(clead):
            base.status(f'ABORTING lead{clead}h: sample regeneration failed, cannot train --texture without it')
            continue
        suffix = base.train_variant(clead, WINNER_LOSS, WINNER_SEED)
        if suffix:
            base.eval_variant(clead, suffix)
            base.run_step(['python', 'bootstrap_ci_6hourly_mlp.py', str(clead), suffix, '--nboot=1000'],
                          f'bootstrap_ci_lead{clead}h.log', timeout_sec=1800)

    base.status('=== Cross-lead resume driver finished all stages ===')


if __name__ == '__main__':
    main()
