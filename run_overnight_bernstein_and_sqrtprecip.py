"""
run_overnight_bernstein_and_sqrtprecip.py -- autonomous overnight driver.
Tom is going to bed; this must run unattended and produce a clear report by
morning, per his explicit instruction (2026-08-10).

Sequence:
  1. Wait for the already-running Bernstein seed2026 eval (launched
     interactively earlier this session, PID tracked below) to finish.
  2. Bootstrap CI for it, so it's directly comparable to seed999's
     already-recorded result -- answers whether the first Bernstein seed's
     disappointing calibration numbers were real or seed noise.
  3. Train + evaluate + bootstrap the sqrt-precip variant of the current
     winning recipe (plain NLL+GRU / texture_gru), seeds 999 and 2026 --
     matching the existing log1p baseline's exact seeds for a clean paired
     comparison. Scope confirmed with Tom before he went offline: texture_gru
     only, not hurdle+GRU or the Bernstein head (those would roughly double
     or triple the overnight compute for a question that's specifically
     about this one feature's transform, on the current best recipe).
  4. Write a single, plain-language OVERNIGHT_REPORT.md at the repo root
     with everything, since Tom won't be watching live and shouldn't have
     to dig through logs to find the numbers.

Reuses run_multiseed_validation_overnight.py's tested run_step/eval_variant/
checkpoint_exists/percache_exists/status functions directly (monkeypatches
STATUS_PATH/_status_lines first so this writes its own status file rather
than clobbering earlier runs' history).

Tom Hamill / Claude, Aug 2026
"""

import os
import time
import _pickle as cPickle
from datetime import datetime, timezone
import run_multiseed_validation_overnight as base

REPO_DIR = base.REPO_DIR
STATUS_PATH = os.path.join(REPO_DIR, 'OVERNIGHT_STATUS_BERNSTEIN_SQRTPRECIP.md')
base.STATUS_PATH = STATUS_PATH
base._status_lines = []

RELIA_DIR = base.RELIA_DIR
CLEAD = 12
SEEDS = [999, 2026]
BERNSTEIN_SEED2026_PID = 2635832
REPORT_PATH = os.path.join(REPO_DIR, 'OVERNIGHT_REPORT.md')


def wait_for_pid(pid, label, poll_sec=90, timeout_sec=21600):
    t0 = time.time()
    base.status(f'Waiting for external process to finish: {label} (pid {pid})')
    while True:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        if not alive:
            break
        if time.time() - t0 > timeout_sec:
            base.status(f'TIMEOUT waiting for {label} after {timeout_sec/60:.0f} min -- proceeding anyway')
            break
        time.sleep(poll_sec)
    base.status(f'Done waiting for {label} ({(time.time()-t0)/60:.1f} min)')


def train_sqrtprecip(seed, timeout_sec=3600):
    suffix = f'texture_gru_sqrtprecip_seed{seed}'
    if base.checkpoint_exists(CLEAD, suffix):
        base.status(f'SKIP training {suffix} (checkpoint already exists)')
        return suffix
    cmd = ['python', 'train_6hourly_mlp.py', str(CLEAD), '--texture', '--gru',
          '--sqrt-precip', f'--seed={seed}']
    ok, rc = base.run_step(cmd, f'train_lead{CLEAD}h_{suffix}.log', timeout_sec)
    return suffix if ok else None


def load_pick(clead, variant):
    variant_suffix = f'_{variant}' if variant else ''
    fname = os.path.join(RELIA_DIR,
                         f'relia_6h_MLP_3panel_q0.6_2025011200_to_2025123118_lead{clead}h{variant_suffix}.cPick')
    if not os.path.exists(fname):
        return None
    with open(fname, 'rb') as fh:
        return cPickle.load(fh)


def summarize_pick(pick):
    if pick is None:
        return 'N/A (eval never completed / percache missing)'
    return {float(t): round(float(pick['BSS'][i]), 4) for i, t in enumerate(pick['pthresholds'])}


def main():
    base.status('=== Overnight Bernstein-seed2026-wait + sqrt-precip driver starting ===')
    base.status(f'PID={os.getpid()}  clead={CLEAD}  seeds={SEEDS}')

    # Stage 1: wait for the interactively-launched Bernstein seed2026 eval.
    wait_for_pid(BERNSTEIN_SEED2026_PID, 'Bernstein seed2026 eval')
    bernstein_variant = 'texture_gru_bernstein_seed2026'
    if base.percache_exists(CLEAD, bernstein_variant):
        base.run_step(['python', 'bootstrap_ci_6hourly_mlp.py', str(CLEAD), bernstein_variant, '--nboot=1000'],
                      f'bootstrap_ci_lead{CLEAD}h_bernstein_seed2026_followup.log', timeout_sec=1800)
    else:
        base.status(f'WARNING: {bernstein_variant} percache not found after waiting -- '
                   f'eval may have failed; skipping its bootstrap CI')

    # Stage 2: sqrt-precip variant of the winning texture_gru recipe, both seeds.
    sqrt_variants = []
    for seed in SEEDS:
        suffix = train_sqrtprecip(seed)
        if suffix:
            sqrt_variants.append(suffix)

    base.status(f'Evaluating sqrt-precip variants: {sqrt_variants}')
    for suffix in sqrt_variants:
        base.eval_variant(CLEAD, suffix)

    if sqrt_variants:
        base.run_step(['python', 'bootstrap_ci_6hourly_mlp.py', str(CLEAD)] + sqrt_variants + ['--nboot=1000'],
                      f'bootstrap_ci_lead{CLEAD}h_sqrtprecip.log', timeout_sec=1800)

    write_report(bernstein_variant, sqrt_variants)
    base.status('=== Overnight driver finished all stages ===')


def write_report(bernstein_variant, sqrt_variants):
    lines = []
    lines.append('# Overnight report -- Bernstein seed2026 + sqrt-precip test')
    lines.append(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}')
    lines.append('')
    lines.append('## Bernstein-quantile head, seed2026 (reproducibility check vs. seed999)')
    lines.append('seed999 first result was disappointing: calibration (mean|dev|) worse than the')
    lines.append('plain-NLL+GRU baseline at every threshold, BSS roughly comparable but slightly lower.')
    lines.append('This seed checks whether that was real or an unlucky seed, same as we found for GRU earlier.')
    pick999 = load_pick(CLEAD, 'texture_gru_bernstein_seed999')
    pick2026 = load_pick(CLEAD, bernstein_variant)
    lines.append(f'- seed999  BSS by threshold: {summarize_pick(pick999)}')
    lines.append(f'- seed2026 BSS by threshold: {summarize_pick(pick2026)}')
    lines.append('Full CI/mean|dev| numbers: overnight_logs/bootstrap_ci_lead12h_bernstein_seed2026_followup.log')
    lines.append('')
    lines.append('## sqrt-precip vs. log1p-precip for graf_precip_6h (plain NLL+GRU / texture_gru recipe)')
    lines.append('Scope per Tom: texture_gru only, seeds 999/2026 matching the existing log1p baseline exactly.')
    for seed in SEEDS:
        baseline_variant = f'texture_gru_seed{seed}'
        sqrt_variant = f'texture_gru_sqrtprecip_seed{seed}'
        pick_base = load_pick(CLEAD, baseline_variant)
        pick_sqrt = load_pick(CLEAD, sqrt_variant)
        lines.append(f'- seed{seed}: log1p BSS={summarize_pick(pick_base)}')
        lines.append(f'           sqrt  BSS={summarize_pick(pick_sqrt)}')
    lines.append('Full CI/mean|dev| numbers: overnight_logs/bootstrap_ci_lead12h_sqrtprecip.log')
    lines.append('')
    lines.append('Reminder: BSS alone is not the whole story -- check mean|dev|/reliability in the')
    lines.append('bootstrap logs above too, and remember the whole-session lesson that a single seed\'s')
    lines.append('calibration number is not reliable on its own (established via the multi-seed check earlier).')
    with open(REPORT_PATH, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    base.status(f'Wrote report to {REPORT_PATH}')


if __name__ == '__main__':
    main()
