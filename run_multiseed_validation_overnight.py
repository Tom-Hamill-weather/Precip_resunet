"""
run_multiseed_validation_overnight.py -- unattended master driver for the
multi-seed / bootstrap-CI / cross-lead validation of the 6h MLP GRU result.

Context (see project_6h_mlp_multiseed_validation.md in Claude's memory for
the full history): the GRU+texture+hurdle-loss checkpoint looked like a
clean sweep at lead12h, but every decision this session was validated
against the same day12-end-2025 test window across 8+ rounds -- a real
multiple-comparisons risk. Tom approved: (1) multi-seed retraining of both
the hurdle-loss and plain-NLL GRU recipes + bootstrap CI on the results, to
check whether the result is reproducible; (2) a cross-lead spot-check at
24h/48h using whichever loss variant wins, once (1) is in. Meant to run
fully unattended overnight -- launch this via setsid/nohup (see the
launch command in the accompanying memory note), not as a normal foreground
job, since the whole point is surviving the controlling session/terminal
disappearing.

Design principles:
- Every external command runs via run_step(), which never raises -- a
  failure is logged and the driver moves on, so one bad stage doesn't
  waste the rest of the night.
- Status is rewritten to OVERNIGHT_STATUS.md after every single stage,
  not just at the end, so a look-in at any point shows real progress.
- Stdout/stderr for each subprocess streams straight to its own log file
  (not captured in memory), both for safety on long runs and so `tail -f`
  works for live inspection.

Tom Hamill / Claude, Aug 2026
"""

import os
import re
import sys
import time
import subprocess
import numpy as np
import _pickle as cPickle
from datetime import datetime, timezone

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(REPO_DIR, 'overnight_logs')
STATUS_PATH = os.path.join(REPO_DIR, 'OVERNIGHT_STATUS.md')
RELIA_DIR  = '/data/resnet_data/relia'

CLEAD_MAIN = 12
CROSS_LEADS = [24, 48]
CROSS_SEED = 999

# (variant_label, train_args, is_new_training_needed)
# variant_label matches reliability_6hourly_mlp_3panel.py's freeform suffix arg exactly.
HURDLE_SEEDS = [999, 2026]   # 999 already trained (smoke test); 2026 is new. Plus the pre-existing unseeded "hurdle_texture_gru" original (not listed here, already trained+evaluated).
NLL_SEEDS    = [999, 2026]   # both new

os.makedirs(LOG_DIR, exist_ok=True)


def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


_status_lines = []


def status(msg, also_print=True):
    line = f'[{now_utc()}] {msg}'
    _status_lines.append(line)
    if also_print:
        print(line, flush=True)
    with open(STATUS_PATH, 'w') as fh:
        fh.write('# Overnight multi-seed validation run -- live status\n\n')
        fh.write('\n'.join(_status_lines))
        fh.write('\n')


def run_step(cmd, log_name, timeout_sec, cwd=REPO_DIR):
    """Run a subprocess, streaming output to LOG_DIR/log_name. Never raises;
    returns (ok: bool, returncode_or_None)."""
    log_path = os.path.join(LOG_DIR, log_name)
    status(f'START  {log_name}  cmd={" ".join(cmd)}')
    t0 = time.time()
    try:
        with open(log_path, 'w') as logfh:
            proc = subprocess.run(cmd, cwd=cwd, stdout=logfh, stderr=subprocess.STDOUT,
                                  timeout=timeout_sec)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            status(f'DONE   {log_name}  ({elapsed/60:.1f} min)')
            return True, 0
        else:
            status(f'FAILED {log_name}  rc={proc.returncode}  ({elapsed/60:.1f} min)  '
                  f'see {log_path}')
            return False, proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status(f'TIMEOUT {log_name}  after {elapsed/60:.1f} min (limit {timeout_sec/60:.0f} min)')
        return False, None
    except Exception as e:
        status(f'EXCEPTION {log_name}: {e!r}')
        return False, None


def checkpoint_exists(clead, variant):
    path = os.path.join(REPO_DIR, 'mlp_trainings', f'6h_mlp_lead{clead}h_{variant}.pth')
    return os.path.exists(path)


def percache_exists(clead, variant):
    path = os.path.join(RELIA_DIR, f'relia_6h_MLP_3panel_percache_lead{clead}h_{variant}.cPick')
    return os.path.exists(path)


def train_variant(clead, loss_variant, seed, timeout_sec=7200):
    """loss_variant is 'hurdle' or None (plain NLL). Returns the checkpoint
    variant-suffix string used by reliability_6hourly_mlp_3panel.py."""
    suffix = (f'{loss_variant}_' if loss_variant else '') + f'texture_gru_seed{seed}'
    if checkpoint_exists(clead, suffix):
        status(f'SKIP training {suffix} (checkpoint already exists)')
        return suffix
    cmd = ['python', 'train_6hourly_mlp.py', str(clead)]
    if loss_variant:
        cmd.append(loss_variant)
    cmd += ['--texture', '--gru', f'--seed={seed}']
    ok, rc = run_step(cmd, f'train_lead{clead}h_{suffix}.log', timeout_sec)
    return suffix if ok else None


def eval_variant(clead, suffix, timeout_sec=9000):
    if percache_exists(clead, suffix):
        status(f'SKIP eval {suffix} (percache already exists -- assuming complete run; '
              f'partial percaches from a crashed run would resume via the per-date cache '
              f'anyway if re-run, so this skip is conservative not risky)')
        return True
    cmd = ['python', 'reliability_6hourly_mlp_3panel.py', str(clead), suffix]
    ok, rc = run_step(cmd, f'eval_lead{clead}h_{suffix}.log', timeout_sec)
    return ok


def launch_texture_gen(clead, workers=6, timeout_sec=14400):
    """Non-blocking: returns a Popen handle. CPU/IO-bound, safe to run
    alongside GPU training/eval (see control_save_graf_texture_features.py's
    own docstring for why)."""
    tex_dir = '/data/resnet_data/graf_texture'
    # crude completeness check: expect ~1460 files/year at 4-cycles/day for a full 2025 backfill
    existing = [f for f in os.listdir(tex_dir) if f.endswith(f'_{clead}_graf_texture_features.nc')]
    if len(existing) > 1400:
        status(f'SKIP texture-gen for lead{clead}h ({len(existing)} files already present)')
        return None
    log_path = os.path.join(LOG_DIR, f'texture_gen_lead{clead}h.log')
    cmd = ['python', 'control_save_graf_texture_features.py',
          '2025010100', '2025123118', str(clead), str(workers)]
    status(f'BACKGROUND START texture-gen lead{clead}h  cmd={" ".join(cmd)}')
    logfh = open(log_path, 'w')
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=logfh, stderr=subprocess.STDOUT)
    return proc


def wait_for(proc_list, label, poll_sec=60, timeout_sec=14400):
    if not proc_list:
        return
    t0 = time.time()
    status(f'Waiting for background jobs: {label}')
    while True:
        alive = [p for p in proc_list if p is not None and p.poll() is None]
        if not alive:
            break
        if time.time() - t0 > timeout_sec:
            status(f'TIMEOUT waiting for {label} after {timeout_sec/60:.0f} min -- '
                  f'proceeding anyway (killing stragglers)')
            for p in alive:
                p.kill()
            break
        time.sleep(poll_sec)
    elapsed = time.time() - t0
    status(f'Background jobs done: {label}  ({elapsed/60:.1f} min)')


def load_bootstrap_summary(clead):
    path = os.path.join(RELIA_DIR, f'bootstrap_ci_summary_lead{clead}h.cPick')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as fh:
        return cPickle.load(fh)


def pick_winning_variant(summary):
    """Group variants by recipe (strip _seed{N}), average point-estimate
    mean|dev| across seeds within a group and across the 3 thresholds, pick
    the lower sum. Returns the winning base recipe name ('hurdle_texture_gru'
    or 'texture_gru'), or None if not enough data."""
    groups = {}
    for v, result in summary.items():
        base = re.sub(r'_seed\d+$', '', v)
        groups.setdefault(base, []).append(result)
    scores = {}
    for base, results in groups.items():
        if not results:
            continue
        per_result_sum = [float(np.nansum(r['point_mdev'])) for r in results]
        scores[base] = float(np.mean(per_result_sum))
    if not scores:
        return None
    winner = min(scores, key=scores.get)
    status(f'Winning-variant scores (lower=better, sum of mean|dev| across 3 thresholds, '
          f'averaged over seeds): {scores}  -> winner={winner}')
    return winner


def main():
    status('=== Overnight multi-seed validation driver starting ===')
    status(f'PID={os.getpid()}  main clead={CLEAD_MAIN}  cross-check leads={CROSS_LEADS}')

    # Stage 0: launch texture-feature generation for the cross-lead leads in
    # the background now, so it overlaps with the GPU-bound work below
    # rather than serializing after it.
    texture_procs = [launch_texture_gen(clead) for clead in CROSS_LEADS]

    # Stage 1+2: train (if needed) and evaluate each seed variant at lead12h.
    all_variants = ['hurdle_texture_gru']  # pre-existing, already trained+evaluated
    for seed in HURDLE_SEEDS:
        suffix = train_variant(CLEAD_MAIN, 'hurdle', seed)
        if suffix:
            all_variants.append(suffix)
    for seed in NLL_SEEDS:
        suffix = train_variant(CLEAD_MAIN, None, seed)
        if suffix:
            all_variants.append(suffix)

    status(f'Variants to evaluate at lead{CLEAD_MAIN}h: {all_variants}')
    for suffix in all_variants:
        eval_variant(CLEAD_MAIN, suffix)

    # Stage 3: bootstrap CI across everything evaluated so far.
    cmd = ['python', 'bootstrap_ci_6hourly_mlp.py', str(CLEAD_MAIN)] + all_variants + ['--nboot=1000']
    run_step(cmd, f'bootstrap_ci_lead{CLEAD_MAIN}h.log', timeout_sec=1800)

    # Stage 4: decide winning loss variant.
    summary = load_bootstrap_summary(CLEAD_MAIN)
    winner_base = pick_winning_variant(summary)
    if winner_base is None:
        status('Could not determine a winning variant (no bootstrap summary data) -- '
              'defaulting to hurdle_texture_gru for the cross-lead check')
        winner_base = 'hurdle_texture_gru'
    winner_loss = 'hurdle' if winner_base.startswith('hurdle_') else None
    status(f'Cross-lead check will use: loss={winner_loss or "nll"}  base={winner_base}  seed={CROSS_SEED}')

    # Stage 5: make sure texture data for the cross-lead leads is ready.
    wait_for(texture_procs, 'texture-feature generation (24h/48h)')

    # Stage 6: cross-lead train + eval, winning variant only, single seed.
    for clead in CROSS_LEADS:
        suffix = train_variant(clead, winner_loss, CROSS_SEED)
        if suffix:
            eval_variant(clead, suffix)
            run_step(['python', 'bootstrap_ci_6hourly_mlp.py', str(clead), suffix, '--nboot=1000'],
                     f'bootstrap_ci_lead{clead}h.log', timeout_sec=1800)

    status('=== Overnight multi-seed validation driver finished all stages ===')


if __name__ == '__main__':
    main()
