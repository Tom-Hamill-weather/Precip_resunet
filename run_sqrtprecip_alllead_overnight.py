"""
run_sqrtprecip_alllead_overnight.py -- extend the sqrt-vs-log1p-precip
transform comparison for the `sample_graf_precip_6h` feature (see
project_6h_mlp_sqrtprecip_and_bernstein_seed2026 in Claude's memory) from
lead12h-only (where it was tested at seeds 999/2026 and came out a wash on
BSS but meaningfully better/tighter on 10mm calibration) to all 8 six-hourly
MLP lead times: 6, 12, 18, 24, 30, 36, 42, 48h.

Scope (Tom asked to run this autonomously; these are judgment calls made to
keep it proportional, matching precedent already set by Tom for the earlier
24h/48h cross-lead check in project_6h_mlp_multiseed_validation):
  - lead12h is already fully done (both seeds 999/2026, both transforms,
    all evaluated) -- this driver does NOT re-touch it, only reads its
    existing results into the final report.
  - Every other lead gets a SINGLE seed (999), matching the cross-lead
    precedent ("deliberately scoped to 1 seed per lead to control cost").
  - Leads 24h/48h already have the log1p (default) texture_gru_seed999
    checkpoint trained+evaluated from that earlier cross-lead check -- only
    the sqrt-precip variant is new there.
  - Leads 6h/18h/30h/36h/42h have NO texture data at all yet (full-domain
    texture netCDFs only ever existed for 12/24/48h) -- both the log1p
    baseline and the sqrt-precip variant are new there, and a texture-gen +
    training-sample-regen prerequisite pipeline has to run first.

Computational efficiency measures taken (per Tom's explicit ask to look for
these before running everything):
  1. reliability_6hourly_mlp_3panel.py was refactored (see VariantState
     class there) to evaluate multiple variants of the same architecture in
     ONE pass over the ~900 test dates, sharing the per-date netCDF reads
     (params/MRMS/control/copula/texture -- the dominant per-date cost,
     comparable in magnitude to the model forward pass itself) instead of
     re-reading them once per variant. Verified bit-identical to the
     pre-refactor per-variant output before relying on it here. This
     driver's joint_eval() always passes both variants for a lead in one
     call so this saving is actually realized.
  2. The CPU/IO-bound data-prep stage (texture-gen + sample-regen) for the
     5 texture-less leads is pipelined with the GPU-bound train+eval stage:
     while lead N's GPU work runs, lead N+1's data prep already runs in the
     background, rather than serializing the two resource types.

Reuses run_multiseed_validation_overnight.py's tested run_step/status/
checkpoint_exists/wait_for functions (monkeypatches STATUS_PATH first so
this writes its own status file rather than clobbering that run's history).

Launch this via setsid/nohup+disown (see the launch note in
project_6h_mlp_sqrtprecip_alllead memory), not as a normal foreground job --
the whole point is surviving the controlling session ending.

Tom Hamill / Claude, Aug 2026
"""

import os
import sys
import subprocess
import _pickle as cPickle
from datetime import datetime, timezone
import run_multiseed_validation_overnight as base

REPO_DIR = base.REPO_DIR
STATUS_PATH = os.path.join(REPO_DIR, 'OVERNIGHT_STATUS_SQRTPRECIP_ALLLEAD.md')
base.STATUS_PATH = STATUS_PATH
base._status_lines = []

RELIA_DIR = base.RELIA_DIR
SEED = 999
DONE_LEAD = 12
NEEDS_TEXTURE_LEADS = [6, 18, 30, 36, 42]
HAS_TEXTURE_LEADS = [24, 48]
ALL_NEW_LEADS = NEEDS_TEXTURE_LEADS + HAS_TEXTURE_LEADS
REPORT_PATH = os.path.join(REPO_DIR, 'OVERNIGHT_REPORT_SQRTPRECIP_ALLLEAD.md')

MONTHS = list(range(1, 13))
# Leads 12/24/48 (already fully backfilled) have 1388 files each out of a
# possible 1460 (365 days x 4 cycles) -- the shortfall is the known ~5%
# sporadic-GRAF-gap rate seen throughout this project, not a bug. Use a
# threshold below that observed-complete count so an already-finished lead
# is never mistaken for incomplete and re-run.
TEXTURE_MIN_FILES = 1300


def texture_ready(clead):
    tex_dir = '/data/resnet_data/graf_texture'
    n = len([f for f in os.listdir(tex_dir) if f.endswith(f'_{clead}_graf_texture_features.nc')])
    return n >= TEXTURE_MIN_FILES


def samples_have_texture_fields(clead):
    import netCDF4 as nc
    path = f'/data/resnet_data/prob_samples/prob_MRMS_samples_202501_lead{clead}h.nc'
    if not os.path.exists(path):
        return False
    with nc.Dataset(path) as f:
        return 'sample_graf_precip_6h' in f.variables


def prep_lead_data(clead, n_workers=8, timeout_texture=14400, timeout_sample_per_month=6000):
    """Blocking. Used for the very first lead in the pipeline (nothing to
    overlap with yet), so it's fine to go through base.run_step/status."""
    if not texture_ready(clead):
        base.run_step(['python', 'control_save_graf_texture_features.py',
                       '2025010100', '2025123118', str(clead), str(n_workers)],
                      f'texture_gen_lead{clead}h.log', timeout_texture)
    else:
        base.status(f'SKIP texture-gen for lead{clead}h (already >= {TEXTURE_MIN_FILES} files)')

    if not samples_have_texture_fields(clead):
        for mm in MONTHS:
            cmd = ['python', 'sample_6hourly_prob_mrms.py',
                  f'2025{mm:02d}0100', f'2025{mm:02d}0918', str(clead)]
            base.run_step(cmd, f'sample_regen_lead{clead}h_month{mm:02d}.log',
                          timeout_sample_per_month)
    else:
        base.status(f'SKIP sample-regen for lead{clead}h (texture fields already present)')


def prep_lead_data_bg(clead):
    """Same steps as prep_lead_data, but plain print() instead of
    base.status(). Called from a SEPARATE backgrounded subprocess (see
    launch_prep_background) -- it must not touch the shared STATUS_PATH
    file, since that file is rewritten wholesale from an in-memory list on
    every status() call, and this process's list would be empty/incomplete
    relative to the main driver's. Two processes each doing a full-file
    rewrite from their own partial view would clobber each other's history.
    This process's own stdout (redirected to a per-lead log file by the
    caller) is the only record of its progress."""
    if not texture_ready(clead):
        print(f'START texture-gen lead{clead}h', flush=True)
        subprocess.run(['python', 'control_save_graf_texture_features.py',
                        '2025010100', '2025123118', str(clead), '6'],
                       cwd=REPO_DIR, timeout=14400)
        print(f'DONE texture-gen lead{clead}h', flush=True)
    else:
        print(f'SKIP texture-gen lead{clead}h (already present)', flush=True)

    if not samples_have_texture_fields(clead):
        for mm in MONTHS:
            print(f'START sample-regen lead{clead}h month{mm:02d}', flush=True)
            subprocess.run(['python', 'sample_6hourly_prob_mrms.py',
                            f'2025{mm:02d}0100', f'2025{mm:02d}0918', str(clead)],
                           cwd=REPO_DIR, timeout=6000)
            print(f'DONE sample-regen lead{clead}h month{mm:02d}', flush=True)
    else:
        print(f'SKIP sample-regen lead{clead}h (texture fields already present)', flush=True)


def launch_prep_background(clead):
    log_path = os.path.join(base.LOG_DIR, f'prep_lead{clead}h_wrapper.log')
    logfh = open(log_path, 'w')
    cmd = ['python', '-c',
          f'import run_sqrtprecip_alllead_overnight as m; m.prep_lead_data_bg({clead})']
    base.status(f'BACKGROUND START data-prep lead{clead}h (log: {log_path})')
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=logfh, stderr=subprocess.STDOUT)
    return proc


def train_texture_variant(clead, sqrt, seed=SEED, timeout_sec=3600):
    suffix = f'texture_gru_sqrtprecip_seed{seed}' if sqrt else f'texture_gru_seed{seed}'
    if base.checkpoint_exists(clead, suffix):
        base.status(f'SKIP training {suffix} (checkpoint already exists)')
        return suffix
    cmd = ['python', 'train_6hourly_mlp.py', str(clead), '--texture', '--gru', f'--seed={seed}']
    if sqrt:
        cmd.append('--sqrt-precip')
    ok, rc = base.run_step(cmd, f'train_lead{clead}h_{suffix}.log', timeout_sec)
    return suffix if ok else None


def joint_eval(clead, variants, timeout_sec=9000):
    """One call, comma-joined variants -- shares per-date I/O reads across
    whichever variants still need (re)computing."""
    cmd = ['python', 'reliability_6hourly_mlp_3panel.py', str(clead), ','.join(variants)]
    ok, rc = base.run_step(cmd, f'eval_lead{clead}h_joint.log', timeout_sec)
    return ok


def bootstrap(clead, variants, timeout_sec=1800):
    cmd = ['python', 'bootstrap_ci_6hourly_mlp.py', str(clead)] + variants + ['--nboot=1000']
    base.run_step(cmd, f'bootstrap_ci_lead{clead}h_sqrtprecip_alllead.log', timeout_sec)


def process_lead(clead):
    base.status(f'=== Starting lead{clead}h ===')
    variants = []
    log1p_suffix = train_texture_variant(clead, sqrt=False)
    if log1p_suffix:
        variants.append(log1p_suffix)
    sqrt_suffix = train_texture_variant(clead, sqrt=True)
    if sqrt_suffix:
        variants.append(sqrt_suffix)
    if not variants:
        base.status(f'lead{clead}h: no variants trained successfully, skipping eval')
        return
    joint_eval(clead, variants)
    bootstrap(clead, variants)
    base.status(f'=== Finished lead{clead}h ===')


def load_pick(clead, variant):
    variant_suffix = f'_{variant}' if variant else ''
    fname = os.path.join(
        RELIA_DIR,
        f'relia_6h_MLP_3panel_q0.6_2025011200_to_2025123118_lead{clead}h{variant_suffix}.cPick')
    if not os.path.exists(fname):
        return None
    with open(fname, 'rb') as fh:
        return cPickle.load(fh)


def summarize_pick(pick):
    if pick is None:
        return 'N/A (eval never completed / percache missing)'
    return {float(t): round(float(pick['BSS'][i]), 4) for i, t in enumerate(pick['pthresholds'])}


def write_report():
    lines = []
    lines.append('# sqrt-precip vs log1p-precip -- all-lead extension report')
    lines.append(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}')
    lines.append('')
    lines.append('Extends the lead12h-only sqrt-vs-log1p comparison to all 8 6-hourly MLP '
                f'leads. Single seed ({SEED}) per newly-added lead (6/18/24/30/36/42/48h), '
                'matching the precedent already set for the 24h/48h cross-lead check -- '
                'lead12h alone has 2 seeds (999/2026), done in an earlier session.')
    lines.append('')
    lines.append('BSS shown by threshold (mm): {0.25: ..., 2.5: ..., 10.0: ...}. See the '
                'bootstrap CI logs (overnight_logs/) for mean|dev| and confidence intervals -- '
                'BSS alone is not the whole calibration story.')
    lines.append('')
    for clead in [DONE_LEAD] + ALL_NEW_LEADS:
        lines.append(f'## lead{clead}h')
        if clead == DONE_LEAD:
            for seed in (999, 2026):
                base_v = f'texture_gru_seed{seed}'
                sqrt_v = f'texture_gru_sqrtprecip_seed{seed}'
                lines.append(f'- seed{seed}: log1p BSS={summarize_pick(load_pick(clead, base_v))}')
                lines.append(f'           sqrt  BSS={summarize_pick(load_pick(clead, sqrt_v))}')
            lines.append('  (done in an earlier session -- see project_6h_mlp_sqrtprecip_and_'
                         'bernstein_seed2026 memory / OVERNIGHT_REPORT.md for full detail)')
        else:
            base_v = f'texture_gru_seed{SEED}'
            sqrt_v = f'texture_gru_sqrtprecip_seed{SEED}'
            lines.append(f'- seed{SEED}: log1p BSS={summarize_pick(load_pick(clead, base_v))}')
            lines.append(f'           sqrt  BSS={summarize_pick(load_pick(clead, sqrt_v))}')
            lines.append(f'  Full CI/mean|dev|: overnight_logs/bootstrap_ci_lead{clead}h_'
                         'sqrtprecip_alllead.log')
        lines.append('')
    lines.append('Reminder: a single seed\'s calibration number is not fully reliable on its '
                'own (established in the earlier multi-seed validation work) -- these new '
                'leads are single-seed by design (cost-controlled, matching the 24h/48h '
                'cross-lead precedent), so treat mean|dev| differences at the new leads as '
                'suggestive, not conclusive, unless/until a second seed is added.')
    with open(REPORT_PATH, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    base.status(f'Wrote report to {REPORT_PATH}')


def main():
    base.status('=== Overnight sqrt-precip-all-leads driver starting ===')
    base.status(f'PID={os.getpid()}  needs-texture leads={NEEDS_TEXTURE_LEADS}  '
               f'has-texture leads={HAS_TEXTURE_LEADS}  seed={SEED}  '
               f'(lead{DONE_LEAD}h already complete, both seeds/variants -- not re-touched)')

    # Prep lead[0] blocking (nothing to overlap with yet); from then on,
    # prep lead[i+1] in the background while lead[i]'s GPU-bound train+eval
    # runs, so the CPU-bound prep stage and the GPU-bound stage overlap
    # across leads instead of serializing.
    pending_prep = None
    for i, clead in enumerate(NEEDS_TEXTURE_LEADS):
        if pending_prep is not None:
            base.wait_for([pending_prep], f'data-prep lead{clead}h', timeout_sec=18000)
        else:
            prep_lead_data(clead)

        if i + 1 < len(NEEDS_TEXTURE_LEADS):
            pending_prep = launch_prep_background(NEEDS_TEXTURE_LEADS[i + 1])
        else:
            pending_prep = None

        process_lead(clead)

    for clead in HAS_TEXTURE_LEADS:
        process_lead(clead)

    write_report()
    base.status('=== Overnight sqrt-precip-all-leads driver finished all stages ===')


if __name__ == '__main__':
    main()
