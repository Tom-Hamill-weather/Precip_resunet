#!/bin/bash
# run_may_jul_aug_2026_backfill.sh
#
# Tom asked (2026-09-09): regenerate baseline-model ("monthly by month and
# lead time dependent training") probability files for May/Jul/Aug 2026,
# every 6-h cycle (00/06/12/18Z), leads 6-48h every 6h. Backfill GFS for
# August (the one real data gap -- MRMS is NOT needed, see below), train
# month-specific checkpoints, generate inference, then tar + upload to S3.
#
# MRMS note: traced save_patched_GRAF_MRMS_GFS2.py's training-window math --
# all 4 date ranges it builds only look BACKWARD from the training anchor
# (by at most ndays_train + a few days), so training for a 2026-05/07/08
# anchor never needs MRMS data from that anchor's own month. MRMS is
# complete through 2026-07 already. No MRMS backfill needed or performed.
#
# GFS note: resunet_inference_gamma_mixture_optimized.py reads GFS RH for
# the INFERENCE date itself (not the training window), and /data/resnet_data/gfs
# has no 202608/ directory at all -- so August inference would fail without
# this backfill.
set -uo pipefail

cd /home/thamill/resnet
SELF_LOG=/data/resnet_data/run_may_jul_aug_2026_backfill.log
exec >>"$SELF_LOG" 2>&1

LOCK_FILE=/tmp/resnet_may_jul_aug_2026_backfill.lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "=== $(date -u +%FT%TZ) another instance already holds the lock -- exiting ==="
    exit 0
fi

DATES="2026050100 2026070100 2026080100"
LEADS="6 12 18 24 30 36 42 48"
NJOBS=6

echo "=== $(date -u +%FT%TZ) PIPELINE START ==="

# ---------------------------------------------------------------------------
# PHASE G: GFS backfill for August 2026 (full 0-72h build). Runs in the
# background -- only gates PHASE I's August inference calls, not patch-pool
# building or training.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE G: GFS backfill for August 2026 ==="
python fetch_gfs_extension.py --full-range 20260801 20260831 --workers 8 \
    > /data/resnet_data/gfs_aug_2026_backfill.log 2>&1 &
GFS_PID=$!
echo "PHASE G launched, PID $GFS_PID, log /data/resnet_data/gfs_aug_2026_backfill.log"

# ---------------------------------------------------------------------------
# PHASE P: patch-pool builds for the 3 anchors x 8 leads (24 jobs), NJOBS=6
# concurrent -- same convention as control_save_patched_GRAF_MRMS_GFS2.sh.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE P: patch-pool builds ==="

declare -a RUNNING_PIDS=()
throttle() {
    while true; do
        local alive=()
        for pid in "${RUNNING_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        RUNNING_PIDS=("${alive[@]}")
        [ "${#RUNNING_PIDS[@]}" -lt "$NJOBS" ] && break
        sleep 5
    done
}

P_START=$(date +%s)
for date in $DATES; do
    for lead in $LEADS; do
        throttle
        logfile="/data/resnet_data/log_patches_${date}_${lead}h.txt"
        (
            python save_patched_GRAF_MRMS_GFS2.py "$date" "$lead" > "$logfile" 2>&1
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "[$(date -u +%FT%TZ)] patch-pool done   $date lead=${lead}h"
            else
                echo "[$(date -u +%FT%TZ)] patch-pool FAILED $date lead=${lead}h (rc=$rc) -- see $logfile"
            fi
        ) &
        RUNNING_PIDS+=($!)
    done
done
wait
P_ELAPSED=$(( $(date +%s) - P_START ))
echo "=== $(date -u +%FT%TZ) PHASE P COMPLETE (${P_ELAPSED}s) ==="

# ---------------------------------------------------------------------------
# PHASE T: sequential training, one GPU job at a time (24 total).
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE T: training ==="
T_START=$(date +%s)
NFAIL_T=0
for date in $DATES; do
    for lead in $LEADS; do
        logfile="/data/resnet_data/log_train_${date}_${lead}h.txt"
        echo "[$(date -u +%FT%TZ)] training start  $date lead=${lead}h"
        python pytorch_train_resunet_gamma_mixture.py "$date" "$lead" > "$logfile" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "[$(date -u +%FT%TZ)] training done   $date lead=${lead}h"
        else
            NFAIL_T=$((NFAIL_T+1))
            echo "[$(date -u +%FT%TZ)] training FAILED $date lead=${lead}h (rc=$rc) -- see $logfile"
        fi
    done
done
T_ELAPSED=$(( $(date +%s) - T_START ))
echo "=== $(date -u +%FT%TZ) PHASE T COMPLETE (${T_ELAPSED}s, ${NFAIL_T} failed) ==="

# ---------------------------------------------------------------------------
# PHASE I: inference (baseline model only), all 3 months, every 6h cycle,
# leads 6-48h. Waits on the GFS backfill (should be long done by now).
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE I: waiting on GFS backfill (PID $GFS_PID) ==="
wait "$GFS_PID"
GFS_RC=$?
echo "=== $(date -u +%FT%TZ) GFS backfill finished, rc=$GFS_RC ==="

echo "=== $(date -u +%FT%TZ) PHASE I: inference ==="
I_START=$(date +%s)
python control_resunet_inference_2026_may_jul_aug.py 8 > /data/resnet_data/log_inference_may_jul_aug.txt 2>&1
I_RC=$?
I_ELAPSED=$(( $(date +%s) - I_START ))
if [ $I_RC -ne 0 ]; then
    echo "=== $(date -u +%FT%TZ) PHASE I FAILED (rc=$I_RC, ${I_ELAPSED}s) -- see log_inference_may_jul_aug.txt -- NOT tarring ==="
    exit 1
fi
echo "=== $(date -u +%FT%TZ) PHASE I COMPLETE (${I_ELAPSED}s) ==="

# ---------------------------------------------------------------------------
# PHASE TAR: tar the new May/Jul/Aug probs files only (not the whole probs/
# archive) and upload to s3://twc-nvidia.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE TAR: building file list ==="
FILELIST=/data/resnet_data/may_jul_aug_2026_filelist.txt
python3 -c "
from dateutils import daterange
leads = [6, 12, 18, 24, 30, 36, 42, 48]
dates = daterange('2026050100','2026053118',6) + daterange('2026070100','2026073118',6) + daterange('2026080100','2026083118',6)
with open('$FILELIST', 'w') as f:
    for d in dates:
        for l in leads:
            f.write(f'{d}_{l}_probs_gamma_mixture.nc\n')
"
# Only tar files that actually exist (tolerate any residual failed cases).
FILELIST_EXIST=/data/resnet_data/may_jul_aug_2026_filelist_exist.txt
> "$FILELIST_EXIST"
N_MISSING=0
while read -r fname; do
    if [ -f "/data/resnet_data/probs/$fname" ]; then
        echo "$fname" >> "$FILELIST_EXIST"
    else
        N_MISSING=$((N_MISSING+1))
    fi
done < "$FILELIST"
echo "$(wc -l < "$FILELIST_EXIST") files present, $N_MISSING missing"

TAR_PATH=/data/resnet_data/resnet_probs_2026_may_jul_aug.tar
S3_KEY=s3://twc-nvidia/resnet/probs_2026/resnet_probs_2026_may_jul_aug.tar

echo "=== $(date -u +%FT%TZ) PHASE TAR: tar cf $TAR_PATH ==="
tar cf "$TAR_PATH" -C /data/resnet_data/probs --files-from="$FILELIST_EXIST"
if [ $? -ne 0 ]; then
    echo "$(date -u +%FT%TZ) PHASE TAR tar FAILED -- pipeline stopping"
    exit 1
fi
N_ENTRIES=$(tar tf "$TAR_PATH" | wc -l)
TAR_BYTES=$(stat --format=%s "$TAR_PATH")
echo "=== $(date -u +%FT%TZ) PHASE TAR COMPLETE: $TAR_PATH ($TAR_BYTES bytes, $N_ENTRIES entries, tar tf verified) ==="

echo "=== $(date -u +%FT%TZ) PHASE S3: aws s3 cp $TAR_PATH $S3_KEY ==="
aws s3 cp "$TAR_PATH" "$S3_KEY"
if [ $? -ne 0 ]; then
    echo "$(date -u +%FT%TZ) PHASE S3 aws s3 cp FAILED -- pipeline stopping"
    exit 1
fi
S3_BYTES=$(aws s3 ls "$S3_KEY" | awk '{print $3}')
if [ "$S3_BYTES" != "$TAR_BYTES" ]; then
    echo "=== $(date -u +%FT%TZ) PHASE S3 BYTE MISMATCH: local=$TAR_BYTES s3=$S3_BYTES -- INVESTIGATE ==="
    exit 1
fi
echo "=== $(date -u +%FT%TZ) PHASE S3 COMPLETE: verified $S3_BYTES bytes match local file ==="

echo "=== $(date -u +%FT%TZ) PIPELINE COMPLETE ==="
