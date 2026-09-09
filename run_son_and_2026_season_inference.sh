#!/bin/bash
# run_son_and_2026_season_inference.sh
#
# Tom asked (2026-09-09): finish the SON season+FiLM retrain, then run
# season-model inference for 2026010100 through 2026083118, every 6-h
# cycle, every 6-h lead through 48h. Tar the resulting probability files
# and upload to s3://twc-nvidia.
#
# Launched via systemd-run --user so it survives session logout, per
# [[feedback_background_tasks]].
set -uo pipefail

cd /home/thamill/resnet
SELF_LOG=/data/resnet_data/run_son_and_2026_season_inference.log
exec >>"$SELF_LOG" 2>&1

LOCK_FILE=/tmp/resnet_son_and_2026_season_inference.lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "=== $(date -u +%FT%TZ) another instance already holds the lock -- exiting ==="
    exit 0
fi

echo "=== $(date -u +%FT%TZ) PIPELINE START ==="

# ---------------------------------------------------------------------------
# PHASE G: finish the August 2026 GFS backfill (days 1-8 already done from
# the earlier, aborted attempt; full_build_range() skips existing files).
# CPU/network only -- runs in the background alongside SON training.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE G: finishing August 2026 GFS backfill ==="
/home/thamill/miniconda3/bin/python3 fetch_gfs_extension.py --full-range 20260801 20260831 --workers 8 \
    >> /data/resnet_data/gfs_aug_2026_backfill.log 2>&1 &
GFS_PID=$!
echo "PHASE G launched, PID $GFS_PID"

# ---------------------------------------------------------------------------
# PHASE SON: resume the SON season+FiLM retrain (patience=4). Resumes from
# the existing checkpoint (best val loss 0.2905 as of epoch 11) -- same
# command as the original launch, no --init-from needed.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE SON: resuming training ==="
SON_START=$(date +%s)
/home/thamill/miniconda3/bin/python3 pytorch_train_resunet_gamma_mixture_season.py --season SON --patience 4 \
    >> /data/resnet_data/season_training_logs/SON_patience4.log 2>&1
SON_RC=$?
SON_ELAPSED=$(( $(date +%s) - SON_START ))
if [ $SON_RC -ne 0 ]; then
    echo "=== $(date -u +%FT%TZ) PHASE SON FAILED (rc=$SON_RC, ${SON_ELAPSED}s) -- see season_training_logs/SON_patience4.log -- pipeline stopping ==="
    exit 1
fi
echo "=== $(date -u +%FT%TZ) PHASE SON COMPLETE (${SON_ELAPSED}s) ==="

# ---------------------------------------------------------------------------
# PHASE I: season-model inference, 2026-01-01 through 2026-08-31, every 6h
# cycle, leads 6-48h. Waits on the GFS backfill first (should be long done).
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) waiting on GFS backfill (PID $GFS_PID) ==="
wait "$GFS_PID"
echo "=== $(date -u +%FT%TZ) GFS backfill finished, rc=$? ==="

echo "=== $(date -u +%FT%TZ) PHASE I: season-model inference, Jan-Aug 2026 ==="
I_START=$(date +%s)
/home/thamill/miniconda3/bin/python3 control_resunet_inference_2026_season_jan_aug.py 8 \
    > /data/resnet_data/log_inference_season_jan_aug.txt 2>&1
I_RC=$?
I_ELAPSED=$(( $(date +%s) - I_START ))
if [ $I_RC -ne 0 ]; then
    echo "=== $(date -u +%FT%TZ) PHASE I FAILED (rc=$I_RC, ${I_ELAPSED}s) -- see log_inference_season_jan_aug.txt -- NOT tarring ==="
    exit 1
fi
echo "=== $(date -u +%FT%TZ) PHASE I COMPLETE (${I_ELAPSED}s) ==="

# ---------------------------------------------------------------------------
# PHASE TAR: tar all season-model probs files for 2026-01-01 through
# 2026-08-31 (the full requested range, not just the newly-added ones) and
# upload to s3://twc-nvidia.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) PHASE TAR: building file list ==="
FILELIST=/data/resnet_data/season_jan_aug_2026_filelist.txt
/home/thamill/miniconda3/bin/python3 -c "
from dateutils import daterange
leads = [6, 12, 18, 24, 30, 36, 42, 48]
dates = daterange('2026010100', '2026083118', 6)
with open('$FILELIST', 'w') as f:
    for d in dates:
        for l in leads:
            f.write(f'{d}_{l}_probs_gamma_mixture_season.nc\n')
"
FILELIST_EXIST=/data/resnet_data/season_jan_aug_2026_filelist_exist.txt
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

TAR_PATH=/data/resnet_data/resnet_probs_2026_season_jan_aug.tar
S3_KEY=s3://twc-nvidia/resnet/probs_2026/resnet_probs_2026_season_jan_aug.tar

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
