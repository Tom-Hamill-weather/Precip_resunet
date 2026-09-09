#!/bin/bash
# Wait for interpolate_stage4_to_graf.py's bugfix regen to finish, then
# bundle the corrected GRAF-grid BSS reference climatology into a tarball
# and upload to s3://twc-nvidia for use on another AWS instance (Tom
# 2026-09-09: need GRAF climatology available elsewhere for Brier Skill
# Score computations).
#
# Mirrors HRRRcal's tar_upload_bss_climatology.sh convention (tar cf, no
# gzip -- .nc already zlib-compressed, verify with tar tf, upload via
# aws s3 cp, verify via aws s3 ls byte count) but uploads under
# s3://twc-nvidia/resnet/bss_climatology/ -- a separate key from
# HRRRcal's own s3://twc-nvidia/hrrrcal/bss_climatology/, since this repo's
# stage4_climo_on_graf.nc is an independently-generated file (this repo's
# own interpolate_stage4_to_graf.py, not HRRRcal's stage4_climo_to_graf.py).
set -uo pipefail

cd /home/thamill/resnet
SELF_LOG=/data/resnet_data/tar_upload_bss_climatology.log
exec >>"$SELF_LOG" 2>&1

LOCK_FILE=/tmp/resnet_tar_upload_bss_climatology.lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "=== $(date -u +%FT%TZ) another instance already holds the lock — exiting ==="
    exit 0
fi

REGEN_LOG=/data/resnet_data/interpolate_stage4_to_graf_refix.log
GRAF_CLIMO=/data/resnet_data/stage4_climo_on_graf.nc
TAR_PATH=/data/resnet_data/resnet_bss_climatology.tar
S3_KEY=s3://twc-nvidia/resnet/bss_climatology/resnet_bss_climatology.tar

echo "=== $(date -u +%FT%TZ) tar+upload watcher started, waiting on $REGEN_LOG ==="

while true; do
    if [ -f "$REGEN_LOG" ] && grep -q '^Done\.' "$REGEN_LOG"; then
        echo "=== $(date -u +%FT%TZ) climo regen complete, proceeding ==="
        break
    fi
    if [ -f "$REGEN_LOG" ] && grep -qE 'Traceback|SystemExit|ERROR' "$REGEN_LOG"; then
        echo "=== $(date -u +%FT%TZ) climo regen FAILED — not tarring, exiting ==="
        exit 1
    fi
    sleep 30
done

if [ ! -f "$GRAF_CLIMO" ]; then
    echo "=== $(date -u +%FT%TZ) missing input file: $GRAF_CLIMO — exiting ==="
    exit 1
fi

# ── tar (no gzip, matches HRRRcal convention) ──────────────────────────────

echo "=== $(date -u +%FT%TZ) PHASE T: tar cf $TAR_PATH ==="
tar cf "$TAR_PATH" -C /data/resnet_data stage4_climo_on_graf.nc
if [ $? -ne 0 ]; then
    echo "$(date -u +%FT%TZ) PHASE T tar FAILED — pipeline stopping"
    exit 1
fi
N_ENTRIES=$(tar tf "$TAR_PATH" | wc -l)
if [ $? -ne 0 ]; then
    echo "$(date -u +%FT%TZ) PHASE T tar tf verification FAILED — pipeline stopping"
    exit 1
fi
TAR_BYTES=$(stat --format=%s "$TAR_PATH")
echo "=== $(date -u +%FT%TZ) PHASE T COMPLETE: $TAR_PATH ($TAR_BYTES bytes, $N_ENTRIES entries, tar tf verified) ==="

# ── upload to s3://twc-nvidia ───────────────────────────────────────────────

echo "=== $(date -u +%FT%TZ) PHASE S3: aws s3 cp $TAR_PATH $S3_KEY ==="
aws s3 cp "$TAR_PATH" "$S3_KEY"
if [ $? -ne 0 ]; then
    echo "$(date -u +%FT%TZ) PHASE S3 aws s3 cp FAILED — pipeline stopping"
    exit 1
fi

S3_BYTES=$(aws s3 ls "$S3_KEY" | awk '{print $3}')
if [ "$S3_BYTES" != "$TAR_BYTES" ]; then
    echo "=== $(date -u +%FT%TZ) PHASE S3 BYTE MISMATCH: local=$TAR_BYTES s3=$S3_BYTES — INVESTIGATE ==="
    exit 1
fi
echo "=== $(date -u +%FT%TZ) PHASE S3 COMPLETE: verified $S3_BYTES bytes match local file, key=$S3_KEY ==="

echo "=== $(date -u +%FT%TZ) TAR+UPLOAD BSS CLIMATOLOGY PIPELINE COMPLETE ==="
