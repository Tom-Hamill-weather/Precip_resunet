#!/bin/bash
# control_save_patched_GRAF_MRMS_GFS2.sh
#
# Runs save_patched_GRAF_MRMS_GFS2.py for 4 seasons × 16 lead times in parallel.
# NJOBS controls the sliding-window concurrency limit.  Each job peaks at roughly
# 3-4 GB RAM; with 26 GB available on the G5 instance, 4 concurrent jobs is safe.
# Raise to 6 if jobs turn out to be lighter than expected.

NJOBS=4

DATES="2025120100 2025090100 2025060100 2025030100"
LEADS="3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48"

TOTAL_JOBS=$(echo $DATES | wc -w)
TOTAL_JOBS=$(( TOTAL_JOBS * $(echo $LEADS | wc -w) ))
DONE_COUNT=0

# Throttle: block until fewer than NJOBS background jobs are running
throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do
        sleep 5
    done
}

# Background ticker: prints a summary line every 60 seconds
ticker() {
    local total=$1
    while true; do
        sleep 60
        local n_running
        n_running=$(jobs -rp | wc -l)
        local n_done
        n_done=$(grep -rl "Final patch counts" log_patches_*.txt 2>/dev/null | wc -l)
        local n_failed
        n_failed=$(grep -rl "Traceback\|Error\|FAILED" log_patches_*.txt 2>/dev/null | wc -l)
        echo "[$(date '+%H:%M:%S')] Progress: ${n_done}/${total} done, ${n_running} running, ${n_failed} failed"
    done
}

ticker "$TOTAL_JOBS" &
TICKER_PID=$!

START_TIME=$(date +%s)
echo "[$(date '+%H:%M:%S')] Launching $TOTAL_JOBS jobs (max $NJOBS concurrent)"

for date in $DATES; do
    for lead in $LEADS; do
        throttle
        logfile="log_patches_${date}_${lead}h.txt"
        echo "[$(date '+%H:%M:%S')] Started  $date  lead=${lead}h"
        (
            python save_patched_GRAF_MRMS_GFS2.py "$date" "$lead" > "$logfile" 2>&1
            rc=$?
            if [ $rc -eq 0 ]; then
                counts=$(grep "Final patch counts" "$logfile" | tail -1 | sed 's/.*INFO: //')
                echo "[$(date '+%H:%M:%S')] Done     $date  lead=${lead}h  — $counts"
            else
                echo "[$(date '+%H:%M:%S')] FAILED   $date  lead=${lead}h  (rc=$rc) — see $logfile"
            fi
        ) &
    done
done

wait
kill "$TICKER_PID" 2>/dev/null

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "[$(date '+%H:%M:%S')] All $TOTAL_JOBS jobs complete in $(( ELAPSED/60 ))m $(( ELAPSED%60 ))s."
