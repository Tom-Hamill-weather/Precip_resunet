#!/bin/bash
# control_pytorch_train_resunet_gamma_mixture.sh
#
# Trains the gamma-mixture ResUNet for 4 seasons × 16 lead times (3–48 h).
# Training is intentionally serial: the single L4 GPU is the bottleneck, and
# running multiple training jobs simultaneously would split GPU compute without
# any net throughput gain.  Each job runs to completion before the next starts.
#
# Leads are run 3→6→...→48 within each date so that checkpoints accumulate
# progressively (each lead can warm-start from its own prior run if re-run).

DATES="2025120100 2025090100 2025060100 2025030100"
LEADS="3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48"

TOTAL=$(echo $DATES | wc -w)
TOTAL=$(( TOTAL * $(echo $LEADS | wc -w) ))
COUNT=0
START_TIME=$(date +%s)

for date in $DATES; do
    for lead in $LEADS; do
        COUNT=$(( COUNT + 1 ))
        logfile="log_train_${date}_${lead}h.txt"
        echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL)  Training $date  lead=${lead}h"
        python pytorch_train_resunet_gamma_mixture.py "$date" "$lead" \
            > "$logfile" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            summary=$(grep "Best Val Loss\|best_val_loss\|Epoch.*Val" "$logfile" | tail -1)
            echo "[$(date '+%H:%M:%S')] Done     $date  lead=${lead}h  — $summary"
        else
            echo "[$(date '+%H:%M:%S')] FAILED   $date  lead=${lead}h  (rc=$rc) — see $logfile"
        fi
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "[$(date '+%H:%M:%S')] All $TOTAL jobs complete in $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m."
