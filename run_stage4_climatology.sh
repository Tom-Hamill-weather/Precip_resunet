#!/bin/bash
# run_stage4_climatology.sh
# Wrapper for compute_stage4_climatology.py.
# Sends an email via AWS SES if the script exits with a non-zero status.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/compute_stage4_climatology.py"
LOG_FILE="/data/resnet_data/stage4_climo.log"
HOST="$(hostname -s)"
FROM="tom.hamill@weather.com"
TO="tom.hamill@weather.com"
REGION="us-east-1"

notify_failure() {
    local exit_code=$1
    local last_lines
    last_lines=$(tail -20 "$LOG_FILE" 2>/dev/null || echo "(log not found)")

    aws ses send-email \
        --region "$REGION" \
        --from "$FROM" \
        --to "$TO" \
        --subject "CRASH: compute_stage4_climatology on $HOST (exit $exit_code)" \
        --text "compute_stage4_climatology.py crashed on $HOST.

Exit code : $exit_code
Time      : $(date)
Log       : $LOG_FILE

--- last 20 log lines ---
$last_lines
" 2>&1 | tee -a "$LOG_FILE"
}

cd "$SCRIPT_DIR"
python compute_stage4_climatology.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    notify_failure "$EXIT_CODE"
    exit $EXIT_CODE
fi
