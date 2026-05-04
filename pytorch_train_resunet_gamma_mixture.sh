#!/bin/bash
# ./pytorch_train_resunet_gamma_mixture.sh [START_DATE [START_LEAD]]
# Simple script to train 2-component Gamma mixture model over multiple lead times.
# The pytorch_train_resunet_gamma_mixture.py is smart enough to
# begin its training with the weights of the training 3 h previous
# rather than starting from random.
# Tom Hamill with Claude Code assistance, Feb 2026
#
# Restart from a specific point:
#   ./pytorch_train_resunet_gamma_mixture.sh 2025060100 9
# Runs all lead times for that date starting at 9h, then continues normally.

START_DATE=${1:-""}
START_LEAD=${2:-0}

_skip=true
[ -z "$START_DATE" ] && _skip=false

run() {
    local date=$1 lead=$2
    if $_skip; then
        if [ "$date" = "$START_DATE" ] && [ "$lead" -eq "$START_LEAD" ]; then
            _skip=false
        else
            echo "Skipping $date ${lead}h"
            return
        fi
    fi
    python pytorch_train_resunet_gamma_mixture.py "$date" "$lead"
}

#run 2025030100 3
run 2025030100 6
run 2025030100 9
run 2025030100 12
run 2025030100 15
run 2025030100 18
run 2025030100 21
run 2025030100 24

run 2025030100 27
run 2025030100 30
run 2025030100 33
run 2025030100 36
run 2025030100 39
run 2025030100 42
run 2025030100 45
run 2025030100 48

run 2025060100 3
run 2025060100 6
run 2025060100 9
run 2025060100 12
run 2025060100 15
run 2025060100 18
run 2025060100 21
run 2025060100 24

run 2025060100 27
run 2025060100 30
run 2025060100 33
run 2025060100 36
run 2025060100 39
run 2025060100 42
run 2025060100 45
run 2025060100 48

run 2025090100 3
run 2025090100 6
run 2025090100 9
run 2025090100 12
run 2025090100 15
run 2025090100 18
run 2025090100 21
run 2025090100 24

run 2025090100 27
run 2025090100 30
run 2025090100 33
run 2025090100 36
run 2025090100 39
run 2025090100 42
run 2025090100 45
run 2025090100 48

run 2025120100 3
run 2025120100 6
run 2025120100 9
run 2025120100 12
run 2025120100 15
run 2025120100 18
run 2025120100 21
run 2025120100 24

run 2025120100 27
run 2025120100 30
run 2025120100 33
run 2025120100 36
run 2025120100 39
run 2025120100 42
run 2025120100 45
run 2025120100 48
