#!/bin/bash
# control_train_resunet_gamma_mixture_noterrain.sh
#
# Train the NO-TERRAIN gamma-mixture ablation models.
#
# To mirror the monthly-retraining setup used for the full (with-terrain)
# results in the paper, we train one model per evaluation month at its
# matching IC date, lead = 24 h only (the no-terrain dots in Fig. 11 are
# only added at the 24-h lead time).
#
# Inference auto-selects the most recent training-date weights <= the forecast
# date, so:
#   2025030100 -> used for March evaluation
#   2025060100 -> used for June evaluation
#   2025090100 -> used for September evaluation
#   2025120100 -> used for December evaluation
#
# usage: ./control_train_resunet_gamma_mixture_noterrain.sh

set -e

LEAD=24
for IC in 2025030100 2025060100 2025090100 2025120100 ; do
    echo "=============================================================="
    echo " Training NO-TERRAIN model: IC=${IC}  lead=${LEAD}h"
    echo "=============================================================="
    python pytorch_train_resunet_gamma_mixture_noterrain.py ${IC} ${LEAD}
done

echo "All no-terrain training runs complete."
