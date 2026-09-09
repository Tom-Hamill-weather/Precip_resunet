#!/bin/bash
# run_noterrain_experiment.sh
#
# End-to-end driver for the NO-TERRAIN ablation test used to add the black
# "Attention ResUNet w/o terrain" dots at the 24-h lead time to the
# right-hand (Western US) panels of Fig. 11 (Brier Skill Score vs. lead time).
#
# Steps:
#   1. Train 4 monthly no-terrain models (IC 2025030100/060100/090100/120100),
#      lead 24 h.
#   2. Run no-terrain full-domain inference over Mar/Jun/Sep/Dec 2025, lead 24 h.
#   3. Compute no-terrain reliability + BSS at lead 24 h.
#   4. Re-make the BSS-vs-lead-time figure with the black 24-h dots.
#
# usage: ./run_noterrain_experiment.sh

set -e

echo "### STEP 1/4: train no-terrain models (lead 24h) ###"
./control_train_resunet_gamma_mixture_noterrain.sh

echo "### STEP 2/4: no-terrain inference, Mar/Jun/Sep/Dec, lead 24h ###"
python control_resunet_inference_gamma_mixture_noterrain.py

echo "### STEP 3/4: no-terrain reliability + BSS at lead 24h ###"
python reliability_resunet_mixture_noterrain.py 24

echo "### STEP 4/4: re-make BSS-vs-lead figure with black no-terrain dots ###"
python plot_BSS_leadtime.py

echo "Done. See BSS_leadtime_q0.5_*.png"
