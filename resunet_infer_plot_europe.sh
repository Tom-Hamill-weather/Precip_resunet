#!/bin/bash
# usage: ./resunet_infer_plot_europe.sh YYYYMMDDHH [START_LEAD]
# Runs fulldomain inference and plots for European domain, forecast hours 1-48.
# Optionally restart from a specific lead time:
#   ./resunet_infer_plot_europe.sh 2025120412 12

if [ -z "$1" ]; then
    echo "Usage: $0 YYYYMMDDHH [START_LEAD]"
    exit 1
fi

CYYYYMMDDHH=$1
START_LEAD=${2:-1}

for LEAD in $(seq 1 48); do
    if [ "$LEAD" -lt "$START_LEAD" ]; then
        echo "Skipping lead ${LEAD}h"
        continue
    fi
    echo "--- Lead ${LEAD}h ---"
    #python resunet_inference_gamma_mixture_fulldomain_europe.py ${CYYYYMMDDHH} ${LEAD}
    #if [ $? -ne 0 ]; then
    #    echo "ERROR: inference failed for lead ${LEAD}h, skipping plot."
    #    continue
    #fi
    python make_plots_gamma_mixture2_europe.py ${CYYYYMMDDHH} ${LEAD}
    python make_plots_gamma_mixture2_3panel_europe.py ${CYYYYMMDDHH} ${LEAD}
done
