#!/bin/bash
# usage: ./resunet_infer_plot_europe_season.sh YYYYMMDDHH [START_LEAD] [END_LEAD]
# Runs season+FiLM+precip-climo fulldomain inference and plots for the
# European domain (resunet_inference_gamma_mixture_season_europe.py), for
# forecast hours START_LEAD-END_LEAD.

if [ -z "$1" ]; then
    echo "Usage: $0 YYYYMMDDHH [START_LEAD] [END_LEAD]"
    exit 1
fi

CYYYYMMDDHH=$1
START_LEAD=${2:-1}
END_LEAD=${3:-48}

for LEAD in $(seq "$START_LEAD" "$END_LEAD"); do
    echo "--- Lead ${LEAD}h ---"
    python resunet_inference_gamma_mixture_season_europe.py "${CYYYYMMDDHH}" "${LEAD}"
    if [ $? -ne 0 ]; then
        echo "ERROR: inference failed for lead ${LEAD}h, skipping plot."
        continue
    fi
    python make_plots_gamma_mixture2_season_europe.py "${CYYYYMMDDHH}" "${LEAD}"
    python make_plots_gamma_mixture2_3panel_season_europe.py "${CYYYYMMDDHH}" "${LEAD}"
done
