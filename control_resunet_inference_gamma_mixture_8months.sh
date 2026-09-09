#!/bin/bash
# Generate full 1-48h lead gamma-mixture probability forecasts, 00Z/12Z
# cycles only, every DAY_STRIDE-th day, for the 8 months not currently in
# the probs/ archive (Mar/Jun/Sep/Dec 2025 already exist).  Used to build
# out more independent full-domain test data at reduced cost.
#
# Do NOT launch this while another training/inference job is using the
# GPU -- check `nvidia-smi` first.  N_WORKERS controls how many
# concurrent inference calls share the GPU; raise it once the GPU is
# otherwise idle.

N_WORKERS=8
DAY_STRIDE=3

python control_resunet_inference_gamma_mixture.py 2025010100 2025013100 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025020100 2025022800 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025040100 2025043000 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025050100 2025053100 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025070100 2025073100 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025080100 2025083100 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025100100 2025103100 $N_WORKERS $DAY_STRIDE
python control_resunet_inference_gamma_mixture.py 2025110100 2025113000 $N_WORKERS $DAY_STRIDE
