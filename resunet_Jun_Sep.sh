#!/bin/bash
# usage: $./resunet_Jun_Sep.sh
# generates forecast inferences for chosen date/hour.

#sleep 2h
#sleep 50m

#python3 control_resunet_inference_gamma_mixture.py 2025060100 2025063018
#python3 control_resunet_inference_gamma_mixture.py 2025090100 2025093018

#python save_patched_GRAF_MRMS_GFS.py 2025120100 3
#python save_patched_GRAF_MRMS_GFS.py 2025120100 6
#python save_patched_GRAF_MRMS_GFS.py 2025120100 9
#python save_patched_GRAF_MRMS_GFS.py 2025120100 12
#python save_patched_GRAF_MRMS_GFS.py 2025120100 15
#python save_patched_GRAF_MRMS_GFS.py 2025120100 18
#python save_patched_GRAF_MRMS_GFS.py 2025120100 21
#python save_patched_GRAF_MRMS_GFS.py 2025120100 24
#python save_patched_GRAF_MRMS_GFS.py 2025120100 27
#python save_patched_GRAF_MRMS_GFS.py 2025120100 30
#python save_patched_GRAF_MRMS_GFS.py 2025120100 33
#python save_patched_GRAF_MRMS_GFS.py 2025120100 36
#python save_patched_GRAF_MRMS_GFS.py 2025120100 39
#python save_patched_GRAF_MRMS_GFS.py 2025120100 42
#python save_patched_GRAF_MRMS_GFS.py 2025120100 45
#python save_patched_GRAF_MRMS_GFS.py 2025120100 48

python pytorch_train_resunet_gamma_mixture.py 2025120100 3
python pytorch_train_resunet_gamma_mixture.py 2025120100 6
python pytorch_train_resunet_gamma_mixture.py 2025120100 9
python pytorch_train_resunet_gamma_mixture.py 2025120100 12
python pytorch_train_resunet_gamma_mixture.py 2025120100 15
python pytorch_train_resunet_gamma_mixture.py 2025120100 18
python pytorch_train_resunet_gamma_mixture.py 2025120100 21
python pytorch_train_resunet_gamma_mixture.py 2025120100 24
python pytorch_train_resunet_gamma_mixture.py 2025120100 27
python pytorch_train_resunet_gamma_mixture.py 2025120100 30
python pytorch_train_resunet_gamma_mixture.py 2025120100 33
python pytorch_train_resunet_gamma_mixture.py 2025120100 36
#python pytorch_train_resunet_gamma_mixture.py 2025120100 39
#python pytorch_train_resunet_gamma_mixture.py 2025120100 42
#python pytorch_train_resunet_gamma_mixture.py 2025120100 45
#python pytorch_train_resunet_gamma_mixture.py 2025120100 48

