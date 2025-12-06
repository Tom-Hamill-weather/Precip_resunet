"""
python control_plot_graf_mrms_samples.py
"""

import os, sys
samples = list(range(0,5000,100))
for isample in samples:
    cmd = 'python plot_graf_mrms_samples.py '+\
        '../resnet_data/GRAF_Unet_data_train_2025112100_12h.cPick '+str(isample)
    istat = os.system(cmd)