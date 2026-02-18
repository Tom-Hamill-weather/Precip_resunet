"""
python control_plot_graf_mrms_samples.py

hard-coded here to a specific file; the intent is to generate lots
of png files that show what the GRAF forecast, terrain devation, 
and MRMS data look like, so that we can visually verify to see if
the saved data look realistic and if they tend to emphasize heavier
precipitation.

Tom Hamill, Dec 2025
"""

import os, sys
samples = list(range(0,5000,100))
for isample in samples:
    cmd = 'python plot_graf_mrms_gfs_samples.py '+\
        '../resnet_data/GRAF_Unet_data_train_2025120100_12h.cPick '+str(isample)
    istat = os.system(cmd)
