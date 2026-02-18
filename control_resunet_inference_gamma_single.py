"""
python control_resunet_inference_gamma_single.py cyyyymmddhh
"""
import resunet_inference_gamma
from dateutils import daterange
import os, sys

cyyyymmddhh = sys.argv[1]
for ilead in range(1,49):
    clead = str(ilead)
    
    cmd = 'python resunet_inference_gamma.py '+cyyyymmddhh+' '+clead
    print (cmd)
    istat = os.system(cmd)
    
    cmd = 'python make_plots_gamma.py '+cyyyymmddhh+' '+clead
    print (cmd)
    istat = os.system(cmd)
    
    
