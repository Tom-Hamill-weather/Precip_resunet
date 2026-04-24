"""
python control_resunet_inference_gamma_mixture.py cyyyymmddhh_begin cyyyymmddhh_end
"""
import resunet_inference_gamma_mixture_optimized
from dateutils import daterange
import os, sys

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]

date_list = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 12)
for idate, date in enumerate(date_list):
    for ilead in range (48, 49):   # 1 to 48 previously
        clead = str(ilead)
        cmd = 'python resunet_inference_gamma_mixture_optimized.py '+date+' '+clead
        istat = os.system(cmd)

