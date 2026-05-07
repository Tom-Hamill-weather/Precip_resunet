"""
python control_resunet_inference_gamma_mixture_fulldomain.py cyyyymmddhh_begin cyyyymmddhh_end
"""
import resunet_inference_gamma_mixture_optimized
from dateutils import daterange, dateshift
import os, sys

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]

# Offset start by 6h to pick up 06Z and 18Z initialization cycles
cyyyymmddhh_begin_06 = dateshift(cyyyymmddhh_begin, 6)
date_list = daterange(cyyyymmddhh_begin_06, cyyyymmddhh_end, 12)
for idate, date in enumerate(date_list):
    for ilead in range (6, 49, 6):   # 1 to 48 previously
        clead = str(ilead)
        cmd = 'python resunet_inference_gamma_mixture_fulldomain.py '+date+' '+clead
        istat = os.system(cmd)

