"""
python control_save_graf_convolve_multilen.py cyyyymmddhh_begin cyyyymmddhh_end clead
"""

import numpy as np 
from dateutils import daterange
import os, sys

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]
#clead = sys.argv[3]
cyyyymmddhh_list = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 48)

for cyyyymmddhh in cyyyymmddhh_list:
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 6'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 12'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 18'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 24'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 30'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 36'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 42'
    istat = os.system(cmd)
    cmd = 'python3 save_graf_convolve_multilen.py '+\
        cyyyymmddhh + ' 48'
    print (cmd)
    istat = os.system(cmd)

