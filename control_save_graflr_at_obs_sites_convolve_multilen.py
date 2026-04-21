"""
python control_save_graflr_at_obs_sites_convolve_multilen.py cyyyymmddhh_begin cyyyymmddhh_end clead
"""

import numpy as np 
from dateutils import daterange
import os, sys

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]
clead = sys.argv[3]
cyyyymmddhh_list = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 6)

ilead = int(clead)
for cyyyymmddhh in cyyyymmddhh_list:
    cmd = 'python3 save_graflr_at_obs_sites_convolve_multilen.py '+\
        cyyyymmddhh + ' ' + clead
    print (cmd)
    istat = os.system(cmd)

