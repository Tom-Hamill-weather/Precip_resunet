"""
python control_resunet_inference_gamma.py cyyyymmddhh_begin cyyyymmddhh_end 
"""
import resunet_inference_gamma
from dateutils import daterange
import os, sys

cyyyymmddhh_begin = sys.argv[1]
cyyyymmddhh_end = sys.argv[2]

date_list = daterange(cyyyymmddhh_begin, cyyyymmddhh_end, 12)
for idate, date in enumerate(date_list):
    cmd = 'python resunet_inference_gamma.py '+date+' 6'
    print (cmd)
    istat = os.system(cmd)
    cmd = 'python resunet_inference_gamma.py '+date+' 12'
    istat = os.system(cmd)
    cmd = 'python resunet_inference_gamma.py '+date+' 24'
    istat = os.system(cmd)
    cmd = 'python resunet_inference_gamma.py '+date+' 36'
    istat = os.system(cmd)
    cmd = 'python resunet_inference_gamma.py '+date+' 48'
    istat = os.system(cmd)
    
