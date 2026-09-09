"""
control_resunet_inference_gamma_mixture_noterrain.py

Drive NO-TERRAIN full-domain inference over the four evaluation months
(March, June, September, December 2025), lead = 24 h only.

These are exactly the initialization dates/hours that
reliability_resunet_mixture_noterrain.py evaluates (every 6 h within each
month). Each call writes:

    <date>_24_probs_gamma_mixture_noterrain.nc

usage:
    python control_resunet_inference_gamma_mixture_noterrain.py
"""
import os
from dateutils import daterange

LEAD = '24'

# Same four-month windows, every 6 h, as reliability_resunet_mixture.py
months = {
    'Mar': daterange('2025030100', '2025033118', 6),
    'Jun': daterange('2025060100', '2025063018', 6),
    'Sep': daterange('2025090100', '2025093018', 6),
    'Dec': daterange('2025120100', '2025123118', 6),
}

date_list = []
for m in ('Mar', 'Jun', 'Sep', 'Dec'):
    date_list.extend(months[m])

print(f"NO-TERRAIN inference: {len(date_list)} init dates, lead={LEAD}h")

for idate, date in enumerate(date_list):
    print(f"[{idate+1}/{len(date_list)}] {date}  lead={LEAD}h")
    cmd = ('python resunet_inference_gamma_mixture_noterrain_fulldomain.py '
           + date + ' ' + LEAD)
    istat = os.system(cmd)
    if istat != 0:
        print(f"   WARNING: nonzero exit ({istat}) for {date}")
