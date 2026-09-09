"""
python save_hourly_graf_mrms_cdf_data.py clead [mmdd_begin] [mmdd_end]

e.g.,

python save_hourly_graf_mrms_cdf_data.py 24
python save_hourly_graf_mrms_cdf_data.py 24 0401 0731

Diagnostic for the 2026 reliability issue: accumulates histogram counts of
hourly GRAF forecast precip and contemporaneous MRMS observed precip, at a
fixed lead time, over many 00/06/12/18Z initial conditions in a given
month-day window (default Jan01-Jun30) of 2025 and of 2026.  Only pixels
with MRMS data_quality > 0.5 are included.  Saves histogram counts (not
the plot) so plotting can be iterated on separately via
plot_hourly_graf_mrms_cdf_comparison.py.
"""

import os, sys
import numpy as np
import pygrib
import _pickle as cPickle
from netCDF4 import Dataset
from dateutils import dateshift, daterange
from configparser import ConfigParser

# ---------------------------------------------------------------------

def read_config_file(config_file):
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object["DIRECTORIES"]
    return directory["GRAFdatadir_conus_new"], directory["mrms_data_directory"]

# ---------------------------------------------------------------------

def graf_filename(GRAFdatadir, cinit, clead):
    ilead = int(clead)
    cyyyymmdd = cinit[0:8]
    chh = cinit[8:10]
    cvalid = dateshift(cinit, ilead)
    cyyyymmdd_valid = cvalid[0:8]
    chh_valid = cvalid[8:10]
    input_dir = os.path.join(GRAFdatadir, cyyyymmdd, chh)
    filename = (f"grid.hdo-graf_conus.{cyyyymmdd_valid}T{chh_valid}0000Z."
                f"{cyyyymmdd}T{chh}0000Z.PT{ilead}H.CONUS@4km.APCP.SFC.grb2")
    return os.path.join(input_dir, filename)

# ---------------------------------------------------------------------

def read_graf_precip(grib_path, ilead):
    if not os.path.exists(grib_path):
        return None
    try:
        with pygrib.open(grib_path) as grb_file:
            grb_msgs = grb_file.select(endStep=ilead)
            if not grb_msgs:
                return None
            return grb_msgs[0].values
    except (IOError, ValueError, RuntimeError) as e:
        print(f'  ERROR reading {grib_path}: {e}')
        return None

# ---------------------------------------------------------------------

def read_mrms(mrms_dir, cvalid):
    cyyyymm = cvalid[0:6]
    filename = f'MRMS_1h_pamt_and_data_qual_{cvalid}.nc'
    filepath = os.path.join(mrms_dir, cyyyymm, filename)
    if not os.path.exists(filepath):
        return None, None
    try:
        with Dataset(filepath, 'r') as nc:
            quality = nc.variables['data_quality'][:, :]
            precip = nc.variables['precipitation'][:, :]
        # --- these are netCDF4 MaskedArrays; indexing/histogramming a
        #     MaskedArray with a MaskedArray boolean mask silently falls
        #     back to the raw underlying buffers and ignores which
        #     entries were flagged invalid, corrupting counts.  Fill to
        #     plain ndarrays with sentinel values that will never pass
        #     the quality > 0.5 test.
        quality = np.asarray(quality.filled(-1.0))
        precip = np.asarray(precip.filled(-1.0))
        return precip, quality
    except Exception as e:
        print(f'  ERROR reading MRMS {filepath}: {e}')
        return None, None

# ---------------------------------------------------------------------

def accumulate_histograms(cinit_list, clead, GRAFdatadir, mrms_dir, bin_edges):

    """ Loop over init times, read GRAF + MRMS, mask by quality > 0.5,
        and accumulate histogram counts for both. """

    ilead = int(clead)
    counts_graf = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    counts_mrms = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    nused = 0

    for cinit in cinit_list:
        cvalid = dateshift(cinit, ilead)

        graf_path = graf_filename(GRAFdatadir, cinit, clead)
        precip_graf = read_graf_precip(graf_path, ilead)
        if precip_graf is None:
            continue

        precip_mrms, quality = read_mrms(mrms_dir, cvalid)
        if precip_mrms is None:
            continue

        mask = quality > 0.5
        if not np.any(mask):
            continue

        counts_graf += np.histogram(precip_graf[mask], bins=bin_edges)[0]
        counts_mrms += np.histogram(precip_mrms[mask], bins=bin_edges)[0]
        nused += 1

    print(f'  used {nused} of {len(cinit_list)} init times for lead {clead}h')
    return counts_graf, counts_mrms, nused

# =======================================================================

def main():

    clead = sys.argv[1]
    mmdd_begin = sys.argv[2] if len(sys.argv) > 2 else '0101'
    mmdd_end = sys.argv[3] if len(sys.argv) > 3 else '0630'

    config_file = 'config_aws.ini'
    GRAFdatadir, mrms_dir = read_config_file(config_file)

    bin_edges = np.concatenate([np.arange(0.0, 25.01, 0.25), [np.inf]])

    windows = {
        '2025': (f'2025{mmdd_begin}00', f'2025{mmdd_end}18'),
        '2026': (f'2026{mmdd_begin}00', f'2026{mmdd_end}18'),
    }

    results = {'bin_edges': bin_edges}
    for year, (cbegin, cend) in windows.items():
        print(f'Processing {year}: {cbegin} to {cend}, lead {clead}h')
        all_inits = daterange(cbegin, cend, 6)
        all_inits = [c for c in all_inits if c[8:10] in ('00', '06', '12', '18')]
        counts_graf, counts_mrms, nused = accumulate_histograms(
            all_inits, clead, GRAFdatadir, mrms_dir, bin_edges)
        results[year] = {
            'counts_graf': counts_graf,
            'counts_mrms': counts_mrms,
            'nused': nused,
            'ntotal': len(all_inits),
        }

    outfile = f'CDF_GRAF_vs_MRMS_data_lead{clead}h_{mmdd_begin}-{mmdd_end}.cPick'
    with open(outfile, 'wb') as f:
        cPickle.dump(results, f)
    print(f'Saved {outfile}')

if __name__ == '__main__':
    main()
