"""
plot_case_day_baseline_vs_season.py cyyyymmddhh clead clat_center clon_center

Recreates the original manuscript's case-study figure format (Fig.
washington_ar: (a) raw GRAF 1-h precipitation, (b) Attention ResUNet
probability, (c) a third reference panel), but for the "Methodological
Changes" note's own purpose: showing how the ResUNet's own postprocessed
probability of exceeding 0.25 mm changes between the original training
(panel b) and the current, season-pooled + FiLM training (panel c),
for the same case day used in probability_read()'s cache.

Case day: 2026-03-12 00Z Washington-state atmospheric river event
(IC 2026031112, +12h), selected as the strongest box-mean 1-h MRMS
accumulation over the Olympic Peninsula/Cascades region across the
full 2026 Jan-Aug record (see scratch scan, box lat 46-48.7N,
lon 121-124.8W).

Usage:
    python plot_case_day_baseline_vs_season.py 2026031112 12 47.5 -122.5
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from configparser import ConfigParser
import pygrib
from dateutils import dateshift
from netCDF4 import Dataset

# ---- Helpers copied from reliability_resunet_mixture.py (that script is a
# top-level driver, not importable as a module without re-running its own
# CLI-argument-driven scoring loop).

def detect_environment():
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            return 'aws', path
    return 'laptop', None


def read_config_file(config_file, directory_object_name):
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]
    if "GRAFdatadir_conus" in directory:
        GRAFdatadir_conus = directory["GRAFdatadir_conus"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus"]
        GRAF_plot_dir = directory["GRAF_plot_dir"]
        mrms_data_directory = os.path.expanduser(directory["mrms_data_directory"])
    else:
        GRAFdatadir_conus = directory.get("GRAFdatadir_conus_new")
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"
        mrms_data_directory = f"{base_dir}/MRMS/"
    return GRAFdatadir_conus, GRAFprobsdir_conus, GRAF_plot_dir, mrms_data_directory


def read_gribdata(gribfilename, endStep):
    istat = -1
    if os.path.exists(gribfilename):
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep=endStep)[0]
            lats, lons = grb.latlons()
            precipitation = grb.values
            precipitation = np.where(precipitation > 75., 75., precipitation)
            lon_0 = grb.projparams["lon_0"]
            lat_0 = grb.projparams["lat_0"]
            lat_1 = grb.projparams["lat_1"]
            lat_2 = grb.projparams["lat_2"]
            istat = 0
            fcstfile.close()
        except Exception as e:
            print(f'   Error in read_gribdata reading {gribfilename}: {e}')
            istat = -1
    else:
        print('grib file does not exist.')
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0))
        lons = np.empty((0, 0))
        lon_0 = 0; lat_0 = 0; lat_1 = 0; lat_2 = 0
    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2


def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]
    prefix = ('grid.hdo-graf_conus.' if int(cyyyymmddhh) >= 2024040100
              else 'grid.hdo-graflr_conus.')
    input_directory = GRAFdatadir_conus + cyyyymmdd + '/' + chh + '/'
    input_file = (prefix + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.' +
                  cyyyymmdd + 'T' + chh + '0000Z.PT' + clead +
                  'H.CONUS@4km.APCP.SFC.grb2')
    infile = input_directory + input_file
    fexist1 = os.path.exists(infile)
    print(infile, fexist1)
    if fexist1:
        istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2 = \
            read_gribdata(infile, il)
    else:
        print('  could not find ', infile)
        istat = -1
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0), dtype=float)
        lons = np.empty((0, 0), dtype=float)
        lon_0 = lat_0 = lat_1 = lat_2 = -999.99
    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2


def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus,
                      probs_suffix='_probs_gamma_mixture.nc'):
    infile = GRAFprobsdir_conus + cyyyymmddhh + '_' + clead + probs_suffix
    if not os.path.exists(infile):
        return -1, None, np.empty((0, 0)), np.empty((0, 0))
    nc = Dataset(infile, 'r')
    lat = nc.variables['lat'][:, :]
    lon = nc.variables['lon'][:, :]
    probs = {
        0.25: {'raw': nc.variables['raw_p0p25mm_prob'][:, :],
               'gamma': nc.variables['gamma_p0p25mm_prob'][:, :]},
    }
    nc.close()
    return 0, probs, lat, lon


cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]
clat_center = float(sys.argv[3])
clon_center = float(sys.argv[4])

ENVIRONMENT, AWS_BASE_PATH = detect_environment()
config_file_name = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
GRAFdatadir_conus, GRAFprobsdir_conus, GRAF_plot_dir, mrms_data_directory = \
    read_config_file(config_file_name, 'DIRECTORIES')

# ---- Raw GRAF 1-h precipitation forecast, panel (a)

istat_graf, precip_graf, lats, lons, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus)
if istat_graf != 0:
    print(f'ERROR: could not read raw GRAF precip for {cyyyymmddhh} +{clead}h')
    sys.exit(1)

# ---- ResUNet postprocessed probability of exceeding 0.25 mm,
#      original training (panel b) vs. current training (panel c)

istat_b, probs_baseline, lat_b, lon_b = probability_read(
    clead, cyyyymmddhh, GRAFprobsdir_conus, probs_suffix='_probs_gamma_mixture.nc')
istat_s, probs_season, lat_s, lon_s = probability_read(
    clead, cyyyymmddhh, GRAFprobsdir_conus, probs_suffix='_probs_gamma_mixture_season.nc')
if istat_b != 0 or istat_s != 0:
    print(f'ERROR: could not read baseline/season probability files for '
          f'{cyyyymmddhh} +{clead}h')
    sys.exit(1)

prob_baseline = probs_baseline[0.25]['gamma']
prob_season = probs_season[0.25]['gamma']

# ===========================================================
# Draw the three-panel figure, same Basemap/zoom conventions as
# plot_GRAF_MRMS_zoom.py (method.tex's own case-study figures).
# ===========================================================

latb = clat_center - 6.0
late = clat_center + 6.0
lonb = clon_center - 9.0
lone = clon_center + 9.0

m = Basemap(rsphere=(6378137.00, 6356752.3142),
            resolution='l', area_thresh=1000., projection='lcc',
            lat_1=35., lat_2=45, lat_0=clat_center, lon_0=clon_center,
            llcrnrlon=lonb, llcrnrlat=latb, urcrnrlon=lone, urcrnrlat=late)
x, y = m(lons, lats)

colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
                   '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
                   '#FCD5D9', '#F6A3AE', '#f17484']
clevs_precip = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 50])

colorst_prob = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
                 '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
                 '#FCD5D9', '#F6A3AE', '#f17484']
clevs_prob = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])

fig = plt.figure(figsize=(13.0, 4.3))

panels = [
    ('(a) Raw GRAF 1-h precipitation forecast', precip_graf, colorst_precip,
     clevs_precip, '1-h accumulated precipitation (mm)'),
    ('(b) Attention ResUNet probability, original training', 100. * prob_baseline,
     colorst_prob, clevs_prob, 'Probability of exceeding 0.25 mm (%)'),
    ('(c) Attention ResUNet probability, current training', 100. * prob_season,
     colorst_prob, clevs_prob, 'Probability of exceeding 0.25 mm (%)'),
]

for ipanel, (title, data_to_plot, colorst, clevs, clabel) in enumerate(panels):
    axloc = [0.01 + ipanel * 0.335, 0.13, 0.31, 0.74]
    caxloc = [0.02 + ipanel * 0.335, 0.08, 0.29, 0.02]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "", colorst, N=len(colorst))
    norm = colors.BoundaryNorm(boundaries=clevs, ncolors=len(colorst), clip=True)

    ax = fig.add_axes(axloc)
    ax.set_title(title, fontsize=10.5, color='Black')
    CS = m.pcolormesh(x, y, data_to_plot, cmap=cmap, shading='nearest',
                       norm=norm, ax=ax)
    m.drawcoastlines(linewidth=0.8, color='Gray', ax=ax)
    m.drawcountries(linewidth=0.6, color='Gray', ax=ax)
    m.drawstates(linewidth=0.3, color='Gray', ax=ax)

    cax = fig.add_axes(caxloc)
    cb = plt.colorbar(CS, orientation='horizontal', cax=cax,
                       drawedges=True, ticks=clevs, format='%g', extend='max')
    cb.ax.tick_params(labelsize=6)
    cb.set_label(clabel, fontsize=7)

outfile = f'case_day_baseline_vs_season_IC{cyyyymmddhh}_{clead}h.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {outfile}')
