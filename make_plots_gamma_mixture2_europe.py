"""
python make_plots_gamma_mixture2_europe.py cyyyymmddhh clead
e.g.,
python make_plots_gamma_mixture2_europe.py 2025120812 12

Four-panel plot for the European domain:
  (a) GRAF hourly precipitation amount
  (b) GRAF terrain elevation
  (c) Gamma mixture model P(>0.25 mm/h)
  (d) Smoothed raw ensemble P(>0.25 mm/h)

Reads probability file produced by
resunet_inference_gamma_mixture_optimized_europe.py.
"""

from configparser import ConfigParser
import numpy as np
import os, sys
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from dateutils import dateshift
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# ---------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if 'GRAFdatadir_conus_laptop' in directory:
        GRAFdatadir_europe = directory.get(
            'GRAFdatadir_europe_laptop',
            directory['GRAFdatadir_conus_laptop'].replace('conus', 'europe'))
        GRAFprobsdir = directory['GRAFprobsdir_conus_laptop']
        GRAF_plot_dir = directory.get('GRAF_plot_dir',
                                      directory['GRAFprobsdir_conus_laptop'])
        terrain_file = directory.get(
            'terrain_file_europe',
            directory.get('terrain_file', None))
    else:
        GRAFdatadir_europe = directory.get(
            'GRAFdatadir_europe',
            '/data/resnet_data/GRAF/hdo-graf_europe/')
        base_dir = directory.get('resnet_data_directory',
                                 AWS_BASE_PATH or '/data/resnet_data')
        GRAFprobsdir  = f'{base_dir}/probs/'
        GRAF_plot_dir = f'{base_dir}/plots/'
        terrain_file  = f'{base_dir}/terrain/GRAF_Europe_terrain_info.nc'

    print(f'  GRAF Europe path:  {GRAFdatadir_europe}')
    print(f'  Probs path:        {GRAFprobsdir}')
    print(f'  Plot directory:    {GRAF_plot_dir}')
    print(f'  Terrain file:      {terrain_file}')
    return GRAFdatadir_europe, GRAFprobsdir, GRAF_plot_dir, terrain_file

# ---------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    import pygrib
    if not os.path.exists(gribfilename):
        print('grib file does not exist.')
        return -1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), 0,0,0,0
    try:
        f = pygrib.open(gribfilename)
        grb = f.select(endStep=endStep)[0]
        lats, lons = grb.latlons()
        precipitation = np.where(grb.values > 75., 75., grb.values)
        lon_0 = grb.projparams['lon_0']; lat_0 = grb.projparams['lat_0']
        lat_1 = grb.projparams['lat_1']; lat_2 = grb.projparams['lat_2']
        f.close()
        return 0, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2
    except Exception as e:
        print(f'  Error reading {gribfilename}: {e}')
        return -1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), 0,0,0,0

# ---------------------------------------------------------------

def GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe):
    il = int(clead)
    cyyyymmdd        = cyyyymmddhh[:8]
    chh              = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst   = cyyyymmddhh_fcst[:8]
    chh_fcst         = cyyyymmddhh_fcst[8:10]

    input_dir = GRAFdatadir_europe + cyyyymmdd + '/' + chh + '/'

    for prefix in ('grid.hdo-graf_europe.', 'grid.hdo-graflr_europe.'):
        fname = (prefix
                 + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.'
                 + cyyyymmdd + 'T' + chh + '0000Z.'
                 + 'PT' + clead + 'H.EUROPE@4km.APCP.SFC.grb2')
        infile = input_dir + fname
        if os.path.exists(infile):
            break
    else:
        print(f'  Could not find European GRAF file in {input_dir}')
        return (-1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)),
                0, 0, -99., -99., -999., -999., np.empty((0,0)),
                -999., -999., -999., -999.)

    print(infile, True)
    istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2 = \
        read_gribdata(infile, il)
    if istat != 0:
        return (-1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)),
                0, 0, -99., -99., -999., -999., np.empty((0,0)),
                -999., -999., -999., -999.)

    ny, nx = lats.shape
    tzoff = lons * 12 / 180.
    verif_local_time = int(chh_fcst) + tzoff
    return (0, precipitation, lats, lons, ny, nx,
            lats.min(), lats.max(), lons.min(), lons.max(),
            verif_local_time, lon_0, lat_0, lat_1, lat_2)

# ---------------------------------------------------------------

def probability_read(clead, cyyyymmddhh, GRAFprobsdir):
    infile = GRAFprobsdir + cyyyymmddhh + '_' + clead + \
             '_probs_europe_gamma_mixture.nc'
    if not os.path.exists(infile):
        print(f'Probability file not found: {infile}')
        sys.exit(1)
    nc = Dataset(infile, 'r')
    lat               = nc.variables['lat'][:,:]
    lon               = nc.variables['lon'][:,:]
    raw_p0p25mm_prob  = nc.variables['raw_p0p25mm_prob'][:,:]
    gamma_p0p25mm_prob = nc.variables['gamma_p0p25mm_prob'][:,:]
    nc.close()
    print(f'max raw/gamma P(>0.25mm) = '
          f'{np.max(raw_p0p25mm_prob):.3f} / {np.max(gamma_p0p25mm_prob):.3f}')
    return raw_p0p25mm_prob, gamma_p0p25mm_prob, lat, lon

# ---------------------------------------------------------------

def read_terrain(terrain_file):
    if terrain_file is None or not os.path.exists(terrain_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        terrain_file = os.path.join(script_dir, 'GRAF_Europe_terrain_info.nc')
    print(f'Reading terrain from {terrain_file}')
    nc = Dataset(terrain_file, 'r')
    terrain_height = nc.variables['terrain_height'][:,:]
    lats_terrain   = nc.variables['lats'][:,:]
    lons_terrain   = nc.variables['lons'][:,:]
    nc.close()
    return terrain_height, lats_terrain, lons_terrain

# ---------------------------------------------------------------

def plot_GRAF_europe(lat_1, lat_2, lat_0, lon_0,
                    lons, lats,
                    llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                    cyyyymmddhh, clead, precipitation_GRAF,
                    terrain_height, lats_terrain, lons_terrain,
                    gamma_p0p25mm_prob, raw_p0p25mm_prob,
                    GRAF_plot_dir):

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
                resolution='l', projection='lcc', area_thresh=1000.,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    x, y = m(lons, lats)
    x_terrain, y_terrain = m(lons_terrain, lats_terrain)

    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cyyyy  = cyyyymmddhh_valid[:4]
    cmm    = cyyyymmddhh_valid[4:6]
    cdd    = cyyyymmddhh_valid[6:8]
    chh    = cyyyymmddhh_valid[8:10]
    cmonths = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
    datestring = (chh + ' UTC ' + cdd + ' ' + cmonths[int(cmm)-1]
                  + ' ' + cyyyy)

    colorst_precip = [
        'White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange', '#FCD5D9',
        '#F6A3AE', '#FA5257', 'Orchid', '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]

    colorst_terrain = [
        'White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
        '#FCD5D9', '#F6A3AE', '#f17484']
    clevs_terrain = [-300, 0, 5, 10, 20, 50, 100, 300, 600,
                     1000, 1500, 2000, 2500, 3000]
    cmap_terrain = mpl.colors.LinearSegmentedColormap.from_list(
        '', colorst_terrain, N=len(colorst_terrain))
    norm_terrain = mcolors.BoundaryNorm(
        boundaries=clevs_terrain, ncolors=len(colorst_terrain), clip=True)

    clevs_prob = [0, 0.02, .1, .2, .3, .4, .5, .6, .7, 0.8, .9, .95, .97, 1.]

    clead_minus = str(int(clead) - 1)
    fig = plt.figure(figsize=(9, 11.))
    plt.suptitle(clead_minus + ' to ' + clead +
                 '-h GRAF+GFS-based forecasts (Europe), valid ' + datestring,
                 fontsize=15, y=0.988)

    def draw_map_borders(m):
        m.drawcoastlines(linewidth=0.6, color='Gray')
        m.drawcountries(linewidth=0.4, color='Gray')

    # --- (a) GRAF hourly precipitation
    ax1 = fig.add_axes([0.01, 0.540, 0.48, 0.36])
    ax1.set_title('(a) GRAF hourly precipitation', fontsize=12)
    CS1 = m.contourf(x, y, precipitation_GRAF, clevs_precip,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map_borders(m)
    cax = fig.add_axes([0.03, 0.520, 0.44, 0.013])
    cb = plt.colorbar(CS1, orientation='horizontal', cax=cax,
                      drawedges=True, ticks=clevs_precip, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Precipitation (mm)', fontsize=10)

    # --- (b) terrain elevation
    ax2 = fig.add_axes([0.51, 0.540, 0.48, 0.36])
    ax2.set_title('(b) GRAF terrain elevation', fontsize=12)
    CS2 = m.pcolormesh(x_terrain, y_terrain, terrain_height,
                       cmap=cmap_terrain, norm=norm_terrain, shading='nearest')
    draw_map_borders(m)
    cax = fig.add_axes([0.53, 0.520, 0.44, 0.013])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,
                      drawedges=True, ticks=clevs_terrain, format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Terrain elevation (m)', fontsize=10)

    # --- (c) Gamma mixture P(>0.25 mm/h)
    ax3 = fig.add_axes([0.01, 0.095, 0.48, 0.36])
    ax3.set_title('(c) Attention ResUNet Gamma Prob > 0.25 mm/h',
                  fontsize=11)
    CS3 = m.contourf(x, y, gamma_p0p25mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map_borders(m)
    cax = fig.add_axes([0.03, 0.075, 0.44, 0.013])
    cb = plt.colorbar(CS3, orientation='horizontal', cax=cax,
                      drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # --- (d) smoothed raw GRAF P(>0.25 mm/h)
    ax4 = fig.add_axes([0.51, 0.095, 0.48, 0.36])
    ax4.set_title('(d) Smoothed GRAF Prob > 0.25 mm/h', fontsize=11)
    CS4 = m.contourf(x, y, raw_p0p25mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map_borders(m)
    cax = fig.add_axes([0.53, 0.075, 0.44, 0.013])
    cb = plt.colorbar(CS4, orientation='horizontal', cax=cax,
                      drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    outfile = (GRAF_plot_dir + 'ResUnet_Europe_terrain_probs_IC'
               + cyyyymmddhh + '_lead' + clead + 'h_gamma.png')
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    print(f'Saved plot: {outfile}')
    return 0

# ====================================================================

cyyyymmddhh = sys.argv[1]
clead       = sys.argv[2]

config_file = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
print(f'Config: {config_file}')

GRAFdatadir_europe, GRAFprobsdir, GRAF_plot_dir, terrain_file = \
    read_config_file(config_file, 'DIRECTORIES')

os.makedirs(GRAF_plot_dir, exist_ok=True)

istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, \
    latmin, latmax, lonmin, lonmax, verif_local_time, \
    lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read_europe(clead, cyyyymmddhh, GRAFdatadir_europe)

raw_p0p25mm_prob, gamma_p0p25mm_prob, lat, lon = \
    probability_read(clead, cyyyymmddhh, GRAFprobsdir)

terrain_height, lats_terrain, lons_terrain = read_terrain(terrain_file)

if istat_GRAF == 0:
    # Use the actual SW and NE corners of the LCC grid rather than
    # lonmin/latmin/lonmax/latmax.  For an LCC grid stored south-to-north,
    # [0,0] is the SW corner and [-1,-1] is the NE corner.  lonmin comes
    # from the NW corner (where meridians converge) and does not pair with
    # latmin — mixing them creates a phantom corner far to the NW that
    # produces the blank white space in the upper-left of the plot.
    llcrnrlon = lons[0,  0 ]
    llcrnrlat = lats[0,  0 ]
    urcrnrlon = lons[-1, -1]
    urcrnrlat = lats[-1, -1]

    plot_GRAF_europe(lat_1, lat_2, lat_0, lon_0,
                     lons, lats,
                     llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                     cyyyymmddhh, clead, precipitation_GRAF,
                     terrain_height, lats_terrain, lons_terrain,
                     gamma_p0p25mm_prob, raw_p0p25mm_prob,
                     GRAF_plot_dir)
else:
    print('European GRAF forecast data not found.')
