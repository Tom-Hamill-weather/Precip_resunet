"""
python make_plots_gamma_mixture2.py cyyyymmddhh clead
e.g.,
python make_plots_gamma_mixture2.py 2025120812 12

Four-panel plot:
  (a) GRAF hourly precipitation amount
  (b) GRAF terrain elevation
  (c) Gamma mixture model P(>0.25 mm/h)
  (d) Smoothed raw ensemble P(>0.25 mm/h)
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

# --- Auto-detect environment (AWS vs local) ---
def detect_environment():
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if "GRAFdatadir_conus_laptop" in directory:
        GRAFdatadir_conus_new = directory["GRAFdatadir_conus_laptop"]
        GRAFdatadir_conus_old = directory["GRAFdatadir_conus_laptop"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus_laptop"]
        GRAF_plot_dir = directory.get("GRAF_plot_dir", directory["GRAFprobsdir_conus_laptop"])
        terrain_file = directory.get("terrain_file", None)
    else:
        GRAFdatadir_conus_new = directory.get("GRAFdatadir_conus_new")
        GRAFdatadir_conus_old = directory.get("GRAFdatadir_conus_old", GRAFdatadir_conus_new)
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"
        terrain_file = directory.get("terrain_file", f"{base_dir}/terrain/GRAF_CONUS_terrain_info.nc")

    print(f"  GRAF new path: {GRAFdatadir_conus_new}")
    print(f"  GRAF old path: {GRAFdatadir_conus_old}")
    print(f"  Probs path: {GRAFprobsdir_conus}")
    print(f"  Plot directory: {GRAF_plot_dir}")
    print(f"  Terrain file: {terrain_file}")

    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, \
        GRAF_plot_dir, terrain_file

# ---------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    import pygrib
    istat = -1
    if os.path.exists(gribfilename):
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep=endStep)[0]
            lats, lons = grb.latlons()
            precipitation = np.where(grb.values > 75., 75., grb.values)
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

# ---------------------------------------------------------------

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm = cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_old
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = prefix + cyyyymmdd_fcst + \
        'T' + chh_fcst + '0000Z.' + cyyyymmdd + 'T' + chh + \
        '0000Z.PT' + clead + 'H.CONUS@4km.APCP.SFC.grb2'
    infile = input_directory + input_file
    fexist1 = os.path.exists(infile)
    print(infile, fexist1)

    if fexist1:
        istat, precipitation, lats, lons, lon_0, \
            lat_0, lat_1, lat_2 = read_gribdata(infile, il)
        ny, nx = np.shape(lats)
        latmax = np.max(lats); latmin = np.min(lats)
        lonmax = np.max(lons); lonmin = np.min(lons)
        tzoff = lons * 12 / 180.
        verif_local_time = int(chh_fcst) + tzoff
    else:
        print('  could not find ', infile)
        istat = -1
        ny = 0; nx = 0
        latmin = -99.99; latmax = -99.99
        lonmin = -999.99; lonmax = -999.99
        lon_0 = -999.99; lat_0 = -999.99
        lat_1 = -999.99; lat_2 = -999.99
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0), dtype=float)
        lons = np.empty((0, 0), dtype=float)
        verif_local_time = np.empty((0, 0), dtype=float)

    return istat, precipitation, lats, lons, ny, nx, \
        latmin, latmax, lonmin, lonmax, verif_local_time, \
        lon_0, lat_0, lat_1, lat_2

# -------------------------------------------------------------

def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus):
    infile = GRAFprobsdir_conus + cyyyymmddhh + \
        '_' + clead + '_probs_gamma_mixture.nc'
    fexist = os.path.exists(infile)
    if not fexist:
        print(infile)
        print('no such file exists.  Exiting.')
        sys.exit()

    nc = Dataset(infile, 'r')
    lat = nc.variables['lat'][:,:]
    lon = nc.variables['lon'][:,:]
    raw_p0p25mm_prob = nc.variables['raw_p0p25mm_prob'][:,:]
    gamma_p0p25mm_prob = nc.variables['gamma_p0p25mm_prob'][:,:]
    nc.close()

    print('max raw/gamma P(>0.25mm) = ',
        np.max(raw_p0p25mm_prob), np.max(gamma_p0p25mm_prob))
    return raw_p0p25mm_prob, gamma_p0p25mm_prob, lat, lon

# -------------------------------------------------------------

def read_terrain(terrain_file):
    if terrain_file is None or not os.path.exists(terrain_file):
        # Fall back to file alongside this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        terrain_file = os.path.join(script_dir, 'GRAF_CONUS_terrain_info.nc')
    print(f'Reading terrain from {terrain_file}')
    nc = Dataset(terrain_file, 'r')
    terrain_height = nc.variables['terrain_height'][:,:]
    lats_terrain = nc.variables['lats'][:,:]
    lons_terrain = nc.variables['lons'][:,:]
    nc.close()
    return terrain_height, lats_terrain, lons_terrain

# -------------------------------------------------------------

def plot_GRAF2(lat_1, lat_2, lat_0, lon_0, lons, lats,
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
        cyyyymmddhh, clead, precipitation_GRAF,
        terrain_height, lats_terrain, lons_terrain,
        gamma_p0p25mm_prob, raw_p0p25mm_prob, GRAF_plot_dir):

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
        resolution='l', projection='lcc', area_thresh=1000.,
        lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
        llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
        urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    x, y = m(lons, lats)
    x_terrain, y_terrain = m(lons_terrain, lats_terrain)

    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cyyyy_valid = cyyyymmddhh_valid[0:4]
    cmm_valid = cyyyymmddhh_valid[4:6]
    cdd_valid = cyyyymmddhh_valid[6:8]
    chh_valid = cyyyymmddhh_valid[8:10]
    cmonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cmonth = cmonths[int(cmm_valid) - 1]
    datestring = chh_valid + ' UTC ' + cdd_valid + ' ' + cmonth + ' ' + cyyyy_valid

    # Precipitation color scale
    colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange', '#FCD5D9',
        '#F6A3AE', '#FA5257', 'Orchid', '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]

    # Terrain color scale
    colorst_terrain = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
        '#FCD5D9', '#F6A3AE', '#f17484']
    clevs_terrain = [-300, 0, 5, 10, 20, 50, 100, 300, 600, 1000, 1500, 2000, 2500, 3000]
    cmap_terrain = mpl.colors.LinearSegmentedColormap.from_list(
        "", colorst_terrain, N=len(colorst_terrain))
    norm_terrain = mcolors.BoundaryNorm(boundaries=clevs_terrain,
        ncolors=len(colorst_terrain), clip=True)

    # Probability color scale
    clevs_prob = [0, 0.02, .1, .2, .3, .4, .5, .6, .7, 0.8, .9, .95, .97, 1.]

    clead_minus = str(int(clead) - 1)
    fig = plt.figure(figsize=(9, 11.))
    plt.suptitle(clead_minus + ' to ' + clead +
        '-h GRAF+GFS-based forecasts, valid ' + datestring, fontsize=17, y=0.975)

    # --- panel (a): GRAF hourly precipitation
    axloc = [0.01, 0.545, 0.48, 0.40]
    ax1 = fig.add_axes(axloc)
    ax1.set_title('(a) GRAF hourly precipitation', fontsize=12, color='Black')
    CS1 = m.contourf(x, y, precipitation_GRAF, clevs_precip,
        cmap=None, colors=colorst_precip, extend='both')
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.4, color='Gray')
    m.drawstates(linewidth=0.2, color='Gray')
    cax = fig.add_axes([0.03, 0.537, 0.44, 0.013])
    cb = plt.colorbar(CS1, orientation='horizontal', cax=cax,
        drawedges=True, ticks=clevs_precip, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Precipitation (mm)', fontsize=10)

    # --- panel (b): terrain elevation
    axloc = [0.51, 0.545, 0.48, 0.40]
    ax2 = fig.add_axes(axloc)
    ax2.set_title('(b) GRAF terrain elevation', fontsize=12, color='Black')
    CS2 = m.pcolormesh(x_terrain, y_terrain, terrain_height,
        cmap=cmap_terrain, norm=norm_terrain, shading='nearest')
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.4, color='Gray')
    m.drawstates(linewidth=0.2, color='Gray')
    cax = fig.add_axes([0.53, 0.537, 0.44, 0.013])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,
        drawedges=True, ticks=clevs_terrain, format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Terrain elevation (m)', fontsize=10)

    # --- panel (c): Gamma mixture P(>0.25 mm/h)
    axloc = [0.01, 0.080, 0.48, 0.40]
    ax3 = fig.add_axes(axloc)
    ax3.set_title('(c) Attention ResUNet Gamma Prob > 0.25 mm/h', fontsize=11, color='Black')
    CS3 = m.contourf(x, y, gamma_p0p25mm_prob, clevs_prob,
        cmap=None, colors=colorst_precip, extend='both')
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.4, color='Gray')
    m.drawstates(linewidth=0.2, color='Gray')
    cax = fig.add_axes([0.03, 0.072, 0.44, 0.013])
    cb = plt.colorbar(CS3, orientation='horizontal', cax=cax,
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # --- panel (d): smoothed raw ensemble P(>0.25 mm/h)
    axloc = [0.51, 0.080, 0.48, 0.40]
    ax4 = fig.add_axes(axloc)
    ax4.set_title('(d) Smoothed GRAF Prob > 0.25 mm/h', fontsize=11, color='Black')
    CS4 = m.contourf(x, y, raw_p0p25mm_prob, clevs_prob,
        cmap=None, colors=colorst_precip, extend='both')
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.4, color='Gray')
    m.drawstates(linewidth=0.2, color='Gray')
    cax = fig.add_axes([0.53, 0.072, 0.44, 0.013])
    cb = plt.colorbar(CS4, orientation='horizontal', cax=cax,
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    plot_title = GRAF_plot_dir + 'ResUnet_GRAF_terrain_probs_IC' + \
        cyyyymmddhh + '_lead' + clead + 'h_gamma.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print('saving plot to file = ', plot_title)
    return 0

# ====================================================================

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

print(f"Using config file: {config_file_name}")
GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, \
    GRAF_plot_dir, terrain_file = read_config_file(config_file_name, 'DIRECTORIES')

if not os.path.exists(GRAF_plot_dir):
    try:
        os.makedirs(GRAF_plot_dir)
        print(f"Created plot directory: {GRAF_plot_dir}")
    except OSError as e:
        print(f"Warning: Could not create plot directory {GRAF_plot_dir}: {e}")

istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
    lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)

raw_p0p25mm_prob, gamma_p0p25mm_prob, lat, lon = \
    probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus)

terrain_height, lats_terrain, lons_terrain = read_terrain(terrain_file)

if istat_GRAF == 0:

    # Domain selection for zoomed view
    if cyyyymmddhh == '2025120412':
        llcrnrlon = -125; llcrnrlat = 33.5
        urcrnrlon = -103; urcrnrlat = 53.
    elif cyyyymmddhh == '2025120812':
        llcrnrlon = -125; llcrnrlat = 33.5
        urcrnrlon = -103; urcrnrlat = 53.
    elif cyyyymmddhh == '2025122500':
        llcrnrlon = -125; llcrnrlat = 25
        urcrnrlon = -108.; urcrnrlat = 42
    elif cyyyymmddhh == '2025120300':
        llcrnrlon = -112; llcrnrlat = 33.
        urcrnrlon = -90; urcrnrlat = 48.
    else:
        llcrnrlon = -95; llcrnrlat = 32.
        urcrnrlon = -55.; urcrnrlat = 47.

    istat = plot_GRAF2(lat_1, lat_2, lat_0, lon_0, lons, lats,
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
        cyyyymmddhh, clead, precipitation_GRAF,
        terrain_height, lats_terrain, lons_terrain,
        gamma_p0p25mm_prob, raw_p0p25mm_prob, GRAF_plot_dir)

else:
    print('GRAF forecast data not found.')
