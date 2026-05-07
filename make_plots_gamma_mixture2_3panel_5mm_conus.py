"""
python make_plots_gamma_mixture2_3panel_5mm_conus.py cyyyymmddhh clead
e.g.,
python make_plots_gamma_mixture2_3panel_5mm_conus.py 2025120812 12

Three-panel horizontal plot over the full CONUS domain:
  (a) GRAF hourly precipitation amount
  (b) Attention ResUNet Gamma mixture P(>5 mm/h)
  (c) Smoothed raw ensemble P(>5 mm/h)
"""

from configparser import ConfigParser
import numpy as np
import os, sys
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
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
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if "GRAFdatadir_conus_laptop" in directory:
        GRAFdatadir_conus_new = directory["GRAFdatadir_conus_laptop"]
        GRAFdatadir_conus_old = directory["GRAFdatadir_conus_laptop"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus_laptop"]
        GRAF_plot_dir = directory.get("GRAF_plot_dir", directory["GRAFprobsdir_conus_laptop"])
        mrms_data_directory = directory.get("mrms_data_directory", "")
    else:
        GRAFdatadir_conus_new = directory.get("GRAFdatadir_conus_new")
        GRAFdatadir_conus_old = directory.get("GRAFdatadir_conus_old", GRAFdatadir_conus_new)
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"
        mrms_data_directory = directory.get("mrms_data_directory", f"{base_dir}/MRMS")

    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, \
        GRAF_plot_dir, mrms_data_directory

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
        '_' + str(int(clead)) + '_probs_gamma_mixture.nc'
    if not os.path.exists(infile):
        print(infile)
        print('no such file exists.  Exiting.')
        sys.exit()

    nc = Dataset(infile, 'r')
    lat = nc.variables['lat'][:,:]
    lon = nc.variables['lon'][:,:]
    raw_p5mm_prob   = np.ma.masked_invalid(nc.variables['raw_p5mm_prob'][:,:])
    gamma_p5mm_prob = np.ma.masked_invalid(nc.variables['gamma_p5mm_prob'][:,:])
    nc.close()

    print('max raw/gamma P(>5mm) = ',
        np.max(raw_p5mm_prob), np.max(gamma_p5mm_prob))
    return raw_p5mm_prob, gamma_p5mm_prob, lat, lon

# -------------------------------------------------------------

def read_mrms(mrms_data_directory, validity_date):
    """Return (mrms_precip, mrms_lats, mrms_lons) or None on failure.
    Bad-quality pixels are masked."""
    infile = os.path.join(mrms_data_directory, validity_date[:6],
        f'MRMS_1h_pamt_and_data_qual_{validity_date}.nc')
    if not os.path.exists(infile):
        print(f'MRMS file not found: {infile}')
        return None, None, None
    try:
        nc = Dataset(infile, 'r')
        precip  = nc.variables['precipitation'][:,:]
        quality = nc.variables['data_quality'][:,:]
        mlats   = nc.variables['lats'][:,:]
        mlons   = nc.variables['lons'][:,:]
        nc.close()
        precip = np.where(precip > 75., 75., precip)
        precip = np.ma.masked_where(quality <= 0.01, precip)
        return precip, mlats, mlons
    except Exception as e:
        print(f'Error reading MRMS {infile}: {e}')
        return None, None, None

# -------------------------------------------------------------

def max_prob_domain(gamma_p5mm_prob, lat, lon, half_side_km=400.):
    """Return llcrnrlon/lat, urcrnrlon/lat centred on the peak gamma 5mm prob.
    Search is restricted to the CONUS mainland to avoid tropical/ocean artifacts."""
    conus_mask = (lat >= 24.) & (lat <= 52.) & (lon >= -126.) & (lon <= -65.)
    masked = gamma_p5mm_prob.copy()
    masked[~conus_mask] = -1.
    idx = np.unravel_index(np.argmax(masked), masked.shape)
    clat = float(lat[idx])
    clon = float(lon[idx])
    dlat = half_side_km / 111.0
    dlon = half_side_km / (111.0 * np.cos(np.radians(clat)))
    print(f'Max gamma P(>5mm) at lat={clat:.2f} lon={clon:.2f}; '
          f'zoom box ±{half_side_km:.0f} km')
    return clon - dlon, clat - dlat, clon + dlon, clat + dlat

# -------------------------------------------------------------

def plot_3panel(lat_1, lat_2, lat_0, lon_0, lons, lats,
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
        cyyyymmddhh, clead, precipitation_GRAF,
        gamma_p5mm_prob, raw_p5mm_prob, GRAF_plot_dir,
        plot_suffix=''):

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
        resolution='l', projection='lcc', area_thresh=1000.,
        lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
        llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
        urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    x, y = m(lons, lats)

    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cmonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cmonth = cmonths[int(cyyyymmddhh_valid[4:6]) - 1]
    datestring = (cyyyymmddhh_valid[8:10] + ' UTC ' +
                  cyyyymmddhh_valid[6:8] + ' ' + cmonth + ' ' +
                  cyyyymmddhh_valid[0:4])

    # Precipitation color scale
    colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange', '#FCD5D9',
        '#F6A3AE', '#FA5257', 'Orchid', '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]

    # Probability color scale
    clevs_prob = [0, 0.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, 0.8, .9, 1.]

    clead_minus = str(int(clead) - 1)

    pan_left   = [0.015, 0.340, 0.665]
    pan_bot    = 0.12
    pan_w      = 0.313
    pan_h      = 0.742
    cbar_y     = 0.060
    cbar_h     = 0.025
    cbar_inset = 0.010

    fig = plt.figure(figsize=(14, 5))
    plt.suptitle(clead_minus + ' to ' + clead +
        '-h GRAF+GFS-based forecasts, valid ' + datestring,
        fontsize=20, y=0.975)

    def draw_map(ax, CS, panel_x, title, cbar_label, cb_ticks):
        ax.set_title(title, fontsize=14, color='Black')
        m.drawcoastlines(linewidth=0.6, color='Gray')
        m.drawcountries(linewidth=0.4, color='Gray')
        m.drawstates(linewidth=0.2, color='Gray')
        cax = fig.add_axes([panel_x + cbar_inset, cbar_y,
                            pan_w - 2 * cbar_inset, cbar_h])
        cb = plt.colorbar(CS, orientation='horizontal', cax=cax,
                          drawedges=True, ticks=cb_ticks, format='%g')
        cb.ax.tick_params(labelsize=6)
        cb.set_label(cbar_label, fontsize=9)

    # --- panel (a): GRAF hourly precipitation ---
    ax1 = fig.add_axes([pan_left[0], pan_bot, pan_w, pan_h])
    CS1 = m.contourf(x, y, precipitation_GRAF, clevs_precip,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax1, CS1, pan_left[0],
             '(a) GRAF hourly precipitation',
             'Precipitation (mm)', clevs_precip)

    # --- panel (b): ResUNet gamma mixture P(>5 mm/h) ---
    ax2 = fig.add_axes([pan_left[1], pan_bot, pan_w, pan_h])
    CS2 = m.contourf(x, y, gamma_p5mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax2, CS2, pan_left[1],
             r'(b) ResUNet Gamma Prob $>$ 5 mm/h',
             'Probability', clevs_prob)

    # --- panel (c): smoothed raw ensemble P(>5 mm/h) ---
    ax3 = fig.add_axes([pan_left[2], pan_bot, pan_w, pan_h])
    CS3 = m.contourf(x, y, raw_p5mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax3, CS3, pan_left[2],
             r'(c) Smoothed GRAF Prob $>$ 5 mm/h',
             'Probability', clevs_prob)

    plot_title = (GRAF_plot_dir + 'ResUnet_3panel_5mm_CONUS' + plot_suffix +
                  '_IC' + cyyyymmddhh + '_lead' + clead + 'h_gamma.png')
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print('saving plot to file = ', plot_title)
    return 0

# -------------------------------------------------------------

def plot_4panel_zoom(lat_1, lat_2, lat_0, lon_0, lons, lats,
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
        cyyyymmddhh, clead, precipitation_GRAF,
        gamma_p5mm_prob, raw_p5mm_prob,
        mrms_precip, mrms_lats, mrms_lons,
        GRAF_plot_dir):
    """2x2 zoom plot:  top-left=GRAF precip, top-right=MRMS obs,
    bottom-left=ResUNet gamma P(>5mm), bottom-right=smoothed GRAF P(>5mm)."""

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
        resolution='i', projection='lcc', area_thresh=500.,
        lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
        llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
        urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    x_graf,  y_graf  = m(lons, lats)
    if mrms_precip is not None:
        x_mrms, y_mrms = m(mrms_lons, mrms_lats)

    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cmonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cmonth = cmonths[int(cyyyymmddhh_valid[4:6]) - 1]
    datestring = (cyyyymmddhh_valid[8:10] + ' UTC ' +
                  cyyyymmddhh_valid[6:8] + ' ' + cmonth + ' ' +
                  cyyyymmddhh_valid[0:4])

    colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
        '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange', '#FCD5D9',
        '#F6A3AE', '#FA5257', 'Orchid', '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]
    clevs_prob   = [0, 0.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, 0.8, .9, 1.]

    clead_minus = str(int(clead) - 1)

    # Layout: 2 columns × 2 rows, each panel ~square for 800×800 km domain
    # Figure: 10" wide × 9" tall
    pan_w      = 0.44
    pan_h      = 0.38
    left_col   = 0.04
    right_col  = 0.53
    top_row    = 0.55   # bottom of top panels
    bot_row    = 0.10   # bottom of bottom panels
    cbar_h     = 0.022
    cbar_inset = 0.01
    top_cbar_y = top_row - 0.010
    bot_cbar_y = bot_row - 0.010

    fig = plt.figure(figsize=(10, 9))
    plt.suptitle(clead_minus + ' to ' + clead +
        '-h forecasts, valid ' + datestring,
        fontsize=16, y=0.98)

    def draw_map(ax, CS, col_x, cbar_y, title, cbar_label, cb_ticks):
        ax.set_title(title, fontsize=11, color='Black')
        m.drawcoastlines(linewidth=0.7, color='Gray')
        m.drawcountries(linewidth=0.5, color='Gray')
        m.drawstates(linewidth=0.3, color='Gray')
        cax = fig.add_axes([col_x + cbar_inset, cbar_y,
                            pan_w - 2 * cbar_inset, cbar_h])
        cb = plt.colorbar(CS, orientation='horizontal', cax=cax,
                          drawedges=True, ticks=cb_ticks, format='%g')
        cb.ax.tick_params(labelsize=5)
        cb.set_label(cbar_label, fontsize=8)

    # --- (a) top-left: GRAF hourly precipitation ---
    ax1 = fig.add_axes([left_col, top_row, pan_w, pan_h])
    CS1 = m.contourf(x_graf, y_graf, precipitation_GRAF, clevs_precip,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax1, CS1, left_col, top_cbar_y,
             '(a) GRAF hourly precipitation', 'Precipitation (mm)', clevs_precip)

    # --- (b) top-right: MRMS observed precipitation ---
    ax2 = fig.add_axes([right_col, top_row, pan_w, pan_h])
    if mrms_precip is not None:
        CS2 = m.contourf(x_mrms, y_mrms, mrms_precip, clevs_precip,
                         cmap=None, colors=colorst_precip, extend='both')
        draw_map(ax2, CS2, right_col, top_cbar_y,
                 '(b) MRMS observed precipitation', 'Precipitation (mm)', clevs_precip)
    else:
        ax2.set_title('(b) MRMS observed precipitation', fontsize=11)
        ax2.text(0.5, 0.5, 'Not available', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=12, color='gray')
        m.drawcoastlines(linewidth=0.7, color='Gray')
        m.drawcountries(linewidth=0.5, color='Gray')
        m.drawstates(linewidth=0.3, color='Gray')

    # --- (c) bottom-left: ResUNet gamma P(>5mm) ---
    ax3 = fig.add_axes([left_col, bot_row, pan_w, pan_h])
    CS3 = m.contourf(x_graf, y_graf, gamma_p5mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax3, CS3, left_col, bot_cbar_y,
             r'(c) ResUNet Gamma Prob $>$ 5 mm/h', 'Probability', clevs_prob)

    # --- (d) bottom-right: smoothed raw GRAF P(>5mm) ---
    ax4 = fig.add_axes([right_col, bot_row, pan_w, pan_h])
    CS4 = m.contourf(x_graf, y_graf, raw_p5mm_prob, clevs_prob,
                     cmap=None, colors=colorst_precip, extend='both')
    draw_map(ax4, CS4, right_col, bot_cbar_y,
             r'(d) Smoothed GRAF Prob $>$ 5 mm/h', 'Probability', clevs_prob)

    plot_title = (GRAF_plot_dir + 'ResUnet_4panel_5mm_zoom_IC' +
                  cyyyymmddhh + '_lead' + clead + 'h_gamma.png')
    fig.savefig(plot_title, dpi=300, bbox_inches='tight')
    print('saving plot to file = ', plot_title)

# ====================================================================

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, \
    GRAF_plot_dir, mrms_data_directory = \
    read_config_file(config_file_name, 'DIRECTORIES')

if not os.path.exists(GRAF_plot_dir):
    try:
        os.makedirs(GRAF_plot_dir)
    except OSError as e:
        print(f"Warning: Could not create plot directory {GRAF_plot_dir}: {e}")

istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
    lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)

raw_p5mm_prob, gamma_p5mm_prob, lat, lon = \
    probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus)

validity_date = dateshift(cyyyymmddhh, int(clead))
mrms_precip, mrms_lats, mrms_lons = read_mrms(mrms_data_directory, validity_date)

if istat_GRAF == 0:
    # Full CONUS extent — 3-panel
    plot_3panel(lat_1, lat_2, lat_0, lon_0, lons, lats,
        lonmin, latmin, lonmax, latmax,
        cyyyymmddhh, clead, precipitation_GRAF,
        gamma_p5mm_prob, raw_p5mm_prob, GRAF_plot_dir,
        plot_suffix='')

    # Zoomed 4-panel centred on peak 5mm probability (~800 km box)
    zllcrnrlon, zllcrnrlat, zurcrnrlon, zurcrnrlat = \
        max_prob_domain(gamma_p5mm_prob, lat, lon, half_side_km=400.)
    plot_4panel_zoom(lat_1, lat_2, lat_0, lon_0, lons, lats,
        zllcrnrlon, zllcrnrlat, zurcrnrlon, zurcrnrlat,
        cyyyymmddhh, clead, precipitation_GRAF,
        gamma_p5mm_prob, raw_p5mm_prob,
        mrms_precip, mrms_lats, mrms_lons,
        GRAF_plot_dir)
else:
    print('GRAF forecast data not found.')
