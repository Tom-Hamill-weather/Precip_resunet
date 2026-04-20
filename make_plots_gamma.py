"""
e.g.,
python make_plots_gamma.py 2025120812 12

After inference is run with resunet_inference_gamma.py, this loads
the netCDF files produced by Gamma mixture model inference and makes plots.
It's a bit hard-coded to specific cases right now.

This version reads the Gamma model probability files with the "_gamma" suffix.
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
    """Detect if running on AWS or local laptop."""
    # Check for AWS paths (prioritize /data over /data2)
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path

    # Default to laptop
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    # Check if this is laptop config or AWS config
    if "GRAFdatadir_conus_laptop" in directory:
        # Laptop config - use same path for both old and new
        GRAFdatadir_conus_new = directory["GRAFdatadir_conus_laptop"]
        GRAFdatadir_conus_old = directory["GRAFdatadir_conus_laptop"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus_laptop"]
        GRAF_plot_dir = directory.get("GRAF_plot_dir", directory["GRAFprobsdir_conus_laptop"])
    else:
        # AWS/Cray config - has separate paths for old/new GRAF naming
        GRAFdatadir_conus_new = directory.get("GRAFdatadir_conus_new")
        GRAFdatadir_conus_old = directory.get("GRAFdatadir_conus_old", GRAFdatadir_conus_new)
        # For AWS, construct probs and plot directories from resnet_data_directory
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"

    print(f"  GRAF new path: {GRAFdatadir_conus_new}")
    print(f"  GRAF old path: {GRAFdatadir_conus_old}")
    print(f"  Probs path: {GRAFprobsdir_conus}")
    print(f"  Plot directory: {GRAF_plot_dir}")

    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, GRAF_plot_dir

# ---------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    import os
    import pygrib
    istat = -1
    fexist_grib = os.path.exists(gribfilename)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep = endStep)[0]
            lats, lons = grb.latlons()
            precipitation = grb.values # constrain max amt like data save.
            precipitation = np.where(precipitation > 75., \
                75., precipitation)
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
        print ('grib file does not exist.')
        istat = -1
        precipitation = np.empty((0,0))
        lats = np.empty((0,0))
        lons = np.empty((0,0))
        lon_0=0; lat_0=0; lat_1=0; lat_2=0 # dummy defaults

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2

# ---------------------------------------------------------------

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm= cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    # April 1, 2024 00Z is the dividing line between old and new GRAF naming
    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_old
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = prefix +cyyyymmdd_fcst+\
        'T'+chh_fcst+'0000Z.'+cyyyymmdd+'T'+chh+\
        '0000Z.PT'+clead+'H.CONUS@4km.APCP.SFC.grb2'
    infile = input_directory + input_file
    fexist1 = os.path.exists(infile)
    print (infile, fexist1)

    if fexist1 == True:
        istat, precipitation, lats, lons, lon_0, \
            lat_0, lat_1, lat_2 = read_gribdata(infile, il)
        ny, nx = np.shape(lats)
        latmax = np.max(lats); latmin = np.min(lats)
        lonmax = np.max(lons); lonmin = np.min(lons)
        tzoff = lons*12/180.
        verif_local_time = int(chh_fcst) + tzoff
    else:
        print ('  could not find ', infile)
        istat = -1
        ny = 0; nx = 0
        latmin = -99.99; latmax = -99.99
        lonmin = -999.99; lonmax = -999.99
        lon_0 = -999.99; lat_0 = -999.99
        lat_1 = -999.99; lat_2 = -999.99
        precipitation = np.empty((0,0))
        lats = np.empty((0,0), dtype=float)
        lons = np.empty((0,0), dtype=float)
        verif_local_time = np.empty((0,0), dtype=float)

    return istat, precipitation, lats, lons, ny, nx,\
        latmin, latmax, lonmin, lonmax, verif_local_time, \
        lon_0, lat_0, lat_1, lat_2

# -------------------------------------------------------------

def probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus_laptop):

    # Read Gamma mixture model probability files
    infile = GRAFprobsdir_conus_laptop + cyyyymmddhh + \
        '_'+ clead + '_probs_gamma_mixture.nc'

    nc = Dataset(infile,'r')
    fexist = os.path.exists(infile)
    if fexist == True:
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        raw_p0p25mm_prob = nc.variables['raw_p0p25mm_prob'][:,:]
        gamma_p0p25mm_prob = nc.variables['gamma_p0p25mm_prob'][:,:]
        raw_p1mm_prob = nc.variables['raw_p1mm_prob'][:,:]
        gamma_p1mm_prob = nc.variables['gamma_p1mm_prob'][:,:]
        raw_p2p5mm_prob = nc.variables['raw_p2p5mm_prob'][:,:]
        gamma_p2p5mm_prob = nc.variables['gamma_p2p5mm_prob'][:,:]
        raw_p5mm_prob = nc.variables['raw_p5mm_prob'][:,:]
        gamma_p5mm_prob = nc.variables['gamma_p5mm_prob'][:,:]
        raw_p10mm_prob = nc.variables['raw_p10mm_prob'][:,:]
        gamma_p10mm_prob = nc.variables['gamma_p10mm_prob'][:,:]
        print ('max raw Gamma 0p25mm = ', \
            np.max(raw_p0p25mm_prob), np.max(gamma_p0p25mm_prob))
        print ('max raw Gamma 1mm = ', \
            np.max(raw_p1mm_prob), np.max(gamma_p1mm_prob))
        print ('max raw Gamma 2p5mm = ', \
            np.max(raw_p2p5mm_prob), np.max(gamma_p2p5mm_prob))
        print ('max raw Gamma 5mm = ', \
            np.max(raw_p5mm_prob), np.max(gamma_p5mm_prob))
        print ('max raw Gamma 10mm = ', \
            np.max(raw_p10mm_prob), np.max(gamma_p10mm_prob))
        nc.close()
        istat_prob = 0
    else:
        print (infile)
        print ('no such file exists.  Exiting.')
        sys.exit()

    return istat_prob, raw_p0p25mm_prob, gamma_p0p25mm_prob, raw_p1mm_prob, \
        gamma_p1mm_prob, raw_p2p5mm_prob, gamma_p2p5mm_prob, raw_p5mm_prob, \
        gamma_p5mm_prob, raw_p10mm_prob, gamma_p10mm_prob, lat, lon

# -------------------------------------------------------------

def plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_gamma, \
        highprob_gamma, lowprob_raw, ltitle, htitle, GRAF_plot_dir,
        terrain_height=None):

    m = Basemap(rsphere=(6378137.00,6356752.3142),\
        resolution='l',projection='lcc',area_thresh=1000.,\
        lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,\
        llcrnrlon = lons[0,0],llcrnrlat=lats[0,0],\
        urcrnrlon = lons[-1,-1],urcrnrlat=lats[-1,-1])

    x, y = m(lons, lats)
    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cyyyy_valid = cyyyymmddhh_valid[0:4]
    cmm_valid = cyyyymmddhh_valid[4:6]
    cdd_valid = cyyyymmddhh_valid[6:8]
    chh_valid = cyyyymmddhh_valid[8:10]
    cmonths = ['Jan','Feb','Mar','Apr','May','Jun',\
        'Jul','Aug','Sep','Oct','Nov','Dec']
    cmonth = cmonths[int(cmm_valid)-1]
    datestring = chh_valid + ' UTC '+cdd_valid+' '+cmonth+' '+cyyyy_valid

    colorst = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange','#FCD5D9',\
        '#F6A3AE','#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    clevs = [0, 0.1, 0.254, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]
    clevs_prob = [0, 0.02, .1, .2, .3, .4, .5, .6, .7, 0.8, \
        .9, .95, .97, 1.]

    clead_minus = str(int(clead)-1)
    fig = plt.figure(figsize=(9,9.))
    plt.suptitle(clead_minus+' to '+clead+\
        '-h GRAF+GFS-based forecasts, valid '+\
        datestring,fontsize=17)

    # --- panel 1: GRAF deterministic amount

    axloc = [0.01,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(a) GRAF hourly precipitation amount'
    ax1.set_title(title, fontsize=12,color='Black')
    CS2 = m.contourf(x, y, precipitation_GRAF, clevs, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.55,0.44,0.015])
    cb = plt.colorbar(CS2,orientation='horizontal',cax=cax,\
        drawedges=True,ticks=clevs,format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Precipitation amount (mm)', fontsize=10)

    # --- panel 2: terrain height
    clevs_terrain = [0, 300, 600, 900, 1200, 1600, 2000, 2500, 3500]
    colorst_terrain = [
        'White',     # under: sea level and below
        '#D4F0B4',   # 0-300 m   (very light green)
        '#90CC58',   # 300-600 m (light green)
        '#D8C860',   # 600-900 m (yellow-green)
        '#C89020',   # 900-1200 m (golden)
        '#A06828',   # 1200-1600 m (tan-brown)
        '#885040',   # 1600-2000 m (brown)
        '#706060',   # 2000-2500 m (dark brown-gray)
        '#A0A0A0',   # 2500-3500 m (gray)
        '#D8D8D8',   # over 3500 m (light gray)
    ]
    axloc = [0.51,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    ax1.set_title('(b) Terrain height (m)', fontsize=12, color='Black')
    if terrain_height is not None:
        CS3 = m.contourf(x, y, terrain_height, clevs_terrain,
            cmap=None, colors=colorst_terrain, extend='both')
        m.drawcoastlines(linewidth=0.6, color='Gray')
        m.drawcountries(linewidth=0.4, color='Gray')
        m.drawstates(linewidth=0.2, color='Gray')
        cax = fig.add_axes([0.53,0.55,0.44,0.015])
        cb = plt.colorbar(CS3, orientation='horizontal', cax=cax,
            drawedges=True, ticks=clevs_terrain, format='%g')
        cb.ax.tick_params(labelsize=7)
        cb.set_label('Elevation (m)', fontsize=10)

    # --- panel 3: Gamma Model POP
    axloc = [0.01,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(c) Attention ResUNet Gamma '+ltitle
    ax1.set_title(title, fontsize=12,color='Black')
    CS2 = m.contourf(x, y, lowprob_gamma, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.09,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # --- panel 4: POP prob from GRAF convolution
    axloc = [0.51,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(d) Smoothed GRAF '+ltitle
    ax1.set_title(title, fontsize=12,color='Black')
    CS2 = m.contourf(x, y, lowprob_raw, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.53,0.09,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # ---- set plot title with _gamma suffix
    plot_title = GRAF_plot_dir + 'ResUnet_GRAF_probs_IC' + \
         cyyyymmddhh+'_lead'+clead+'h_gamma.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)
    #print ('Done!')
    istat = 0
    return istat

# ------------------------------------------------------------

def plot_GRAF_small(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_gamma, \
        highprob_gamma, lowprob_raw, ltitle, htitle, GRAF_plot_dir,
        terrain_height=None):

    """
    plots in a smaller domain centered roughly on area of interest.
    """

    m = Basemap(rsphere=(6378137.00,6356752.3142),\
        resolution='l',projection='lcc',area_thresh=1000.,\
        lat_1=lat_1,lat_2=lat_2,lat_0=lat_0,lon_0=lon_0,\
        llcrnrlon = llcrnrlon, llcrnrlat=llcrnrlat,\
        urcrnrlon = urcrnrlon, urcrnrlat=urcrnrlat)

    x, y = m(lons, lats)
    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cyyyy_valid = cyyyymmddhh_valid[0:4]
    cmm_valid = cyyyymmddhh_valid[4:6]
    cdd_valid = cyyyymmddhh_valid[6:8]
    chh_valid = cyyyymmddhh_valid[8:10]
    cmonths = ['Jan','Feb','Mar','Apr','May','Jun',\
        'Jul','Aug','Sep','Oct','Nov','Dec']
    cmonth = cmonths[int(cmm_valid)-1]
    datestring = chh_valid + ' UTC '+cdd_valid+' '+cmonth+' '+cyyyy_valid

    colorst = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange','#FCD5D9',\
        '#F6A3AE','#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    clevs = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]
    clevs_prob = [0, 0.02, .1, .2, .3, .4, .5, .6, .7, 0.8, \
        .9, .95, .97, 1.]

    clead_minus = str(int(clead)-1)
    fig = plt.figure(figsize=(7.,9.))
    plt.suptitle(clead_minus.zfill(2)+' to '+clead.zfill(2)+\
        '-h GRAF+GFS-based forecasts, valid '+\
        datestring,fontsize=15)

    # --- panel 1: GRAF deterministic amount

    axloc = [0.01,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(a) GRAF hourly precipitation'
    ax1.set_title(title, fontsize=11,color='Black')
    CS2 = m.contourf(x, y, precipitation_GRAF, clevs, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.55,0.44,0.015])
    cb = plt.colorbar(CS2,orientation='horizontal',cax=cax,\
        drawedges=True,ticks=clevs,format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Precipitation amount (mm)', fontsize=8)

    # --- panel 2: terrain height

    clevs_terrain = [0, 300, 600, 900, 1200, 1600, 2000, 2500, 3500]
    colorst_terrain = [
        'White',     # under: sea level and below
        '#D4F0B4',   # 0-300 m   (very light green)
        '#90CC58',   # 300-600 m (light green)
        '#D8C860',   # 600-900 m (yellow-green)
        '#C89020',   # 900-1200 m (golden)
        '#A06828',   # 1200-1600 m (tan-brown)
        '#885040',   # 1600-2000 m (brown)
        '#706060',   # 2000-2500 m (dark brown-gray)
        '#A0A0A0',   # 2500-3500 m (gray)
        '#D8D8D8',   # over 3500 m (light gray)
    ]
    axloc = [0.51,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    ax1.set_title('(b) Terrain height (m)', fontsize=11, color='Black')
    if terrain_height is not None:
        CS3 = m.contourf(x, y, terrain_height, clevs_terrain,
            cmap=None, colors=colorst_terrain, extend='both')
        m.drawcoastlines(linewidth=0.6, color='Gray')
        m.drawcountries(linewidth=0.4, color='Gray')
        m.drawstates(linewidth=0.2, color='Gray')
        cax = fig.add_axes([0.53,0.55,0.44,0.015])
        cb = plt.colorbar(CS3, orientation='horizontal', cax=cax,
            drawedges=True, ticks=clevs_terrain, format='%g')
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Elevation (m)', fontsize=8)

    # --- panel 3: Gamma Model POP

    axloc = [0.01,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(c) Attention ResUNet '+ltitle
    ax1.set_title(title, fontsize=11,color='Black')
    CS2 = m.contourf(x, y, lowprob_gamma, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.09,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Probability', fontsize=8)

    # --- panel 4: raw GRAF prob

    axloc = [0.51,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(d) Smoothed GRAF '+ltitle
    ax1.set_title(title, fontsize=11,color='Black')
    CS2 = m.contourf(x, y, lowprob_raw, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.53,0.09,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Probability', fontsize=8)

    # ---- set plot title with _gamma suffix

    plot_title = GRAF_plot_dir + 'ResUnet_small_GRAF_probs_IC' + \
         cyyyymmddhh+'_lead'+clead+'h_gamma.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)

    istat = 0
    return istat

# ====================================================================

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

# Select config file based on environment
if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

print(f"Using config file: {config_file_name}")
GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus_laptop, \
    GRAF_plot_dir = read_config_file(config_file_name, 'DIRECTORIES')

# Ensure plot directory exists
if not os.path.exists(GRAF_plot_dir):
    try:
        os.makedirs(GRAF_plot_dir)
        print(f"Created plot directory: {GRAF_plot_dir}")
    except OSError as e:
        print(f"Warning: Could not create plot directory {GRAF_plot_dir}: {e}")

istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
    lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)

istat_prob, raw_p0p25mm_prob, gamma_p0p25mm_prob, raw_p1mm_prob, \
    gamma_p1mm_prob, raw_p2p5mm_prob, gamma_p2p5mm_prob, raw_p5mm_prob, \
    gamma_p5mm_prob, raw_p10mm_prob, gamma_p10mm_prob, lat, lon = \
    probability_read(clead, cyyyymmddhh, GRAFprobsdir_conus_laptop)

# Read terrain height from static file
terrain_height = None
_terrain_candidates = [
    'GRAF_CONUS_terrain_info.nc',
    f'{AWS_BASE_PATH}/terrain/GRAF_CONUS_terrain_info.nc' if AWS_BASE_PATH else None,
]
for _tf in _terrain_candidates:
    if _tf and os.path.exists(_tf):
        _tnc = Dataset(_tf, 'r')
        terrain_height = _tnc.variables['terrain_height'][:,:]
        _tnc.close()
        print(f'Terrain read from {_tf}, max elevation = {terrain_height.max():.0f} m')
        break
if terrain_height is None:
    print('Warning: terrain file not found, panel 3 will be blank.')

# --- Here I've hard-coded domain locations and thresholds for
#     cases of interest.

if istat_GRAF == 0 and istat_prob == 0:

    if cyyyymmddhh == '2025120412':
        llcrnrlon = -125
        llcrnrlat = 33.5
        urcrnrlon = -103
        urcrnrlat = 53.
        lowprob_raw = raw_p0p25mm_prob
        highprob_raw = raw_p2p5mm_prob
        lowprob_gamma = gamma_p0p25mm_prob
        highprob_gamma = gamma_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'
    elif cyyyymmddhh == '2025120812':
        llcrnrlon = -125
        llcrnrlat = 33.5
        urcrnrlon = -103
        urcrnrlat = 53.
        lowprob_raw = raw_p0p25mm_prob
        highprob_raw = raw_p2p5mm_prob
        lowprob_gamma = gamma_p0p25mm_prob
        highprob_gamma = gamma_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'
    elif cyyyymmddhh == '2025122500':
        llcrnrlon = -125
        llcrnrlat = 25
        urcrnrlon = -108.
        urcrnrlat = 42
        lowprob_raw = raw_p1mm_prob
        highprob_raw = raw_p5mm_prob
        lowprob_gamma = gamma_p1mm_prob
        highprob_gamma = gamma_p5mm_prob
        ltitle = 'Prob > 1 mm/h'
        htitle = 'Prob > 5 mm/h'
    elif cyyyymmddhh == '2025120300':
        llcrnrlon = -112
        llcrnrlat = 33.
        urcrnrlon = -90
        urcrnrlat = 48.
        lowprob_raw = raw_p0p25mm_prob
        highprob_raw = raw_p2p5mm_prob
        lowprob_gamma = gamma_p0p25mm_prob
        highprob_gamma = gamma_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'
    else:
        llcrnrlon = -95
        llcrnrlat = 32.
        urcrnrlon = -55.
        urcrnrlat = 47.
        lowprob_raw = raw_p0p25mm_prob
        highprob_raw = raw_p2p5mm_prob
        lowprob_gamma = gamma_p0p25mm_prob
        highprob_gamma = gamma_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'
        
        
        
        36, -117

    # Plotting, first CONUS scale and then zoomed in.

    #istat = plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
    #    cyyyymmddhh, clead, precipitation_GRAF, lowprob_gamma, \
    #    highprob_gamma, lowprob_raw, ltitle, htitle, GRAF_plot_dir,
    #    terrain_height=terrain_height)

    istat = plot_GRAF_small(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_gamma, \
        highprob_gamma, lowprob_raw, ltitle, htitle, GRAF_plot_dir,
        terrain_height=terrain_height)

else:
    print ('GRAF forecast or probability data not found.')
