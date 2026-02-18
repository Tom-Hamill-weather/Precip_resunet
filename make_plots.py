"""
python make_plots.py cyyyymmddhh clead
e.g.,
python make_plots.py 2025120812 12

After inference is run, this loads the netCDF files produced by inference
and makes plots.   It's a bit hard-coded to specific cases right now.
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

TRAIN_DIR = '../resnet_data/trainings'

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]
    GRAFdatadir_conus_laptop = directory["GRAFdatadir_conus_laptop"]
    GRAFprobsdir_conus_laptop = directory["GRAFprobsdir_conus_laptop"]
    GRAF_plot_dir = directory["GRAF_plot_dir"]
    return GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop, GRAF_plot_dir

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

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    cyyyymm= cyyyymmddhh[0:6]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) > 2024040512:
        input_directory =  GRAFdatadir_conus_laptop
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_laptop
        prefix = 'grid.hdo-graflr_conus.'
        
    input_directory = input_directory + cyyyymm + '/'
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
    
    infile = GRAFprobsdir_conus_laptop + cyyyymmddhh + \
        '_'+ clead + '_probs.nc'
    
    nc = Dataset(infile,'r')
    fexist = os.path.exists(infile)
    if fexist == True:
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        raw_p0p25mm_prob = nc.variables['raw_p0p25mm_prob'][:,:]
        dl_p0p25mm_prob = nc.variables['dl_p0p25mm_prob'][:,:]
        raw_p1mm_prob = nc.variables['raw_p1mm_prob'][:,:]
        dl_p1mm_prob = nc.variables['dl_p1mm_prob'][:,:]
        raw_p2p5mm_prob = nc.variables['raw_p2p5mm_prob'][:,:]
        dl_p2p5mm_prob = nc.variables['dl_p2p5mm_prob'][:,:]
        raw_p5mm_prob = nc.variables['raw_p5mm_prob'][:,:]
        dl_p5mm_prob = nc.variables['dl_p5mm_prob'][:,:]
        raw_p10mm_prob = nc.variables['raw_p10mm_prob'][:,:]
        dl_p10mm_prob = nc.variables['dl_p10mm_prob'][:,:]
        raw_p25mm_prob = nc.variables['raw_p25mm_prob'][:,:]
        dl_p25mm_prob = nc.variables['dl_p25mm_prob'][:,:]
        print ('max raw DL 0p25mm = ', \
            np.max(raw_p0p25mm_prob), np.max(dl_p0p25mm_prob))
        print ('max raw DL 1mm = ', \
            np.max(raw_p1mm_prob), np.max(dl_p1mm_prob))
        print ('max raw DL 0p25mm = ', \
            np.max(raw_p2p5mm_prob), np.max(dl_p2p5mm_prob))
        print ('max raw DL 5mm = ', \
            np.max(raw_p5mm_prob), np.max(dl_p5mm_prob))
        print ('max raw DL 10mm = ', \
            np.max(raw_p10mm_prob), np.max(dl_p10mm_prob))
        print ('max raw DL 25mm = ', \
            np.max(raw_p25mm_prob), np.max(dl_p25mm_prob))
        nc.close()
        istat_prob = 0
    else:
        print (infile)
        print ('no such file exists.  Exiting.')
        sys.exit()
        
    return istat_prob, raw_p0p25mm_prob, dl_p0p25mm_prob, raw_p1mm_prob, \
        dl_p1mm_prob, raw_p2p5mm_prob, dl_p2p5mm_prob, raw_p5mm_prob, \
        dl_p5mm_prob, raw_p10mm_prob, dl_p10mm_prob, raw_p25mm_prob, \
        dl_p25mm_prob, lat, lon

# -------------------------------------------------------------

def plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_dl, \
        highprob_dl, lowprob_raw, ltitle, htitle, GRAF_plot_dir):
            
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
        '-h GRAF-based forecasts, valid '+\
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

    # --- panel 2: GRAF POP
    axloc = [0.51,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(b) Attention ResUNet '+ltitle
    ax1.set_title(title, fontsize=12,color='Black')
    CS2 = m.contourf(x, y, lowprob_dl, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.53,0.55,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # --- panel 3: 5 mm prob
    axloc = [0.01,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(c) Attention ResUNet '+htitle 
    ax1.set_title(title, fontsize=12,color='Black')
    CS2 = m.contourf(x, y, highprob_dl, clevs_prob, \
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

    # ---- set plot title
    plot_title = GRAF_plot_dir + 'ResUnet_GRAF_probs_IC' + \
         cyyyymmddhh+'_lead'+clead+'h.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)
    #print ('Done!')
    istat = 0
    return istat
    
# ------------------------------------------------------------
    
def plot_GRAF_small(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_dl, \
        highprob_dl, lowprob_raw, ltitle, htitle, GRAF_plot_dir):
        
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
    plt.suptitle(clead_minus+' to '+clead+\
        '-h GRAF-based forecasts, valid '+\
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

    # --- panel 2: low DL prob
    
    axloc = [0.51,0.58,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(b) Attention ResUNet '+ltitle
    ax1.set_title(title, fontsize=11,color='Black')
    CS2 = m.contourf(x, y, lowprob_dl, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.53,0.55,0.44,0.015])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Probability', fontsize=8)

    # --- panel 3: high prob
    
    axloc = [0.01,0.12,0.48,0.34]
    ax1 = fig.add_axes(axloc)
    title = '(c) Attention ResUNet '+htitle 
    ax1.set_title(title, fontsize=11,color='Black')
    CS2 = m.contourf(x, y, highprob_dl, clevs_prob, \
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

    # ---- set plot title
    
    plot_title = GRAF_plot_dir + 'ResUnet_small_GRAF_probs_IC' + \
         cyyyymmddhh+'_lead'+clead+'h.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)

    istat = 0
    return istat

# ====================================================================

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

config_file_name = 'config_laptop.ini'
GRAFdatadir_conus_laptop, GRAFprobsdir_conus_laptop, \
    GRAF_plot_dir = read_config_file(config_file_name, \
    'DIRECTORIES')

istat_GRAF, precipitation_GRAF, lats, lons, ny, nx, latmin, latmax, \
    lonmin, lonmax, verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop)
    
istat_prob, raw_p0p25mm_prob, dl_p0p25mm_prob, raw_p1mm_prob, \
    dl_p1mm_prob, raw_p2p5mm_prob, dl_p2p5mm_prob, raw_p5mm_prob, \
    dl_p5mm_prob, raw_p10mm_prob, dl_p10mm_prob, raw_p25mm_prob, \
    dl_p25mm_prob, lat, lon = probability_read(clead, cyyyymmddhh, \
    GRAFprobsdir_conus_laptop)

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
        lowprob_dl = dl_p0p25mm_prob
        highprob_dl = dl_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'
    elif cyyyymmddhh == '2025120812':
        llcrnrlon = -125
        llcrnrlat = 33.5
        urcrnrlon = -103
        urcrnrlat = 53.
        lowprob_raw = raw_p2p5mm_prob
        highprob_raw = raw_p5mm_prob
        lowprob_dl = dl_p2p5mm_prob
        highprob_dl = dl_p5mm_prob
        ltitle = 'Prob > 2.5 mm/h'
        htitle = 'Prob > 5 mm/h'
    else:
        llcrnrlon = -112
        llcrnrlat = 33.
        urcrnrlon = -90
        urcrnrlat = 48.
        lowprob_raw = raw_p0p25mm_prob
        highprob_raw = raw_p2p5mm_prob
        lowprob_dl = dl_p0p25mm_prob
        highprob_dl = dl_p2p5mm_prob
        ltitle = 'Prob > 0.25 mm/h'
        htitle = 'Prob > 2.5 mm/h'

    # Plotting, first CONUS scale and then zoomed in.
    
    istat = plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_dl, \
        highprob_dl, lowprob_raw, ltitle, htitle, GRAF_plot_dir)
        
    istat = plot_GRAF_small(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, \
        cyyyymmddhh, clead, precipitation_GRAF, lowprob_dl, \
        highprob_dl, lowprob_raw, ltitle, htitle, GRAF_plot_dir)
                    
else:
    print ('GRAF forecast or probability data not found.')
