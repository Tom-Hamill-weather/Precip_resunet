"""
python GRAF_terrain_height.py

This was the script to generate the GRAF terrain height data.
There should be a netCDF file output from this.  The input 
data is either on Tom's laptop or a directory of his at AWS.

Tom Hamill
Dec 2025

"""
import pygrib
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.colors as colors
from netCDF4 import Dataset
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("once")

# ---------------------------------------------------------

def read_config_file(config_file, directory_object_name):

    """ read appropriate information from the config file
        and return
    """
    from configparser import ConfigParser

    # ---- Read config.ini file

    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)

    # ---- Get the information from dictionary structure

    directory = config_object[directory_object_name]
    GRAFdatadir_fixedfield = \
        directory["GRAFdatadir_fixedfield"]
    return GRAFdatadir_fixedfield

# ---------------------------------------------------------



# ---------------------------------------------------------------------

def read_netCDF(infile):
        
    # ---- set up netCDF file particulars

    nc = Dataset(infile,'r')
    print ('reading', infile)    
    lons = nc.variables['lons'][:,:]
    lats = nc.variables['lats'][:,:]
    ny, nx = np.shape(lons)
    terrain_height_local_difference = \
        nc.variables['terrain_height_local_difference'][:,:]

    return ny, nx, lons, lats, terrain_height_local_difference

# =======================================================



# --- save to netCDF file

infile = 'GRAF_CONUS_terrain_info.nc'
ny, nx, lons, lats, terrain_height_local_difference = read_netCDF(infile)
# ---- make plots if desired.

plotit = True
if plotit == True:
    print ('setting up plotting.')
    colorst = ['White','#E4FFFF','#C4E8FF',\
        '#8FB3FF','#D8F9D8','#A6ECA6','#42F742',\
        'Yellow','Gold','Orange','#FCD5D9','#F6A3AE',\
        '#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    latb = 30 # 20.0
    late = 53.0
    lonb = -123.0
    lone = -100 # -60.0
    
    m = Basemap(rsphere=(6378137.00,6356752.3142),\
        resolution='l',area_thresh=1000.,projection='lcc',\
        lat_1=35.,lat_2=45,lat_0=40.,lon_0=-100., \
        llcrnrlon=lonb,llcrnrlat=latb,urcrnrlon=lone,\
        urcrnrlat=late)
    x, y = m(lons, lats)   
    colorst = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange',\
        '#FCD5D9','#F6A3AE','#f17484']
    colors_red_to_blue = ['DodgerBlue','#6db7ff','#92c9ff','#b0d8ff','#e8f4ff',\
        'White','#fff2f2','#ffbfbf','#ffa6a6','#ff8c8c','Red']
    #clevels = [0,5,10,20,30,40,50,60,70,80,90,95,97,100] 
    clevels = [-300,0,5,10,20,50,100,300,600,1000,1500,2000,2500,3000]
    clevels_difference = [-1000,-500,-300,-100,-50,-10,10,50,100,300,500,1000]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(\
        "", colorst, N=len(colorst))
    norm = colors.BoundaryNorm(boundaries=clevels, \
            ncolors=len(colorst), clip=True)
    cmap_rb = mpl.colors.LinearSegmentedColormap.from_list(\
        "", colors_red_to_blue, N=len(colors_red_to_blue))
    norm_rb = colors.BoundaryNorm(boundaries=clevels_difference, \
            ncolors=len(colors_red_to_blue), clip=True)

    data_to_plot = terrain_height_local_difference
    title = 'Difference, GRAF CONUS terrain height minus smoothed'
    plot_title = 'GRAF_CONUS_terrain_height_difference.png'
    colors = colors_red_to_blue
    cmap_use = cmap_rb
    norm_use = norm_rb
    clevs = clevels_difference
    legend_title = 'difference in terrain height (m)'
        
    # --- plot gridded 

    fig = plt.figure(figsize=(6.,7.))
    axloc = [0.02,0.12,0.96,0.8]
    ax1 = fig.add_axes(axloc)
    ax1.set_title(title, fontsize=14,color='Black')
    CS2 = m.pcolormesh(x, y, data_to_plot, cmap=cmap_use, \
        shading='nearest', norm=norm_use)
    m.drawcoastlines(linewidth=0.9,color='Gray')
    m.drawcountries(linewidth=0.6,color='Gray')
    m.drawstates(linewidth=0.6,color='Gray')

    # ---- use axes_grid toolkit to make colorbar axes.

    cax = fig.add_axes([0.06,0.08,0.88,0.02])
    cb = plt.colorbar(CS2,orientation='horizontal',cax=cax,\
        drawedges=True,ticks=clevs,format='%g')
    cb.ax.tick_params(labelsize=9)
    cb.set_label(legend_title,fontsize=11)

    fig.savefig(plot_title, dpi=300)
    print ('saving plot to file = ',plot_title)
    plt.close()


