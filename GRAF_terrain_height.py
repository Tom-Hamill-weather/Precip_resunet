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

def read_gribdata(gribfilename):

    """ read grib data"""

    import os
    import pygrib

    istat = -1
    fexist_grib = False
    fexist_grib = os.path.exists(gribfilename)
    if fexist_grib:
        print ('   reading ',gribfilename, fexist_grib)
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select()[0]
            lats, lons = grb.latlons()
            terrain_height = grb.values
            istat = 0
            fcstfile.close()
        except IOError:
            print ('   IOError in read_gribdata reading ', \
                gribfilename)
            istat = -1
        except ValueError:
            print ('   ValueError in read_gribdata reading ', \
                gribfilename)
            istat = -1
        except RuntimeError:
            print ('   RuntimeError in read_gribdata reading ', \
                gribfilename)
            istat = -1
    else:
        print ('grib file does not exist.')
        sys.exit()
        
    return istat, terrain_height, lats, lons

# ---------------------------------------------------------

def terrain_slopes(data, lons, lats, ny, nx):

    mpdlat = 111000. # meters per degree longitude
    dterrain_dlon = np.zeros((ny, nx), dtype=float)
    dterrain_dlat = np.zeros((ny, nx), dtype=float)

    # --- interior points
    
    for jy in range(1,ny-1):
        for ix in range(1, nx-1):
            coslat = np.cos(lats[jy,ix]*3.14159/180.)
            mpdlon = mpdlat*coslat # longitude
            dy = ((lats[jy+1,ix]-lats[jy-1,ix])*mpdlat)/2. # in meters
            dx = ((lons[jy,ix+1]-lons[jy,ix-1])*mpdlon)/2.
            dterrain_dlon[jy,ix] = (data[jy,ix+1]-data[jy,ix-1])/(2.*dy)
            dterrain_dlat[jy,ix] = (data[jy+1,ix]-data[jy-1,ix])/(2.*dy)
            
    # --- western boundary
    
    for jy in range(1,ny-1):
        ix = 0
        coslat = np.cos(lats[jy,ix]*3.14159/180.)
        mpdlon = mpdlat*coslat # longitude
        dy = ((lats[jy+1,ix]-lats[jy-1,ix])*mpdlat)/2.
        dx = ((lons[jy,ix+1]-lons[jy,ix])*mpdlon)
        dterrain_dlon[jy,ix] = (data[jy,ix+1]-data[jy,ix])/dx
        dterrain_dlat[jy,ix] = (data[jy+1,ix]-data[jy-1,ix])/(2.*dy)
        
    # --- eastern boundary
    
    for jy in range(1,ny-1):
        ix = nx-1
        coslat = np.cos(lats[jy,ix]*3.14159/180.)
        mpdlon = mpdlat*coslat # longitude
        dy = ((lats[jy+1,ix]-lats[jy-1,ix])*mpdlat)/2.
        dx = ((lons[jy,ix]-lons[jy,ix-1])*mpdlon)
        dterrain_dlon[jy,ix] = (data[jy,ix]-data[jy,ix-1])/dx
        dterrain_dlat[jy,ix] = (data[jy+1,ix]-data[jy-1,ix])/(2.*dy)
        
    # --- southern boundary
    
    for ix in range(1,nx-1):
        jy = 0
        coslat = np.cos(lats[jy,ix]*3.14159/180.)
        mpdlon = mpdlat*coslat # longitude
        dy = ((lats[1,ix]-lats[0,ix])*mpdlat)
        dx = ((lons[0,ix+1]-lons[0,ix-1])*mpdlon)/2.
        dterrain_dlon[jy,ix] = (data[jy,ix+1]-data[jy,ix-1])/(2.*dx)
        dterrain_dlat[jy,ix] = (data[jy+1,ix]-data[jy,ix])/dy
        
    # --- northern boundary
    
    for ix in range(1,nx-1):
        jy = ny-1
        coslat = np.cos(lats[jy,ix]*3.14159/180.)
        mpdlon = mpdlat*coslat # longitude
        dy = ((lats[jy,ix]-lats[jy-1,ix])*mpdlat)
        dx = ((lons[jy,ix+1]-lons[jy,ix-1])*mpdlon)/2.
        dterrain_dlon[jy,ix] = (data[jy,ix+1]-data[jy,ix-1])/(2.*dx)
        dterrain_dlat[jy,ix] = (data[jy,ix]-data[jy-1,ix])/dy
        
    # --- corners
    
    dterrain_dlon[0,0] = (dterrain_dlon[0,1] + dterrain_dlon[1,0] + dterrain_dlon[1,1])/3.
    dterrain_dlat[0,0] = (dterrain_dlat[0,1] + dterrain_dlat[1,0] + dterrain_dlat[1,1])/3.

    dterrain_dlon[-1,0] = (dterrain_dlon[-1,1] + dterrain_dlon[-2,0] + dterrain_dlon[-2,1])/3.
    dterrain_dlat[-1,0] = (dterrain_dlat[-1,1] + dterrain_dlat[-2,0] + dterrain_dlat[-2,1])/3.

    dterrain_dlon[-1,-1] = (dterrain_dlon[-2,-2] + dterrain_dlon[-2,-1] + dterrain_dlon[-1,-2])/3.
    dterrain_dlat[-1,-1] = (dterrain_dlat[-2,-2] + dterrain_dlat[-2,-1] + dterrain_dlat[-1,-2])/3.

    dterrain_dlon[0,-1] = (dterrain_dlon[1,-2] + dterrain_dlon[1,-1] + dterrain_dlon[0,-2])/3.
    dterrain_dlat[0,-1] = (dterrain_dlat[1,-2] + dterrain_dlat[1,-1] + dterrain_dlat[0,-2])/3.

    return dterrain_dlon, dterrain_dlat

# ---------------------------------------------------------------------

def write_to_netCDF(outfile, ny, nx, nsigma, lons, lats, \
        sigmas, terrain_height, terrain_height_smoothed, \
        terrain_height_local_difference, \
        dterrain_dlon, dterrain_dlat, dterrain_dlon_smoothed, \
        dterrain_dlat_smoothed, dterrain_dlon_difference, \
        dterrain_dlat_difference, terrain_height_smoothed_multisigma):
        
    # ---- set up netCDF file particulars

    nc = Dataset(outfile,'w',format='NETCDF4_CLASSIC')
    print ('writing to ',outfile)
    
    ny = nc.createDimension('ny',ny)
    nx = nc.createDimension('nx',nx)
    nsigmas = nc.createDimension('nsigma',nsigma)

    lons_out = nc.createVariable('lons','f4',('ny','nx',))
    lons_out.long_name = "longitude" 
    lons_out.units = "degrees_east" 

    lats_out = nc.createVariable('lats','f4',('ny','nx',))
    lats_out.long_name = "latitude" 
    lats_out.units = "degrees_north" 
    
    sigmas_out = nc.createVariable('sigmas','f4',('nsigma',))
    sigmas_out.long_name = "smoothing length scale in GRAF grid pts"
    sigmas_out.units = "number of grid points"
    
    terrain_height_out = nc.createVariable(\
        'terrain_height','f4',('ny','nx',), \
        zlib=True,least_significant_digit=2)
    terrain_height_out.units = "m" 
    terrain_height_out.long_name = "Terrain height for GRAF CONUS (m)"
    terrain_height_out.valid_range = [-90,13000] 
    terrain_height_out.missing_value = -99.99
    
    terrain_height_smoothed_out = nc.createVariable(\
        'terrain_height_smoothed','f4',('ny','nx',), \
        zlib=True,least_significant_digit=2)
    terrain_height_smoothed_out.units = "m" 
    terrain_height_smoothed_out.long_name = \
        "Smoothed terrain height for GRAF CONUS, "+\
        "15 grid point sigma Gaussian convolve (m)"
    terrain_height_smoothed_out.valid_range = [-90,13000] 
    terrain_height_smoothed_out.missing_value = -99.99
    
    terrain_height_local_difference_out = nc.createVariable(\
        'terrain_height_local_difference','f4',('ny','nx',), \
        zlib=True,least_significant_digit=2)
    terrain_height_local_difference_out.units = "m" 
    terrain_height_local_difference_out.long_name = \
        "Raw minus smoothed terrain height difference for GRAF\n"+ \
        "CONUS, 15 grid point sigma Gaussian convolve for smooth"
    terrain_height_local_difference_out.valid_range = [-3000,3000] 
    terrain_height_local_difference_out.missing_value = -9999.99

    dterrain_dlon_out = nc.createVariable(\
        'dterrain_dlon','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlon_out.units = "m/m" 
    dterrain_dlon_out.long_name = \
        "change in terrain height with longitude per meter horizontal"
    dterrain_dlon_out.valid_range = [-100,100] 
    dterrain_dlon_out.missing_value = -999.99

    dterrain_dlat_out = nc.createVariable(\
        'dterrain_dlat','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlat_out.units = "m/m" 
    dterrain_dlat_out.long_name = \
        "change in terrain height with latitude per meter horizontal"
    dterrain_dlat_out.valid_range = [-101,100] 
    dterrain_dlat_out.missing_value = -999.99

    dterrain_dlon_smoothed_out = nc.createVariable(\
        'dterrain_dlon_smoothed','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlon_smoothed_out.units = "m" 
    dterrain_dlon_smoothed_out.long_name = \
        "change in (smoothed) terrain height\n"+\
        "with longitude per meter horizontal"
    dterrain_dlon_smoothed_out.valid_range = [-100,100] 
    dterrain_dlon_smoothed_out.missing_value = -999.99

    dterrain_dlat_smoothed_out = nc.createVariable(\
        'dterrain_dlat_smoothed','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlat_smoothed_out.units = "m" 
    dterrain_dlat_smoothed_out.long_name = \
        "change in (smoothed) terrain height\n"+\
        "with latitude per meter horizontal"
    dterrain_dlat_smoothed_out.valid_range = [-100,100] 
    dterrain_dlat_smoothed_out.missing_value = -999.99
    
    
    dterrain_dlon_difference_out = nc.createVariable(\
        'dterrain_dlon_difference','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlon_difference_out.units = "m" 
    dterrain_dlon_difference_out.long_name = \
        "change in (raw-smoothed) terrain height\n"+\
        "with longitude per meter horizontal"
    dterrain_dlon_difference_out.valid_range = [-100,100] 
    dterrain_dlon_difference_out.missing_value = -999.99

    dterrain_dlat_difference_out = nc.createVariable(\
        'dterrain_dlat_difference','f4',('ny','nx',), \
        zlib=True,least_significant_digit=5)
    dterrain_dlat_difference_out.units = "m" 
    dterrain_dlat_difference_out.long_name = \
        "change in (raw-smoothed) terrain height\n"+\
        "with latitude per meter horizontal"
    dterrain_dlat_difference_out.valid_range = [-100,100] 
    dterrain_dlat_difference_out.missing_value = -999.99
    
    terrain_height_smoothed_multisigma_out  = nc.createVariable(\
        'terrain_height_smoothed_multisigma','f4',('nsigma','ny','nx',), \
        zlib=True,least_significant_digit=5)
    terrain_height_smoothed_multisigma_out.units = "m" 
    terrain_height_smoothed_multisigma_out.long_name = \
        "change in (raw-smoothed) terrain height\n"+\
        "with latitude per meter horizontal"
    terrain_height_smoothed_multisigma_out.valid_range = \
        [-90.,13000.] 
    terrain_height_smoothed_multisigma_out.missing_value = \
        -999.99
    
    
    nc.title = "GRAF CONUS terrain information; terrain height" + \
        "smoothed, deviations, gradients"
    nc.history = "Created 1 Nov 2024 by Tom Hamill" 
    nc.institution = "The Weather Company"
    nc.platform = "The Weather Company GRAF" 
    
    # --- write
    
    lons_out[:] = lons[:,:]
    lats_out[:] = lats[:,:]
    sigmas_out[:] = sigmas[:]
    terrain_height_out[:] = terrain_height[:,:]
    terrain_height_smoothed_out[:] = terrain_height_smoothed[:,:]
    terrain_height_local_difference_out[:] = \
        terrain_height_local_difference[:,:]
    dterrain_dlon_out[:] = dterrain_dlon[:,:]
    dterrain_dlat_out[:] = dterrain_dlat[:,:]
    dterrain_dlon_smoothed_out[:] = dterrain_dlon_smoothed[:,:]
    dterrain_dlat_smoothed_out[:] = dterrain_dlat_smoothed[:,:]
    dterrain_dlon_difference_out[:] = dterrain_dlon_difference[:,:]
    dterrain_dlat_difference_out[:] = dterrain_dlat_difference[:,:]
    terrain_height_smoothed_multisigma_out[:] = \
        terrain_height_smoothed_multisigma[:,:,:]
    nc.close()

# =======================================================

config_file = "../ini/config_hdo.ini"
directory_object_name = 'DIRECTORIES'
GRAFdatadir_fixedfield = \
    read_config_file(config_file, directory_object_name)

# --- read terrain file.

gribfilename = 'GRAF_terrain_height.grb2'
istat, terrain_height, lats, lons = read_gribdata(gribfilename)
ny, nx = np.shape(terrain_height)

# --- smooth with a standard deviation of 15 grid points

print ('smoothing 15 gp')
terrain_height_smoothed = ndimage.gaussian_filter(terrain_height, 15.0)

# --- smooth with many sigmas

#sigmas = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, \
#   40.0, 50.0, 60.0, 75.0, 120.0, 150.0] 
#nsigma = len(sigmas)

sigmas = [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0] 
nsigma = len(sigmas)

terrain_height_smoothed_multisigma = np.zeros((nsigma, ny, nx))
for isigma, sigma in enumerate(sigmas):
    print ('smoothing with ',sigma,' gp')
    terrain_height_smoothed_multisigma[isigma,:,:] = \
         ndimage.gaussian_filter(terrain_height, sigma)

# --- difference field.

print ('difference field...')
terrain_height_local_difference = \
    terrain_height - terrain_height_smoothed

# --- get the terrain slopes, E-W and N-S direction

print ('calculate terrain horizontal slopes')
dterrain_dlon, dterrain_dlat = \
    terrain_slopes(terrain_height, lons, lats, ny, nx)
dterrain_dlon_smoothed, dterrain_dlat_smoothed = \
    terrain_slopes(terrain_height_smoothed, lons, lats, ny, nx)
dterrain_dlon_difference, dterrain_dlat_difference = \
    terrain_slopes(terrain_height_smoothed, lons, lats, ny, nx)
    
print ('dterrain_dlon[ny//2,0:10] = ',dterrain_dlon[ny//2,0:10] ) 
print ('dterrain_dlon[ny//2,-10:] = ',dterrain_dlon[ny//2,-10:] ) 
print ('dterrain_dlat[0:10,nx//2] = ',dterrain_dlat[0:10,nx//2] ) 
print ('dterrain_dlat[-10:,nx//2] = ',dterrain_dlat[-10:,nx//2] )    

print ('dterrain_dlon_smoothed[ny//2,0:10] = ',dterrain_dlon_smoothed[ny//2,0:10] ) 
print ('dterrain_dlon_smoothed[ny//2,-10:] = ',dterrain_dlon_smoothed[ny//2,-10:] ) 
print ('dterrain_dlat_smoothed[0:10,nx//2] = ',dterrain_dlat_smoothed[0:10,nx//2] ) 
print ('dterrain_dlat_smoothed[-10:,nx//2] = ',dterrain_dlat_smoothed[-10:,nx//2] )     

print ('dterrain_dlon_difference[ny//2,0:10] = ',dterrain_dlon_difference[ny//2,0:10] ) 
print ('dterrain_dlon_difference[ny//2,-10:] = ',dterrain_dlon_difference[ny//2,-10:] ) 
print ('dterrain_dlat_difference[0:10,nx//2] = ',dterrain_dlat_difference[0:10,nx//2] ) 
print ('dterrain_dlat_difference[-10:,nx//2] = ',dterrain_dlat_difference[-10:,nx//2] )     

# --- save to netCDF file

outfile = GRAFdatadir_fixedfield + 'GRAF_CONUS_terrain_info.nc'
istat = write_to_netCDF(outfile, ny, nx, nsigma, lons, lats, \
    sigmas, terrain_height, terrain_height_smoothed, \
    terrain_height_local_difference, \
    dterrain_dlon, dterrain_dlat, dterrain_dlon_smoothed, \
    dterrain_dlat_smoothed, dterrain_dlon_difference, \
    dterrain_dlat_difference, terrain_height_smoothed_multisigma)

# ---- make plots if desired.

plotit = False
if plotit == True:
    print ('setting up plotting.')
    colorst = ['White','#E4FFFF','#C4E8FF',\
        '#8FB3FF','#D8F9D8','#A6ECA6','#42F742',\
        'Yellow','Gold','Orange','#FCD5D9','#F6A3AE',\
        '#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    latb = 20.0
    late = 53.0
    lonb = -123.0
    lone = -60.0
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

    for itype in range(3):
    
        if itype == 0:
            data_to_plot = terrain_height
            title = 'GRAF CONUS terrain height'
            plot_title = 'GRAF_CONUS_terrain_height.png'
            colors = colorst
            cmap_use = cmap
            norm_use = norm
            clevs = clevels
            legend_title = 'terrain height (m)'
        elif itype == 1:
            data_to_plot = terrain_height_smoothed
            title = r'Smoothed GRAF terrain height, $\sigma$=15 grid points'
            plot_title = 'GRAF_CONUS_terrain_height_smoothed.png'
            colors = colorst
            cmap_use = cmap
            norm_use = norm
            clevs = clevels
            legend_title = 'smoothed terrain height (m)'
        else:
            data_to_plot = terrain_height_local_difference
            title = 'Difference, GRAF CONUS terrain height\nminus smoothed'
            plot_title = 'GRAF_CONUS_terrain_height_difference.png'
            colors = colors_red_to_blue
            cmap_use = cmap_rb
            norm_use = norm_rb
            clevs = clevels_difference
            legend_title = 'difference in terrain height (m)'
        
        # --- plot gridded 

        fig = plt.figure(figsize=(6.,6.))
        axloc = [0.02,0.1,0.96,0.79]
        ax1 = fig.add_axes(axloc)
        ax1.set_title(title, fontsize=16,color='Black')
        CS2 = m.pcolormesh(x, y, data_to_plot, cmap=cmap_use, \
            shading='nearest', norm=norm_use)
        m.drawcoastlines(linewidth=0.6,color='Gray')
        m.drawcountries(linewidth=0.4,color='Gray')
        m.drawstates(linewidth=0.2,color='LightGray')

        # ---- use axes_grid toolkit to make colorbar axes.

        cax = fig.add_axes([0.06,0.08,0.88,0.02])
        cb = plt.colorbar(CS2,orientation='horizontal',cax=cax,\
            drawedges=True,ticks=clevs,format='%g')
        cb.ax.tick_params(labelsize=9)
        cb.set_label(legend_title,fontsize=11)

        fig.savefig(plot_title, dpi=300)
        print ('saving plot to file = ',plot_title)
        plt.close()


