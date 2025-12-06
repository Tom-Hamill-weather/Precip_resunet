"""
python resunet_inference.py cyyyymmddhh clead

e.g.,

python resunet_inference.py 2025112100 12

"""

from configparser import ConfigParser
import numpy as np
import os, sys
import glob
import re
from dateutils import daterange, dateshift
import torch
from mpl_toolkits.basemap import Basemap
# Import ResUNet from your training script
from pytorch_train_resunet import ResUNet
print ('imported ResUNet')
from netCDF4 import Dataset
import scipy.stats as stats
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import _pickle as cPickle
np.set_printoptions(precision=2, suppress=True)
import warnings
warnings.filterwarnings("ignore")

# --- Set device for inference ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (DEVICE)

TRAIN_DIR = 'trainings/'

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    """ read appropriate information from the config file """
    from configparser import ConfigParser

    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)

    directory = config_object[directory_object_name]
    GRAFdatadir_conus_laptop = directory["GRAFdatadir_conus_laptop"]
  
    return GRAFdatadir_conus_laptop

# ---------------------------------------------------------------

def define_manhattan(N):
    """ define manhattan weighting function, like a pyramid. """
    ilocs = np.arange(N)
    jlocs = np.copy(ilocs)
    manhattan = np.zeros((N,N), dtype=float)
    for j in jlocs:
        wj = np.max([0.0, 1. - 2.*np.abs(j+0.5-N/2)/N])
        for i in ilocs:
            wi = np.max([0.0, 1. - 2.*np.abs(i+0.5-N/2)/N])
            manhattan[j,i] = 0.5*wj*wi 
    return manhattan

# ---------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    """ read grib data"""
    import os
    import pygrib

    istat = -1
    fexist_grib = os.path.exists(gribfilename)
    if fexist_grib:
        try:
            fcstfile = pygrib.open(gribfilename)
            grb = fcstfile.select(endStep = endStep)[0]
            lats, lons = grb.latlons()
            precipitation = grb.values
            lon_0 = grb.projparams["lon_0"]
            lat_0 = grb.projparams["lat_0"]
            lat_1 = grb.projparams["lat_1"]
            lat_2 = grb.projparams["lat_2"]
            istat = 0
            fcstfile.close()
        except IOError:
            print ('   IOError in read_gribdata reading ', gribfilename)
            istat = -1
        except ValueError:
            print ('   ValueError in read_gribdata reading ', gribfilename)
            istat = -1
        except RuntimeError:
            print ('   RuntimeError in read_gribdata reading ', gribfilename)
            istat = -1
    else:
        print ('grib file does not exist.')

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2

# ---------------------------------------------------------------

def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_laptop):
    """ Read the stored GRAF data. """
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
        latmax = np.max(lats)
        latmin = np.min(lats)
        lonmax = np.max(lons)
        lonmin = np.min(lons)

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

# ---------------------------------------------------------------

def read_terrain_characteristics(infile):
    """ Read terrain height and slope. """
    fexist1 = os.path.exists(infile)
    if fexist1 == True:
        nc = Dataset(infile, 'r')
        t_diff = nc.variables['terrain_height_local_difference'][:,:]
        dt_dlon = nc.variables['dterrain_dlon_smoothed'][:,:]
        dt_dlat = nc.variables['dterrain_dlat_smoothed'][:,:]
        nc.close()
    else:
        print ('  Could not find desired terrain file.')
        print ('  ',infile)
        sys.exit()

    return t_diff, dt_dlon, dt_dlat

# ---------------------------------------------------------------

def generate_features(nchannels, date, clead, \
        ny, nx, precipitation_GRAF, t_diff, dt_dlon, \
        dt_dlat, verif_local_time, norm_stats=None):
    
    """
    Generate the 7-channel input tensor expected by the new ResUNet.
    Applies normalization if stats are provided.
    Structure: [graf, terrain, diff, dlon, dlat, time, graf]
    """

    # --- Helper for normalization ---
    def normalize(data, idx):
        if norm_stats is None:
            return data
        
        # Ensure we are using float scalars (MPS compatibility)
        vmin = float(norm_stats['min'][idx])
        vmax = float(norm_stats['max'][idx])
        denom = vmax - vmin
        if denom == 0: denom = 1e-8
        
        return (data - vmin) / denom

    Xpredict_all = np.zeros((1,nchannels,ny,nx), dtype=float)
    
    # Channel 0: Forecast Precipitation
    Xpredict_all[0,0,:,:] = normalize(precipitation_GRAF[:,:], 0)
    
    # Channel 1: Terrain
    Xpredict_all[0,1,:,:] = normalize(t_diff[:,:], 1)
    
    # Channel 2: Difference
    Xpredict_all[0,2,:,:] = normalize(t_diff[:,:], 2)
    
    # Channel 3: dlon (Slope)
    Xpredict_all[0,3,:,:] = normalize(dt_dlon[:,:], 3)
    
    # Channel 4: dlat (Slope)
    Xpredict_all[0,4,:,:] = normalize(dt_dlat[:,:], 4)
    
    # Channel 5: Time
    Xpredict_all[0,5,:,:] = normalize(verif_local_time[:,:], 5)
    
    # Channel 6: Forecast Precipitation (Duplicate)
    Xpredict_all[0,6,:,:] = normalize(precipitation_GRAF[:,:], 0)

    return Xpredict_all, precipitation_GRAF

# ---------------------------------------------------------------

def read_pytorch(cyyyymmddhh, clead, num_classes): 
    # --- Logic to find the most appropriate checkpoint ---
    
    # 1. Glob ALL files that match the lead time
    #    Pattern: resunet_ordinal_YYYYMMDDHH_LLh_epoch_E.pth
    glob_pattern = os.path.join(TRAIN_DIR, f"resunet_ordinal_*_{clead}h_epoch_*.pth")
    files = glob.glob(glob_pattern)
    
    if not files:
        print(f"   No training files found in {TRAIN_DIR} matching pattern: {glob_pattern}")
        return None, None

    valid_candidates = []
    inference_date_int = int(cyyyymmddhh)
    
    # 2. Filter and Sort
    for fpath in files:
        basename = os.path.basename(fpath)
        # Regex matches: Date (group 1), Lead (group 2), Epoch (group 3)
        match = re.search(r"resunet_ordinal_(\d{10})_(\d+)h_epoch_(\d+)\.pth", basename)
        
        if match:
            fdate = int(match.group(1))
            flead = match.group(2)
            fepoch = int(match.group(3))
            
            # Ensure we only use trainings that happened ON or BEFORE inference date
            if fdate <= inference_date_int:
                valid_candidates.append({
                    'path': fpath,
                    'date': fdate,
                    'epoch': fepoch
                })
    
    if not valid_candidates:
        print("   No valid training checkpoints found with date <= inference date.")
        return None, None
    
    # 3. Sort logic: 
    #    Primary key: Date (Descending) -> Newest training first
    #    Secondary key: Epoch (Descending) -> Highest epoch first
    valid_candidates.sort(key=lambda x: (x['date'], x['epoch']), reverse=True)
    
    best_candidate = valid_candidates[0]
    best_file = best_candidate['path']
    
    print(f"   Loading best checkpoint: {best_file}")
    print(f"   (Training Date: {best_candidate['date']}, Epoch: {best_candidate['epoch']})")
    
    # 4. Load Model
    model = ResUNet(in_channels=7, num_classes=num_classes)
    normalization_stats = None
    
    try:
        # Load checkpoint (weights_only=False required for numpy stats)
        checkpoint = torch.load(best_file, map_location=DEVICE, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            normalization_stats = checkpoint.get('normalization_stats', None)
        else:
            model.load_state_dict(checkpoint)
            print("   Warning: Legacy checkpoint format (no stats found).")
            
        model.to(DEVICE)
        model.eval()
        print('   Model loaded successfully.')
        if normalization_stats:
            print('   Normalization statistics loaded.')
        
        return model, normalization_stats
        
    except Exception as e:
        print(f"   Error loading model: {e}")
        return None, None
        
# ------------------------------------------------------------------- 

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)
    
# ------------------------------------------------------------------- 

def plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
        cyyyymmddhh, clead, precipitation_GRAF, prob_POP, \
        prob_5mm):

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
    clevs_prob = [0, 0.05, .1, .2, .3, .4, .5, .6, .7, 0.8, \
        .9, .95, .97, 1.]

    clead_minus = str(int(clead)-1)
    fig = plt.figure(figsize=(9,11.))
    plt.suptitle(clead_minus+' to '+clead+\
        '-h GRAF Res-U-Net forecasts, valid '+\
        datestring,fontsize=17)

    # --- panel 1: GRAF deterministic amount
    axloc = [0.01,0.58,0.48,0.32]
    ax1 = fig.add_axes(axloc)
    title = '(a) GRAF hourly precipitation amount'
    ax1.set_title(title, fontsize=13,color='Black')
    CS2 = m.contourf(x, y, precipitation_GRAF, clevs, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.55,0.44,0.02])
    cb = plt.colorbar(CS2,orientation='horizontal',cax=cax,\
        drawedges=True,ticks=clevs,format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Precipitation amount (mm)', fontsize=10)

    # --- panel 2: GRAF POP
    axloc = [0.51,0.58,0.48,0.32]
    ax1 = fig.add_axes(axloc)
    title = '(b) U-Net GRAF-based probability > 0.254 mm/h'
    ax1.set_title(title, fontsize=13,color='Black')
    CS2 = m.contourf(x, y, prob_POP, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.53,0.55,0.44,0.02])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # --- panel 3: 5 mm prob
    axloc = [0.01,0.12,0.48,0.32]
    ax1 = fig.add_axes(axloc)
    title = '(c) U-Net GRAF-based probability > 5 mm/h'
    ax1.set_title(title, fontsize=13,color='Black')
    CS2 = m.contourf(x, y, prob_5mm, clevs_prob, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')
    cax = fig.add_axes([0.03,0.09,0.44,0.02])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs_prob, format='%g')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Probability', fontsize=10)

    # ---- set plot title
    plot_title = 'ResUnet_GRAF_probs_IC' + \
         cyyyymmddhh+'_lead'+clead+'h.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)
    print ('Done!')
    istat = 0
    return istat

# ====================================================================
# ---- Main Execution
# ====================================================================

if len(sys.argv) < 3:
    print("Usage: python resunet_inference.py <YYYYMMDDHH> <lead>")
    sys.exit(1)

cyyyymmddhh = sys.argv[1]
clead = sys.argv[2]

N = 64
ny = 1308 # GRAF y grid points
nx = 1524 # GRAF x grid points
nchannels = 7 # Updated to 7 to match training

# --- Define Classes based on training script logic ---
THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
THRESHOLDS.append(200.0)
NUM_CLASSES = len(THRESHOLDS)-1 # Should be 101

# --- read GRAF directory names from config file.
config_file_name = 'config_laptop.ini'
directory_object_name = 'DIRECTORIES'
GRAFdatadir_conus_laptop = \
    read_config_file(config_file_name, directory_object_name)

# --- define the "Manhattan" weighting function
manhattan = define_manhattan(N)

# --- read the GRAF 1-h precipitation forecast data.
print ('read GRAF')
istat_GRAF, precipitation_GRAF, lats, lons, \
    ny, nx, latmin, latmax, lonmin, lonmax, \
    verif_local_time, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_precip_read(clead, cyyyymmddhh, \
    GRAFdatadir_conus_laptop)

if istat_GRAF == 0:
    print ('  max(precipitation_GRAF) = ', np.max(precipitation_GRAF))
    ny_GRAF, nx_GRAF = np.shape(lats)
    infile = 'GRAF_CONUS_terrain_info.nc'
    print ('read terrain info')
    t_diff, dt_dlon, dt_dlat = read_terrain_characteristics(infile)

    # --- read pytorch model
    # Now returns both model AND normalization stats
    model, norm_stats = read_pytorch(cyyyymmddhh, clead, NUM_CLASSES)
    
    if model:
        model = model.float()

        # --- Generate full-domain arrays of GRAF and terrain feature data.
        # Now accepts norm_stats to normalize inputs consistently with training
        Xpredict_all, precip_save = generate_features(nchannels,\
            cyyyymmddhh, clead, ny, nx, precipitation_GRAF, t_diff, \
            dt_dlon, dt_dlat, verif_local_time, norm_stats=norm_stats)

        # Storage for accumulated probabilities
        precip_fcst_prob_all = np.zeros((NUM_CLASSES,ny,nx), dtype=float)
        sumweights_all = np.zeros((ny,nx), dtype=float)

        # --- Define centers of inference patches
        jcenter1 = range(N//2, ny-N//2+1, N//2)
        icenter1 = range(N//2, nx-N//2+1, N//2)
        jcenter2 = range(N//2 + N//4, ny-3*N//4, N//2)
        icenter2 = range(N//2 + N//4, nx-3*N//4, N//2)
        
        # Helper to process patches
        def process_patches(jcenters, icenters, description):
            print(f'{description}')
            for j in jcenters:
                jmin = j-N//2
                jmax = j+N//2
                for i in icenters:
                    imin = i-N//2
                    imax = i+N//2
                    
                    # --- Extract patch and run inference
                    Xpatch = Xpredict_all[:,:,jmin:jmax,imin:imax]
                    input_tensor = torch.from_numpy(Xpatch).float().to(DEVICE)
                    
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    # Convert Logits to Probabilities via Softmax
                    precipitation_probabilities = \
                        softmax(np.squeeze(output_tensor.cpu().numpy()))
                    
                    # --- Weighted Sum Accumulation
                    precip_fcst_prob_all[:, jmin:jmax, imin:imax] += \
                        precipitation_probabilities * manhattan[None, :, :]
                            
                    sumweights_all[jmin:jmax, imin:imax] += manhattan[:,:]

        # Run inference on both staggered grids
        process_patches(jcenter1, icenter1, 'inference on first set of patches')
        process_patches(jcenter2, icenter2, 'inference on second staggered set of patches')

        # ---- Normalize probabilities by weight sum
        valid_mask = sumweights_all > 0
        for icat in range(NUM_CLASSES):
            precip_fcst_prob_all[icat,:,:] = np.where(valid_mask, \
                precip_fcst_prob_all[icat,:,:] / (sumweights_all[:,:] + 1e-9), 0.0)

        # ---- Calculate Deterministic/Threshold Probabilities for Plotting
        
        # Class 0 covers 0.00 - 0.25mm. 
        # POP (Prob > 0.25mm) is 1.0 - Prob(Class 0)
        prob_POP = 1.0 - precip_fcst_prob_all[0,:,:]
        
        # 5mm threshold (index 20)
        idx_5mm = 20
        prob_5mm = np.sum(precip_fcst_prob_all[idx_5mm:,:,:], axis=0)

        # --- Plotting
        istat = plot_GRAF(lat_1, lat_2, lat_0, lon_0, lons, lats, \
            cyyyymmddhh, clead, precipitation_GRAF, prob_POP, prob_5mm)
            
    else:
        print("Model could not be loaded. Exiting.")

else:
    print ('GRAF forecast data not found.')


