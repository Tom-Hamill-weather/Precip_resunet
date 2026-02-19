"""
python plot_graf_mrms_samples.py filename sample_index

Intended to plot samples of patches from train, test, validation data.
"""
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --------------------------------------------------------------------
# Colormap Definitions (FIXED)
# --------------------------------------------------------------------

def get_precip_colormap():
    """
    Defines the specific colormap and norms for precipitation.
    Matches 13 colors to 13 bins (defined by 14 levels).
    """
    c_list = ['White','#E4FFFF','#C4E8FF','#8FB3FF','#D8F9D8',
              '#A6ECA6','#42F742','Yellow','Gold','Orange',
              '#FCD5D9','#F6A3AE','#f17484']
    
    cmap = colors.ListedColormap(c_list)
    
    # 14 boundaries create 13 bins. Matches the 13 colors above.
    levels = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 50]
    
    # Configure handling for values > 50 (use the last color)
    cmap.set_over('#f17484')
    
    norm = colors.BoundaryNorm(levels, cmap.N)
    
    return cmap, norm, levels

def get_terrain_colormap():
    """
    Defines the specific colormap and norms for terrain deviations.
    Matches 11 colors to 11 bins (defined by 12 levels).
    """
    c_list = ['DodgerBlue','#6db7ff','#92c9ff','#b0d8ff','#e8f4ff',
              'White','#fff2f2','#ffbfbf','#ffa6a6','#ff8c8c','Red']
    
    cmap = colors.ListedColormap(c_list)
    
    # 12 boundaries create 11 bins. Matches the 11 colors above.
    levels = [-1000,-500,-300,-100,-50,-10,10,50,100,300,500,1000]
    
    # Configure handling for values outside the range
    cmap.set_under('DodgerBlue') # Use first color for < -1000
    cmap.set_over('Red')         # Use last color for > 1000
    
    norm = colors.BoundaryNorm(levels, cmap.N)
    
    return cmap, norm, levels

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------

def load_sequential_pickle(filename):
    """
    Reads the file created by 'save_patched_GRAF_MRMS_gemini.py'.
    The format is multiple numpy arrays dumped sequentially.
    """
    keys_order = ['GRAF', 'MRMS', 'MRMS_qual', 'terdiff_x_GRAF', 
        'terrain_diff', 'dt_dlon', 'dt_dlat']
    
    data = {}
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    print(f"Reading file: {filename}")
    
    with open(filename, 'rb') as f:
        try:
            for key in keys_order:
                data[key] = pickle.load(f)
        except EOFError:
            print(f"Warning: Reached end of file early. Missing keys after {key}.")
            
    return data

# ---------------------------------------------------------------------------------
# Main Plotting Logic
# ---------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_graf_mrms_samples.py <filename> <sample_index>")
        print("Example: python plot_graf_mrms_samples.py "
              "../resnet_data/trainings/GRAF_Unet_data_train_2025120100_12h.cPick 12")
        sys.exit(1)

    filename = sys.argv[1]
    
    try:
        sample_idx = int(sys.argv[2])
    except ValueError:
        print("Error: sample_index must be an integer.")
        sys.exit(1)

    # 1. Load Data
    data_store = load_sequential_pickle(filename)
    
    # Validate we have the necessary keys (Added MRMS_qual)
    required_keys = ['GRAF', 'terrain_diff', 'MRMS', 'MRMS_qual']
    for k in required_keys:
        if k not in data_store:
            print(f"Error: Key '{k}' missing from pickle file.")
            sys.exit(1)
            
    # 2. Extract specific sample
    total_samples = data_store['GRAF'].shape[0]
    
    if sample_idx >= total_samples:
        print(f"Error: Index {sample_idx} out of bounds. "
              f"File contains {total_samples} samples.")
        sys.exit(1)

    # Feature 1: Model Forecast (GRAF)
    precip_fcst = data_store['GRAF'][sample_idx]
    
    # Feature 2: Terrain Deviations (terrain_diff)
    terr_dev = data_store['terrain_diff'][sample_idx]
    
    # Target: Analyzed Precip (MRMS)
    precip_anal = data_store['MRMS'][sample_idx]
    
    # Quality: MRMS Data Quality
    quality_anal = data_store['MRMS_qual'][sample_idx]

    # 3. Set up Plotting
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    cmap_p, norm_p, levs_p = get_precip_colormap()
    cmap_t, norm_t, levs_t = get_terrain_colormap()

    # --- Panel 1: GRAF Forecast ---
    
    ax1 = axes[0]
    pcm1 = ax1.pcolormesh(precip_fcst, cmap=cmap_p, \
        norm=norm_p, shading='nearest')
    ax1.set_title(f"Sample {sample_idx}: GRAF Forecast (Feature 1)", fontsize=15)
    ax1.invert_yaxis() 
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical', shrink=0.9, 
        ticks=levs_p, extend='max')
    cb1.set_label('mm')

    # --- Panel 2: Terrain Deviations ---
    
    ax2 = axes[1]
    pcm2 = ax2.pcolormesh(terr_dev, cmap=cmap_t, \
        norm=norm_t, shading='nearest')
    ax2.set_title(f"Sample {sample_idx}: Terrain Deviation (Feature 2)", fontsize=15)
    ax2.invert_yaxis()
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation='vertical', shrink=0.9, 
                       ticks=levs_t, extend='both')
    cb2.set_label('meters')

    # --- Panel 3: MRMS Analysis (with Quality Mask) ---
    
    ax3 = axes[2]
    
    # 3a. Plot precipitation first
    pcm3 = ax3.pcolormesh(precip_anal, cmap=cmap_p, \
        norm=norm_p, shading='nearest')
    
    # 3b. Create and plot quality mask
    # We want to mask (hide) good data (quality >= 0.1) so we can see the precip.
    # We leave bad data (quality < 0.1) unmasked to plot the gray overlay.
    bad_data_mask = np.ma.masked_where(quality_anal >= 0.1, np.ones_like(quality_anal))
    
    # Use a gray colormap for the bad data overlay
    cmap_mask = colors.ListedColormap(['gray'])
    
    # Plot overlay with transparency (alpha=0.5)
    ax3.pcolormesh(bad_data_mask, cmap=cmap_mask, shading='nearest', alpha=0.5)

    ax3.set_title(f"Sample {sample_idx}: MRMS Analysis (Target)", fontsize=15)
    ax3.invert_yaxis()
    cb3 = fig.colorbar(pcm3, ax=ax3, orientation='vertical', shrink=0.9, 
                       ticks=levs_p, extend='max')
    cb3.set_label('mm')

    # 4. Save Output
    base_name = os.path.basename(filename).replace('.cPick', '')
    output_png = f"plot_{base_name}_sample_{sample_idx}.png"
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to {output_png}")
    plt.close()

if __name__ == "__main__":
    main()



