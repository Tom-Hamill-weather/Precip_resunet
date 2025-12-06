"""

python plot_graf_mrms_samples.py filename sample_index

"""
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ---------------------------------------------------------------------------------
# Colormap Definitions (FIXED)
# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------------

def load_sequential_pickle(filename):
    """
    Reads the file created by 'save_patched_GRAF_MRMS_gemini.py'.
    The format is multiple numpy arrays dumped sequentially.
    """
    # The order defined in the saving script:
    keys_order = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain', 
                  'terrain_diff', 'dt_dlon', 'dt_dlat', 'time']
    
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
            # It is possible the file ended early or has fewer keys
            print(f"Warning: Reached end of file early. Missing keys after {key}.")
            
    return data

# ---------------------------------------------------------------------------------
# Main Plotting Logic
# ---------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_graf_mrms_samples.py <filename> <sample_index>")
        print("Example: python plot_graf_mrms_samples.py GRAF_Unet_data_train_2023040112_12h.cPick 10")
        sys.exit(1)

    filename = sys.argv[1]
    
    try:
        sample_idx = int(sys.argv[2])
    except ValueError:
        print("Error: sample_index must be an integer.")
        sys.exit(1)

    # 1. Load Data
    data_store = load_sequential_pickle(filename)
    
    # Validate we have the necessary keys
    required_keys = ['GRAF', 'terrain_diff', 'MRMS']
    for k in required_keys:
        if k not in data_store:
            print(f"Error: Key '{k}' missing from pickle file.")
            sys.exit(1)
            
    # 2. Extract specific sample
    # All arrays are shape (N_samples, 64, 64)
    total_samples = data_store['GRAF'].shape[0]
    
    if sample_idx >= total_samples:
        print(f"Error: Index {sample_idx} out of bounds. File contains {total_samples} samples.")
        sys.exit(1)

    # Feature 1: Model Forecast (GRAF)
    precip_fcst = data_store['GRAF'][sample_idx]
    
    # Feature 2: Terrain Deviations (terrain_diff)
    terr_dev = data_store['terrain_diff'][sample_idx]
    
    # Target: Analyzed Precip (MRMS)
    precip_anal = data_store['MRMS'][sample_idx]

    # 3. Setup Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    cmap_p, norm_p, levs_p = get_precip_colormap()
    cmap_t, norm_t, levs_t = get_terrain_colormap()

    # --- Panel 1: GRAF Forecast ---
    ax1 = axes[0]
    pcm1 = ax1.pcolormesh(precip_fcst, cmap=cmap_p, norm=norm_p, shading='nearest')
    ax1.set_title(f"Sample {sample_idx}: GRAF Forecast (Feature 1)")
    ax1.invert_yaxis() 
    # extend='max' allows values > 50 to be colored with the 'over' color
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical', shrink=0.9, 
                       ticks=levs_p, extend='max')
    cb1.set_label('mm')

    # --- Panel 2: Terrain Deviations ---
    ax2 = axes[1]
    pcm2 = ax2.pcolormesh(terr_dev, cmap=cmap_t, norm=norm_t, shading='nearest')
    ax2.set_title(f"Sample {sample_idx}: Terrain Deviation (Feature 2)")
    ax2.invert_yaxis()
    # extend='both' allows values outside -1000 to 1000 to be colored
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation='vertical', shrink=0.9, 
                       ticks=levs_t, extend='both')
    cb2.set_label('meters')

    # --- Panel 3: MRMS Analysis ---
    ax3 = axes[2]
    pcm3 = ax3.pcolormesh(precip_anal, cmap=cmap_p, norm=norm_p, shading='nearest')
    ax3.set_title(f"Sample {sample_idx}: MRMS Analysis (Target)")
    ax3.invert_yaxis()
    cb3 = fig.colorbar(pcm3, ax=ax3, orientation='vertical', shrink=0.9, 
                       ticks=levs_p, extend='max')
    cb3.set_label('mm')

    # 4. Save Output
    base_name = os.path.basename(filename).replace('.cPick', '')
    output_png = f"plot_{base_name}_sample_{sample_idx}.png"
    
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Successfully saved plot to {output_png}")
    plt.close()

if __name__ == "__main__":
    main()


