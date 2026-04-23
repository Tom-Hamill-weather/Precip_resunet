"""
python plot_graf_mrms_gfs_samples.py filename sample_index

Intended to plot samples of patches from train, test, validation data.
Shows GFS column-average relative humidity (r) as model feature.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids probing display on headless servers
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

def get_rh_colormap():
    """
    Defines colormap for column-average relative humidity (%).
    """
    # Use a brown-to-blue colormap
    c_list = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5',
              '#c7eae5', '#80cdc1', '#35978f', '#01665e']

    cmap = colors.ListedColormap(c_list)

    # RH ranges from 0-100%
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]

    cmap.set_over('#01665e')

    norm = colors.BoundaryNorm(levels, cmap.N)

    return cmap, norm, levels

# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------

def load_single_sample(filename, sample_idx):
    """
    Load only one patch (sample_idx) from a patch data file.

    For NetCDF files this reads a single patch via indexed access without
    loading the full dataset into memory — much faster when files hold
    thousands of patches.  For pickle files the whole file must be read.

    Returns (data_dict, total_samples).  data_dict values are 2-D arrays
    (96×96).  Returns (None, total_samples) when sample_idx is out of range.
    """
    # Resolve alternate extension if the given path doesn't exist
    if not os.path.exists(filename):
        ext_pairs = [('.cPick', '.nc'), ('.nc', '.cPick')]
        for ext_from, ext_to in ext_pairs:
            if filename.endswith(ext_from):
                alt = filename.replace(ext_from, ext_to)
                if os.path.exists(alt):
                    print(f"Note: using {os.path.basename(alt)}")
                    filename = alt
                break
        else:
            print(f"Error: File {filename} not found.")
            sys.exit(1)

    print(f"Reading file: {filename}")

    _PLOT_KEYS = ['GRAF', 'terrain_diff', 'MRMS', 'MRMS_qual', 'GFS_r']

    if filename.endswith('.nc'):
        from netCDF4 import Dataset
        nc = Dataset(filename, 'r')
        total = len(nc.dimensions['patch'])
        if sample_idx >= total:
            nc.close()
            return None, total
        data = {k: np.array(nc.variables[k][sample_idx]) for k in _PLOT_KEYS}
        nc.close()
        return data, total
    else:
        # Pickle is sequential — must load the whole file
        from data_loader_utils import load_training_data
        all_data = load_training_data(filename)
        total = all_data['GRAF'].shape[0]
        if sample_idx >= total:
            return None, total
        return {k: all_data[k][sample_idx] for k in _PLOT_KEYS}, total


def load_patch_file(filename):
    """
    Reads the file created by 'save_patched_GRAF_MRMS_GFS.py'.
    Supports both NetCDF (.nc, used on G5 GPU instance) and pickle (.cPick)
    formats via data_loader_utils.load_training_data().
    If the exact path is not found, tries the alternate extension.
    """
    from data_loader_utils import load_training_data

    if not os.path.exists(filename):
        if filename.endswith('.cPick'):
            alt = filename.replace('.cPick', '.nc')
        elif filename.endswith('.nc'):
            alt = filename.replace('.nc', '.cPick')
        else:
            alt = None

        if alt and os.path.exists(alt):
            print(f"Note: {os.path.basename(filename)} not found; using {os.path.basename(alt)}")
            filename = alt
        else:
            print(f"Error: File {filename} not found.")
            sys.exit(1)

    print(f"Reading file: {filename}")
    return load_training_data(filename)

# ---------------------------------------------------------------------------------
# Main Plotting Logic
# ---------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_graf_mrms_gfs_samples.py <filename> <sample_index>")
        print("Example (G5):  python plot_graf_mrms_gfs_samples.py "
              "/data/resnet_data/patch_data/GRAF_Unet_data_train_2025120100_12h.nc 12")
        print("Example (CPU): python plot_graf_mrms_gfs_samples.py "
              "/data2/resnet_data/trainings/GRAF_Unet_data_train_2025120100_12h.nc 12")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        sample_idx = int(sys.argv[2])
    except ValueError:
        print("Error: sample_index must be an integer.")
        sys.exit(1)

    # 1. Load only the requested sample (NetCDF: single indexed read, not the full file)
    sample, total_samples = load_single_sample(filename, sample_idx)

    if sample is None:
        print(f"Error: Index {sample_idx} out of bounds. "
              f"File contains {total_samples} samples.")
        sys.exit(1)

    # 2. Extract arrays (already 2-D; no second indexing needed)
    precip_fcst  = sample['GRAF']
    terr_dev     = sample['terrain_diff']
    precip_anal  = sample['MRMS']
    quality_anal = sample['MRMS_qual']
    gfs_r        = sample['GFS_r']

    # 3. Set up Plotting - 3 panels top row, 1 centered panel bottom row

    fig = plt.figure(figsize=(18, 10))

    # Top row: 3 equally-spaced panels
    gs_top = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.93, bottom=0.52, wspace=0.35)
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax2 = fig.add_subplot(gs_top[0, 1])
    ax3 = fig.add_subplot(gs_top[0, 2])

    # Bottom row: same column geometry, single panel in the centre column
    gs_bot = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.44, bottom=0.05, wspace=0.35)
    ax5 = fig.add_subplot(gs_bot[0, 1])

    cmap_p, norm_p, levs_p = get_precip_colormap()
    cmap_t, norm_t, levs_t = get_terrain_colormap()
    cmap_rh, norm_rh, levs_rh = get_rh_colormap()

    # --- Panel 1: GRAF Forecast ---

    pcm1 = ax1.pcolormesh(precip_fcst, cmap=cmap_p, \
        norm=norm_p, shading='nearest')
    ax1.set_title(f"Sample {sample_idx}: GRAF Forecast (Feature 1)", fontsize=13)
    ax1.invert_yaxis()
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical', shrink=0.9,
        ticks=levs_p, extend='max')
    cb1.set_label('mm', fontsize=10)

    # --- Panel 2: Terrain Deviations ---

    pcm2 = ax2.pcolormesh(terr_dev, cmap=cmap_t, \
        norm=norm_t, shading='nearest')
    ax2.set_title(f"Sample {sample_idx}: Terrain Deviation (Feature 2)", fontsize=13)
    ax2.invert_yaxis()
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation='vertical', shrink=0.9,
                       ticks=levs_t, extend='both')
    cb2.set_label('meters', fontsize=10)

    # --- Panel 3: MRMS Analysis (with Quality Mask) ---

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

    ax3.set_title(f"Sample {sample_idx}: MRMS Analysis (Target)", fontsize=13)
    ax3.invert_yaxis()
    cb3 = fig.colorbar(pcm3, ax=ax3, orientation='vertical', shrink=0.9,
                       ticks=levs_p, extend='max')
    cb3.set_label('mm', fontsize=10)

    # --- Panel 4 (bottom centre): GFS Column-Average Relative Humidity ---

    pcm5 = ax5.pcolormesh(gfs_r, cmap=cmap_rh, \
        norm=norm_rh, shading='nearest')
    ax5.set_title(f"Sample {sample_idx}: GFS Column RH (Feature)", fontsize=13)
    ax5.invert_yaxis()
    cb5 = fig.colorbar(pcm5, ax=ax5, orientation='vertical', shrink=0.9,
                       ticks=levs_rh, extend='max')
    cb5.set_label('%', fontsize=10)

    # 4. Save Output
    base_name = os.path.basename(filename).replace('.cPick', '').replace('.nc', '')
    output_png = f"plot_{base_name}_sample_{sample_idx}.png"

    plt.savefig(output_png, dpi=300)
    print(f"Successfully saved plot to {output_png}")
    plt.close()

if __name__ == "__main__":
    main()
