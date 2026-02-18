# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements an Attention Residual U-Net for hourly probabilistic quantitative precipitation forecasting. The system uses patches of GRAF (weather forecast) data and terrain information as features, with MRMS (radar) data as targets, to train a deep learning model that produces better precipitation probability forecasts than raw GRAF output.

The workflow is split between a Cray supercomputer (for data processing) and a laptop (for training/inference), since the underlying GRAF and MRMS data reside on the Cray but training is faster on the laptop.

## Key Commands

### Data Preparation (on Cray)
```bash
# Download and remap MRMS data to GRAF grid (slow, run via slurm)
./control_MRMS_download.sh  # calls control_MRMS_download.slurm

# Extract patches of GRAF/MRMS/terrain data for training
# Requires date (YYYYMMDDHH) and lead time (hours)
python save_patched_GRAF_MRMS_gemini.py 2025120100 12

# Or run for multiple lead times via slurm
./control_save_patched_GRAF_MRMS_gemini.sh
```

### Training (on laptop)
```bash
# Train model for a specific date and lead time
python pytorch_train_resunet.py 2025120100 12

# Train multiple lead times sequentially
./pytorch_train_resunet.sh
```

### Inference (on laptop)
```bash
# Copy GRAF data from Cray to laptop for inference
python copy_graf_to_laptop.py 2025120412 12

# Run inference to generate probability forecasts
python resunet_inference.py 2025120412 12

# Generate visualization plots
python make_plots.py 2025120412 12
```

### Data Transfer
```bash
# Sync code from laptop to Cray (using alias: resunet_sync)
rsync -avz /Users/tom.hamill@weather.com/python/resnet thamill@10.66.63.22:resnet/

# Transfer training data from Cray to laptop
cd ~/python/resnet_data
scp cray:/storage1/home/thamill/resnet/resnet_data/g.tar .
tar xvf g.tar
```

### Utilities
```bash
# Check MRMS data quality
python check_mrms.py

# Plot sample GRAF/MRMS patches
python plot_graf_mrms_samples.py
python control_plot_graf_mrms_samples.py

# Analyze GRAF statistics
python process_graf_stats.py
```

## Architecture

### Model: Attention Residual U-Net (AttnResUNet)

**Input**: 5-channel tensor (96×96 patches)
1. GRAF precipitation forecast
2. Terrain interaction (GRAF × terrain height difference)
3. Local terrain height difference
4. Terrain gradient (longitude direction)
5. Terrain gradient (latitude direction)

**Output**: Probability distribution over 102 precipitation classes (0 to 25mm in 0.25mm increments, plus >25mm)

**Structure** (pytorch_train_resunet.py:149-204):
- Encoder: 4 downsampling stages (64→128→256→512 channels)
- Bridge: 1024 channels at 6×6 resolution
- Decoder: 4 upsampling stages with attention gates
- Skip connections: U-Net style with attention mechanism
- Uses ResidualBlocks (conv-BN-ReLU-conv-BN with shortcuts)
- AttentionGates focus decoder on relevant encoder features

### Loss Function: WeightedOrdinalWassersteinLoss

Custom loss function (pytorch_train_resunet.py:281-305) that:
- Treats precipitation as ordinal categories (not independent classes)
- Uses CDF-based Wasserstein distance
- Applies class weights to emphasize rare heavy precipitation events
- Supports asymmetry factor to penalize false alarms vs. misses differently
- Ignores pixels with bad MRMS data quality (quality <= 0.01)

### Patch-Based Inference Strategy

Full-field inference is performed by:
1. Breaking CONUS domain (1308×1524) into overlapping 96×96 patches
2. Running model on each patch
3. Combining patches using "Manhattan" weights (resunet_inference.py:64-76)
   - Weight decreases linearly from center to edges
   - Prevents discontinuities at patch boundaries
4. Two-pass strategy: offset grids for better coverage (resunet_inference.py:314-317)

### Data Sampling Strategy

The patch extraction (save_patched_GRAF_MRMS_gemini.py:154-179) uses dynamic sampling:
- **Macro**: More patches on wet days (50), fewer on dry days (20), normal otherwise (35)
- **Micro**: Weighted random sampling preferring patches with higher precipitation
- **Quality control**: Excludes patches where >10% of pixels have bad MRMS quality
- Samples from: last 60 days + 10-12 months prior (to get seasonal variety)

### Configuration System

Two config files specify directories for different environments:
- `config_hdo.ini`: Cray paths (GRAF archive, MRMS data)
- `config_laptop.ini`: Laptop paths (local GRAF/prob/plot directories)

Key parameters (config_hdo.ini):
- `ndays_train = 60`: Days of recent data to use for training
- GRAF data location switches at 2024-04-05 (graflr → graf)

## Directory Structure

```
python/
├── resnet/                           # Code directory
│   ├── *.py                          # Python scripts
│   ├── *.sh                          # Shell scripts
│   ├── config_*.ini                  # Configuration files
│   └── GRAF_CONUS_terrain_info.nc   # Static terrain data
└── resnet_data/                      # Data directory (laptop)
    ├── GRAF_Unet_data_*.cPick       # Training/validation patches
    ├── trainings/                    # Trained model weights
    ├── GRAF/                         # GRAF forecast samples
    ├── probs/                        # Inference output (netCDF)
    └── plots/                        # Visualization output (PNG)
```

On Cray:
```
/storage1/home/thamill/resnet/
├── resnet/          # Code (synced from laptop)
└── resnet_data/     # Patch data

/storage2/library/archive/grid/
├── hdo-graf_conus/        # GRAF data (post Apr 2024)
└── hdo-graflr_conus/      # GRAF data (pre Apr 2024)

/storage/home/thamill/MRMS/  # MRMS archive
```

## Training Details

### Hyperparameters (pytorch_train_resunet.py:52-94)
- Patch size: 96×96
- Batch size: 128 (GPU/MPS), 16 (CPU)
- Learning rate: 7e-4 (reduced to 70% for lead times ≥12h)
- Epochs: 30 max (with early stopping patience=5)
- Optimizer: Adam with ReduceLROnPlateau scheduler
- Device selection: CUDA > MPS > CPU (automatic)
- AMP (mixed precision): Enabled on CUDA only

### Data Augmentation (pytorch_train_resunet.py:260-267)
Training patches are randomly:
- Flipped horizontally (negates dlon gradient channel)
- Flipped vertically (negates dlat gradient channel)

### Normalization (pytorch_train_resunet.py:233-255)
Each feature channel normalized to [0,1] using fixed max values:
- GRAF precip: 75 mm
- Terrain interaction: 35000
- Terrain diff: 2500
- dlon/dlat: 0.02

Normalization statistics saved in checkpoint for consistent inference.

### Checkpointing
- Saves after every epoch: `resunet_ordinal_{date}_{lead}h_epoch_{N}.pth`
- Includes: model weights, optimizer state, scheduler state, loss, normalization stats
- Automatically resumes from latest checkpoint if available
- Inference searches for weights matching lead time (or closest available)

## Important Notes

### Device Compatibility
The code supports CUDA, MPS (Apple Silicon), and CPU. Mixed precision training (AMP) is only enabled on CUDA due to numerical instability on MPS.

### Lead Time Handling
Training is lead-time specific. The code will look for existing weights at the target lead time first, then fall back to nearby lead times (e.g., 15h weights for 18h training) to warm-start training.

### Data Quality
Bad MRMS pixels (quality ≤ 0.01) are masked with target=-1 and ignored in loss computation. Patches with >10% bad pixels are excluded during sampling.

### GRAF Data Transition
On 2024-04-05, GRAF naming changed from "hdo-graflr_conus" to "hdo-graf_conus". The code handles this automatically based on date comparisons.

### Threshold Mapping (for inference output)
- 0.25mm: class 1+
- 1.0mm: class 4+
- 2.5mm: class 10+
- 5.0mm: class 20+
- 10.0mm: class 40+
- 25.0mm: class 100+

## Dependencies

Key Python libraries:
- PyTorch (with CUDA/MPS support)
- netCDF4
- pygrib (for GRIB2 files)
- numpy, scipy
- matplotlib, basemap (for plotting)
- dateutils (Jeff Whitaker's module)

## Testing

To verify patches look realistic:
```bash
python plot_graf_mrms_samples.py
python control_plot_graf_mrms_samples.py  # batch version
```

To plot full GRAF/MRMS fields (on Cray):
```bash
python plot_GRAF_MRMS.py 2025120100 12
```
