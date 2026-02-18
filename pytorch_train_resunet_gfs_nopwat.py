"""
pytorch_train_resunet_gfs_nopwat.py

Usage example:

$ python pytorch_train_resunet_gfs_nopwat.py 2025120100 12

where you supply the YYYYMMDDHH of initial condition and lead time in h.

This routine will train an Attention Residual U-Net for the prediction of
hourly probabilistic precipitation in classes based on GRAF precipitation
forecast data, terrain information, and GFS features (PWAT, column-average RH, CAPE).
Data were previously saved by running the script save_patched_GRAF_MRMS_GFS.py
on the Cray and transferring data back to the laptop. These saved the previous 60
days and data from 10-12 months prior, giving the training 4 months of
data centered on the Julian day of the year.   Subsequent to training,
you can upload a selected day/hour of GRAF precip to your laptop
(copy_graf_to_laptop.py) and run inference, generating plots
(resunet_inference.py)

The output are a set of trained weights, stored in ../resnet_data/trainings.

Concerning the naming: "Attention Residual U-Net"

"Residual": This refers to the intra-block skip connections implemented
in the ResidualBlock. This allows training deeper networks by preventing
gradient vanishing.

"Attention": This refers to the inter-block gating signals implemented
in the AttentionGate. The decoder uses its own features to "query" the
encoder features, suppressing irrelevant areas (like empty sky) and
highlighting relevant ones (like terrain slopes) before merging them.
It focuses the network on specific spatial locations without requiring
a deeper stack of convolutional layers.

"U-Net": This refers to the overall encoder-decoder "U" shape with
long skip connections.

Coded by Tom Hamill with Claude Code assistance.

Latest version: Adapted from pytorch_train_resunet_gfs.py to REMOVE PWAT and CAPE
(PWAT is climatologically dependent and has poor discriminatory power at low
precip rates. Analysis showed only 1.3 kg/m² difference between dry and light
precip patches, leading to widespread false alarms. RH shows much better
separation: 21% (dry) vs 36% (light) vs 45% (moderate) vs 49% (heavy).
CAPE was removed to simplify the model.)

Input features (7 channels):
(1) GRAF precipitation forecast
(2) Terrain elevation deviation (local terrain height difference)
(3) GFS column-average relative humidity
(4) Interaction: GRAF × terrain elevation deviation
(5) Interaction: GRAF × GFS relative humidity
(6) Terrain gradient (longitude direction)
(7) Terrain gradient (latitude direction)

"""

import os
import sys
import glob
import re
import _pickle as cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================

# --- 1. Set Device (GPU/CPU) ---

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # For Apple Silicon (M1/M2/M3)
else:
    DEVICE = torch.device("cpu")

# --- 2. Set Hardware-Specific Params ---

if DEVICE.type == 'cpu':
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    USE_AMP = False
else:
    BATCH_SIZE = 16  # Reduced from 128 for memory efficiency
    NUM_WORKERS = 2  # Reduced from 5 to lower memory overhead
    # Use AMP (Automatic Mixed Precision) only for CUDA.
    # MPS (Apple) generally runs better in default precision for now.
    USE_AMP = (DEVICE.type == 'cuda')

# Gradient accumulation to simulate larger effective batch size
ACCUMULATION_STEPS = 8  # Effective batch size = 16 * 8 = 128

# --- 3. Training Hyperparameters ---

PATCH_SIZE = 96
BASE_LEARNING_RATE = 7.e-4
NUM_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 5  # Longer allows model to learn rare events

# --- 4. THRESHOLDS for probabilistic categorical forecasts  ---

THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
THRESHOLDS.append(200.0)  # One large category at the end for 25-200 mm
NUM_CLASSES = len(THRESHOLDS)

THRESHOLD_TENSOR = torch.tensor(THRESHOLDS[:-1], device=DEVICE, dtype=torch.float32)

# --- 5. BOUNDARY WEIGHTS (For Wasserstein Metric) ---

weights_np = np.diff(THRESHOLDS)
weights_np = np.clip(weights_np, a_min=None, a_max=5.0)
weights_np[0] = 1.0  # Gives more weight to getting the zero fcst correct
CLASS_WEIGHTS = torch.tensor(weights_np, device=DEVICE, dtype=torch.float32)

# --- 6. PIXEL WEIGHTS (adjusted for enriched data) ---
#     A gentle ramp to give more weight in the loss to large events;
#     Note that the biasing toward getting heavy precipitation
#     correct was also addressed by the routine that saved patches,
#     which loaded up many wet samples.

# Class 0 (0mm)    -> Weight 1.0
# Class 100 (25mm) -> Weight 3.0
pixel_weights_np = 1.0 + (np.arange(NUM_CLASSES) * 0.02)
# Cap at 3.0
pixel_weights_np = np.clip(pixel_weights_np, a_min=None, a_max=3.0)
PIXEL_WEIGHTS = torch.tensor(pixel_weights_np, device=DEVICE, dtype=torch.float32)

# --- 7. ASYMMETRY FACTOR ---
# Penalty multiplier for Under-prediction (Misses).
# 1.0 means symmetric (equal penalty for false alarms and misses)
# >1.0 means missing a storm is worse than a false alarm.

ASYMMETRY_FACTOR = 1.0

# --- 8. LOSS POWER ---
# Power for L^p distance in Wasserstein/CRPS loss.
# 2.0 = standard L² CRPS (more sensitive to large errors)
# 1.5 = intermediate (balanced sensitivity)
# 1.0 = L¹ distance (less sensitive to large errors)

LOSS_POWER = 2.0

TRAIN_DIR = '../resnet_data/trainings'
DATA_DIR = '../resnet_data'

# ====================================================================
# --- MODEL ARCHITECTURE ---
# ====================================================================

class ResidualBlock(nn.Module):
    """
    True Residual Block with Identity Mapping.
    Structure: Output = ReLU( ConvBlock(x) + Shortcut(x) )
    Allows training deeper networks by preventing gradient vanishing.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)

class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing decoder on relevant encoder features.
    Uses decoder features (g) to "query" encoder features (x),
    suppressing irrelevant areas and highlighting important ones.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttnResUNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=NUM_CLASSES):
        super(AttnResUNet, self).__init__()
        # Initial dimensions: (Batch, 7, 96, 96)
        self.inc = ResidualBlock(in_channels, 64)

        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))   # (B, 128, 48, 48)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))  # (B, 256, 24, 24)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))  # (B, 512, 12, 12)

        # Bridge
        self.bridge = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024)) # (B, 1024, 6, 6)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv3 = ResidualBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv4 = ResidualBlock(128, 64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)

        x = self.up1(x5)
        x4 = self.att1(g=x, x=x4)
        x = self.conv1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x3 = self.att2(g=x, x=x3)
        x = self.conv2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x2 = self.att3(g=x, x=x2)
        x = self.conv3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x1 = self.att4(g=x, x=x1)
        x = self.conv4(torch.cat([x, x1], dim=1))
        return self.outc(x)

# ====================================================================
# --- Dataset and Loss ---
# ====================================================================

class GRAF_Dataset(Dataset):
    def __init__(self, pickle_file, thresholds=THRESHOLDS, normalization_stats=None, train=False):
        self.train = train
        try:
            with open(pickle_file, 'rb') as f:
                self.graf = cPickle.load(f)
                self.mrms = cPickle.load(f)
                self.qual = cPickle.load(f)
                self.terdiff_graf = cPickle.load(f)
                self.diff = cPickle.load(f)
                self.dlon = cPickle.load(f)
                self.dlat = cPickle.load(f)
                self.init_times = cPickle.load(f)
                self.valid_times = cPickle.load(f)
                self.gfs_pwat = cPickle.load(f)  # Still in file but not used
                self.gfs_r = cPickle.load(f)
                self.gfs_cape = cPickle.load(f)  # Still in file but not used
        except Exception as e:
            print(f"CRITICAL ERROR loading pickle: {e}")
            sys.exit(1)

        # Validation: check if the loaded arrays match 96x96
        if self.graf.shape[1] != PATCH_SIZE or self.graf.shape[2] != PATCH_SIZE:
             print(f"WARNING: Data shape {self.graf.shape} does not match PATCH_SIZE {PATCH_SIZE}")

        self.thresholds = np.array(thresholds)

        # Compute GRAF × RH interaction term from raw values
        self.graf_rh_interaction = self.graf * self.gfs_r

        # Feature list in order: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
        feature_list = [self.graf, self.diff, self.gfs_r, self.terdiff_graf,
                       self.graf_rh_interaction, self.dlon, self.dlat]

        if normalization_stats is None:
            mins = [float(np.min(arr)) for arr in feature_list]
            maxs = [float(np.max(arr)) for arr in feature_list]
            # Set sensible max values for each feature
            maxs[0] = 75.0          # GRAF precip
            maxs[1] = max(maxs[1], 2500.0)   # terrain diff
            maxs[2] = max(maxs[2], 100.0)    # RH (%)
            maxs[3] = max(maxs[3], 35000.0)  # GRAF × terrain interaction
            maxs[4] = max(maxs[4], 7500.0)   # GRAF × RH interaction (75mm × 100%)
            maxs[5] = max(maxs[5], 0.02)     # dlon
            maxs[6] = max(maxs[6], 0.02)     # dlat
            self.stats = {'min': mins, 'max': maxs}
        else:
            self.stats = normalization_stats

        # Normalize all features in order
        self.graf = self.normalize(self.graf, 0)
        self.diff = self.normalize(self.diff, 1)
        self.gfs_r = self.normalize(self.gfs_r, 2)
        self.terdiff_graf = self.normalize(self.terdiff_graf, 3)
        self.graf_rh_interaction = self.normalize(self.graf_rh_interaction, 4)
        self.dlon = self.normalize(self.dlon, 5)
        self.dlat = self.normalize(self.dlat, 6)

    def normalize(self, data, idx):
        vmin = self.stats['min'][idx]
        vmax = self.stats['max'][idx]
        denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
        return ((data - vmin) / denom).astype(np.float32)

    def __len__(self):
        return len(self.graf)

    def apply_augmentation(self, x, y):
        # Horizontal flip (negate dlon channel, which is channel 5)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2); y = np.flip(y, axis=1)
            x[5, :, :] = -x[5, :, :]
        # Vertical flip (negate dlat channel, which is channel 6)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1); y = np.flip(y, axis=0)
            x[6, :, :] = -x[6, :, :]
        return x.copy(), y.copy()

    def __getitem__(self, idx):
        # Stack in order: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
        x = np.stack([self.graf[idx], self.diff[idx], self.gfs_r[idx],
                     self.terdiff_graf[idx], self.graf_rh_interaction[idx],
                     self.dlon[idx], self.dlat[idx]], axis=0)
        y_raw = self.mrms[idx]
        q_mask = self.qual[idx]
        is_bad = (q_mask <= 0.01)
        y_indices = np.searchsorted(self.thresholds, y_raw, side='right') - 1
        y_indices = np.clip(y_indices, 0, len(self.thresholds) - 2)
        y_indices[is_bad] = -1
        if self.train:
            x, y_indices = self.apply_augmentation(x, y_indices)
        return torch.from_numpy(x), torch.from_numpy(y_indices).long()

class WeightedOrdinalWassersteinLoss(nn.Module):
    def __init__(self, num_classes, boundary_weights=None, class_weights=None, asymmetry_factor=1.0, ignore_index=-1, power=2.0):
        super(WeightedOrdinalWassersteinLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.power = power
        self.asymmetry_factor = asymmetry_factor
        self.register_buffer('boundary_weights', boundary_weights if boundary_weights is not None else torch.ones(num_classes - 1))
        self.register_buffer('class_weights', class_weights if class_weights is not None else torch.ones(num_classes))

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clamp(0, self.num_classes - 1)
        class_idx = torch.arange(self.num_classes, device=logits.device, dtype=safe_targets.dtype).view(1, self.num_classes, 1, 1)
        target_cdf = (class_idx >= safe_targets.unsqueeze(1)).float()
        raw_diff = pred_cdf - target_cdf
        asym_weights = torch.ones_like(raw_diff)
        if self.asymmetry_factor != 1.0: asym_weights[raw_diff > 0] = self.asymmetry_factor
        diff = (torch.abs(raw_diff) ** self.power) * asym_weights
        weighted_diff = diff[:, :-1, :, :] * self.boundary_weights.view(1, self.num_classes - 1, 1, 1)
        pixel_loss = weighted_diff.sum(dim=1) * self.class_weights[safe_targets.to(torch.int32)]
        pixel_loss = pixel_loss * valid_mask.float()
        return pixel_loss.sum() / valid_mask.sum().clamp_min(1.0)

# ====================================================================
# --- Diagnostic Testing ---
# ====================================================================

def print_diagnostics(epoch, batch_idx, loss_val, outputs, \
        targets, model, stats):
    """
    Print diagnostics during training to monitor model learning.

    Adapted from original pytorch_train_resunet.py to include GFS features.
    Shows:
    1. Real data statistics (predicted vs actual distributions)
    2. Synthetic test cases: 0mm (dry) and 1mm (light rain)
    """

    # Print explanation on first call (epoch 0, batch 0)
    if epoch == 0 and batch_idx == 0:
        print("\n" + "="*82)
        print("DIAGNOSTIC OUTPUT EXPLANATION")
        print("="*82)
        print("\nThis table shows how the model is learning precipitation distributions.")
        print("\nColumn descriptions:")
        print("  Category:   Precipitation range being evaluated")
        print("  Range (mm): Physical precipitation amounts in this category")
        print("  Mean Prob:  Average probability the model assigns to this category")
        print("  Pred %:     Percentage of pixels where model predicts this as most likely category")
        print("  True %:     Percentage of pixels where MRMS observations fall in this category")
        print("  Syn(0mm):   Model probability for this category given synthetic DRY input:")
        print("              (GRAF=0mm, RH=20%, CAPE=0 J/kg, flat terrain)")
        print("  Syn(1mm):   Model probability for this category given synthetic LIGHT RAIN input:")
        print("              (GRAF=1mm, RH=80%, CAPE=0 J/kg, flat terrain)")
        print("\nWhat to look for:")
        print("  - Pred % and True % should be similar (model matches observations)")
        print("  - Syn(0mm) should be HIGH for 'No precip' (~0.7+), LOW elsewhere (<0.1)")
        print("  - Syn(1mm) should be HIGH for '0.25-0.5' and '0.5-1.0', moderate for '1-2 mm'")
        print("  - If Syn(0mm) spreads across categories, model has wet bias (false alarms)")
        print("="*82 + "\n")

    # Save current mode and switch to eval for synthetic tests
    was_training = model.training
    model.eval()

    with torch.no_grad():

        # --- REAL DATA STATS ---
        pdf = torch.softmax(outputs, dim=1)

        # --- SYNTHETIC TESTS ---
        def run_synthetic(precip_mm, rh_pct):
            """
            Run model on synthetic uniform input.

            7 features:
            0: GRAF Precip
            1: TerrainDiff
            2: RH (%)
            3: GRAF × TerrainDiff
            4: GRAF × RH
            5: dlon
            6: dlat
            """

            p_val = precip_mm
            t_val = 0.0  # Flat terrain

            f0 = p_val              # GRAF precip
            f1 = t_val              # Terrain diff
            f2 = rh_pct             # RH
            f3 = p_val * t_val      # GRAF × terrain
            f4 = p_val * rh_pct     # GRAF × RH
            f5 = 0.0                # dlon
            f6 = 0.0                # dlat

            phys_vals = [f0, f1, f2, f3, f4, f5, f6]

            # Normalize
            norm_vals = []
            for i, val in enumerate(phys_vals):
                vmin = stats['min'][i]
                vmax = stats['max'][i]
                denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
                n_val = (val - vmin) / denom
                norm_vals.append(n_val)

            # Create Tensor (7 channels)
            syn_x = torch.zeros((1, 7, 96, 96), device=DEVICE)
            for i, nv in enumerate(norm_vals):
                syn_x[:, i, :, :] = nv

            # Specify device_type explicitly
            amp_device = 'cuda' if USE_AMP else 'cpu'
            with torch.amp.autocast(amp_device, enabled=USE_AMP):
                out = model(syn_x)
                syn_p = torch.softmax(out, dim=1)

            return syn_p.mean(dim=(0,2,3)).cpu().numpy()

        # Dry case: 0mm GRAF, 20% RH
        syn_0mm_pdf = run_synthetic(0.0, 20.0)
        # Light rain case: 1mm GRAF, 80% RH
        syn_1mm_pdf = run_synthetic(1.0, 80.0)

        # --- AGGREGATE STATS ---
        print(f"--- Epoch {epoch+1}, Batch {batch_idx} ---")
        print(f"Loss: {loss_val:.4f}")
        print(f"{'Category':<12} | {'Range (mm)':<12} | {'Mean Prob':<9} | "
              f"{'Pred %':<6} | {'True %':<6} | {'Syn(0mm)':<8} | {'Syn(1mm)':<8}")
        print("-" * 82)

        # Target indices are passed directly
        target_indices = targets
        pred_indices = torch.argmax(pdf, dim=1)

        # Define meaningful precipitation categories
        # Class indices correspond to thresholds: 0.00, 0.25, 0.50, 0.75, 1.00, ...
        categories = [
            ('No precip',   0,      0,      '0.00'),          # Class 0: [0.00, 0.25)
            ('0.25-0.5',    1,      1,      '0.25-0.5'),      # Class 1: [0.25, 0.50)
            ('0.5-1.0',     2,      3,      '0.5-1.0'),       # Classes 2-3: [0.50, 1.00)
            ('1-2 mm',      4,      7,      '1.0-2.0'),       # Classes 4-7: [1.00, 2.00)
            ('2-3 mm',      8,      11,     '2.0-3.0'),       # Classes 8-11
            ('3-4 mm',      12,     15,     '3.0-4.0'),       # Classes 12-15
            ('4-5 mm',      16,     19,     '4.0-5.0'),       # Classes 16-19
            ('5-6 mm',      20,     23,     '5.0-6.0'),       # Classes 20-23
            ('6-7 mm',      24,     27,     '6.0-7.0'),       # Classes 24-27
            ('7-8 mm',      28,     31,     '7.0-8.0'),       # Classes 28-31
            ('8-9 mm',      32,     35,     '8.0-9.0'),       # Classes 32-35
            ('9+ mm',       36,     NUM_CLASSES-1, '9.0+'),   # Classes 36+
        ]

        for label, start, end, range_str in categories:
            if start >= NUM_CLASSES:
                continue

            # Compute statistics
            bin_prob = pdf[:, start:end+1].sum(dim=1).mean().item()
            mask_true = (target_indices >= start) & (target_indices <= end)
            true_pct = mask_true.float().mean().item() * 100
            mask_pred = (pred_indices >= start) & (pred_indices <= end)
            pred_pct = mask_pred.float().mean().item() * 100

            # Synthetic test statistics
            syn0 = syn_0mm_pdf[start:end+1].sum()
            syn1 = syn_1mm_pdf[start:end+1].sum()

            print(f"{label:<12} | {range_str:<12} | {bin_prob:.3f}     | "
                  f"{pred_pct:6.3f} | {true_pct:6.3f} | {syn0:.3f}    | {syn1:.3f}")

        print("-" * 82)

    # Restore training mode
    if was_training:
        model.train()

# ====================================================================
# --- Training Logic ---
# ====================================================================

def train_model(date_str, lead_time_str):
    print(f"\n{'='*70}")
    print(f"Training ResUNet with GFS features for {date_str} at {lead_time_str}h lead time")
    print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | AMP: {USE_AMP}")
    print(f"{'='*70}\n")

    # --- Lead-Time Dependent Adjustment ---
    # Longer lead times have higher uncertainty; reducing initial LR for stability
    lead_h = int(lead_time_str)
    current_lr = BASE_LEARNING_RATE
    if lead_h >= 12:
        current_lr = BASE_LEARNING_RATE * 0.7
        print(f"   Adjusting base LR for long lead time ({lead_h}h): {current_lr:.2e}")

    train_pattern = f"{DATA_DIR}/GRAF_Unet_data_train_*{date_str}*_{lead_time_str}h.cPick*"
    val_pattern   = f"{DATA_DIR}/GRAF_Unet_data_test_*{date_str}*_{lead_time_str}h.cPick*"

    train_files = glob.glob(train_pattern)
    val_files = glob.glob(val_pattern)

    if not train_files:
        print(f"ERROR: No training files found matching pattern: {train_pattern}")
        sys.exit(1)
    if not val_files:
        print(f"ERROR: No validation files found matching pattern: {val_pattern}")
        sys.exit(1)

    train_file = train_files[0]
    val_file = val_files[0]

    print(f"Loading training data from: {train_file}")
    print(f"Loading validation data from: {val_file}")

    train_dataset = GRAF_Dataset(train_file, THRESHOLDS, train=True)
    val_dataset = GRAF_Dataset(val_file, THRESHOLDS, normalization_stats=train_dataset.stats, train=False)

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input channels: 7 (GRAF, terrain, GFS RH, interactions, gradients)")
    print(f"  Output classes: {NUM_CLASSES}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = AttnResUNet(in_channels=7, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # --- Scheduler Integration ---
    # Patience: wait 2 epochs of no improvement, then cut LR by half
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    scaler = GradScaler() if USE_AMP else None
    criterion = WeightedOrdinalWassersteinLoss(num_classes=NUM_CLASSES, boundary_weights=CLASS_WEIGHTS, class_weights=PIXEL_WEIGHTS, asymmetry_factor=ASYMMETRY_FACTOR, power=LOSS_POWER).to(DEVICE)

    start_epoch = 0
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)

    # Resume Logic - look for best checkpoint
    best_checkpoint_path = f"{TRAIN_DIR}/resunet_ordinal_gfs_nopwat_{date_str}_{lead_time_str}h_best.pth"
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if os.path.exists(best_checkpoint_path):
        print(f"Found existing checkpoint: {best_checkpoint_path}")
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['loss']
            print(f"   Resuming from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}\n")
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print(f"   WARNING: Checkpoint incompatible (different number of channels)")
                print(f"   Starting fresh training (old checkpoint had different architecture)")
                best_checkpoint_path = None
            else:
                raise
    else:
        best_checkpoint_path = None

    print(f"Starting training from epoch {start_epoch+1}...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Diagnostic output frequency: once per epoch\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            amp_device = 'cuda' if USE_AMP else 'cpu'
            with torch.amp.autocast(amp_device, enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / ACCUMULATION_STEPS  # Scale loss for gradient accumulation
            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS  # Unscale for logging

            # Clear GPU cache periodically to prevent memory fragmentation
            if i % 100 == 0 and DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        # Handle remaining gradients if batch count not divisible by ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS != 0:
            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                amp_device = 'cuda' if USE_AMP else 'cpu'
                with torch.amp.autocast(amp_device, enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate accuracy (correct class prediction)
                pred_classes = torch.argmax(outputs, dim=1)
                # Only count pixels with valid targets (not masked)
                valid_mask = (targets != -1)
                val_correct += ((pred_classes == targets) & valid_mask).sum().item()
                val_total += valid_mask.sum().item()

        avg_val = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        # Print epoch-end diagnostics with synthetic tests
        # Use last validation batch for real data stats
        print_diagnostics(epoch, len(train_loader)-1, avg_train_loss,
                         outputs, targets, model, train_dataset.stats)

        # --- Update Scheduler ---
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = (new_lr != old_lr)

        # Track improvement
        improved = ""
        if avg_val < best_val_loss:
            improved = " *BEST*"

        # Format output with more diagnostics
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val:.4f}{improved} | "
              f"Acc: {val_accuracy:.2f}% | "
              f"LR: {new_lr:.2e}" +
              (" [reduced]" if lr_changed else ""))

        # Save checkpoint ONLY if validation improved
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0

            # Delete previous best checkpoint
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
                print(f"   Deleted previous best checkpoint")

            # Save new best checkpoint
            save_path = f"{TRAIN_DIR}/resunet_ordinal_gfs_nopwat_{date_str}_{lead_time_str}h_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val,
                'normalization_stats': train_dataset.stats
            }, save_path)
            best_checkpoint_path = save_path
            print(f"   Saved new best checkpoint: {os.path.basename(save_path)}")
        else:
            epochs_no_improve += 1
            print(f"   (No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epochs)")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered after {epoch+1} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f} (Epoch {epoch+1-epochs_no_improve})")
                print(f"{'='*70}\n")
                break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"  Best validation loss:  {best_val_loss:.4f}")
    print(f"  Final epoch:           {epoch+1}/{NUM_EPOCHS}")
    print(f"  Final learning rate:   {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Weights saved to:      {TRAIN_DIR}")
    print(f"  Checkpoint filename:   resunet_ordinal_gfs_nopwat_{date_str}_{lead_time_str}h_best.pth")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        train_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python pytorch_train_resunet_gfs_nopwat.py <DATE> <LEAD>")
        print("Example: python pytorch_train_resunet_gfs_nopwat.py 2025120100 12")
