"""
pytorch_train_resunet.py

Usage example:

$ python pytorch_train_resunet.py 2025120100 12

where you supply the YYYYMMDDHH of initial condition and lead time in h.

This routine will train an Attention Residual U-Net for the prediction of 
hourly probabilistic precipitation in classes based on GRAF precipitation
forecast data and terrain information.   Data were previously saved
by running the script save_patched_GRAF_MRMS_gemini.py on the Cray
and transferring data back to the laptop. These saved the previous 60
days and data from 10-12 months prior, giving the training 4 months of
data centered on the Julian day of the year.   Subsequent to training, 
you can upload a selected day/hour of GRAF precip to your laptop 
(copy_graf_to_laptop.py) and run inference, generating plots 
(resunet_inference.py)

The output are a set of trained weights, stored in ../resnet_data/trainings.

Concerning the naming: "Attention Residual U-Net"

"Residual": This refers to the intra-block skip connections you 
implemented in the ResidualBlock. This allows training deeper 
networks by preventing gradient vanishing.

"Attention": This refers to the inter-block gating signals you 
implemented in the AttentionGate. The decoder uses its own 
features to "query" the encoder features, suppressing 
irrelevant areas (like empty sky) and highlighting relevant 
ones (like terrain slopes) before merging them.  It focuses 
the network on specific spatial locations without requiring 
a deeper stack of convolutional layers.

"U-Net": This refers to the overall encoder-decoder "U" shape 
with long skip connections.

Coded by Tom Hamill with Gemini assistance.  

Latest version 14 Dec 2025 (Modified for Asymmetry/Pixel Weighting)
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
from torch.amp import autocast # New location for generic amp
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================

# --- 1. Set Device (GPU/CPU) ---

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # For Apple Silicon (M1/M2/M3)
else:
    DEVICE = torch.device("cpu")
#print(f"Running on: {DEVICE}")

# --- 2. Set Hardware-Specific Params ---

if DEVICE.type == 'cpu':
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    USE_AMP = False
else:
    BATCH_SIZE = 128
    NUM_WORKERS = 5 
    # Use AMP (Automatic Mixed Precision) only for CUDA. 
    # MPS (Apple) generally runs better in default precision for now.
    USE_AMP = (DEVICE.type == 'cuda')

LEARNING_RATE = 5.e-4 
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5 # longer --> allow model to learn rare events

# --- THRESHOLDS for probabilistic categorical forecasts  ---

THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
THRESHOLDS.append(200.0) # one whopping big category at the end for 25-200 mm
NUM_CLASSES = len(THRESHOLDS) 

THRESHOLD_TENSOR = torch.tensor(THRESHOLDS[:-1], \
    device=DEVICE, dtype=torch.float32)

# --- 1. BOUNDARY WEIGHTS (For Wasserstein Metric) ---

weights_np = np.diff(THRESHOLDS)
weights_np = np.clip(weights_np, a_min=None, a_max=5.0)
weights_np[0] = 3.0 # gives a bit more weight to getting the zero fcst correct.
CLASS_WEIGHTS = torch.tensor(weights_np, device=DEVICE, \
    dtype=torch.float32)

# --- 2. PIXEL WEIGHTS (adjusted for enriched data) ---
#     A gentle ramp to give more weight in the loss to large events; 
#     Note that the biasing toward getting heavy precipitation 
#     correct was also addressed by the routine that saved patches,
#     which loaded up many wet samples.   

# Class 0 (0mm)    -> Weight 1.0
# Class 100 (25mm) -> Weight 3.0 
pixel_weights_np = 1.0 + (np.arange(NUM_CLASSES) * 0.02) 
# Cap at 3.0 
pixel_weights_np = np.clip(pixel_weights_np, a_min=None, a_max=3.0)

PIXEL_WEIGHTS = torch.tensor(pixel_weights_np, \
    device=DEVICE, dtype=torch.float32)

# --- 3. ASYMMETRY FACTOR ---
# Penalty multiplier for Under-prediction (Misses).
# 3.0 means missing a storm is 3x worse than a false alarm.
# this will prioritize resolution in CRPS over reliability.

ASYMMETRY_FACTOR = 1.25 # 3.0

TRAIN_DIR = '../resnet_data/trainings' 
DATA_DIR = '../resnet_data'

# ====================================================================

# --- UNet Architecture ---

class ResidualBlock(nn.Module):
    """
    True Residual Block with Identity Mapping.
    Structure: Output = ReLU( ConvBlock(x) + Shortcut(x) )
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
            kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
            kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Identity mapping shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)

# ------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Attention Gate to filter features from skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # W_g: Gate signal processing
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, 
            padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: Skip connection processing
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, 
            padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # psi: Join and rescale to 0-1
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, 
            padding=0, bias=True),
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

# ------------------------------------------------------------------

class AttnResUNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=NUM_CLASSES):
        super(AttnResUNet, self).__init__()
        self.in_channels = in_channels
        
        # --- Encoder ---
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))
        
        # --- Bridge ---
        self.bridge = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))
        
        # --- Decoder with Attention ---
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
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)
        
        # Decoder
        x = self.up1(x5) # x is the gating signal
        x4 = self.att1(g=x, x=x4) # filter the skip connection
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

# ------------------------------------------------------------------

# --- Dataset ---

class GRAF_Dataset(Dataset):
    """
    This loads the series of patched training data 
    """
    # CHANGE 1: Add 'train' argument to __init__
    def __init__(self, pickle_file, thresholds=THRESHOLDS, \
            normalization_stats=None, train=False):
        
        self.train = train  # Save the flag
        
        print(f"   Loading data from {pickle_file}...")
        # ... (Loading code remains the same) ...
        try:
            with open(pickle_file, 'rb') as f:
                self.graf = cPickle.load(f)
                self.mrms = cPickle.load(f)
                self.qual = cPickle.load(f) # Need this for masking.
                self.terdiff_graf = cPickle.load(f)
                self.diff = cPickle.load(f)
                self.dlon = cPickle.load(f)
                self.dlat = cPickle.load(f)
        except Exception as e:
            print(f"CRITICAL ERROR loading pickle: {e}")
            sys.exit(1)

        self.thresholds = np.array(thresholds)
        
        feature_list = [self.graf, self.terdiff_graf, self.diff, 
                       self.dlon, self.dlat]
        
        if normalization_stats is None:
            print("   Computing normalization stats from training data...")
            mins = [float(np.min(arr)) for arr in feature_list]
            maxs = [float(np.max(arr)) for arr in feature_list]
    
            # Feature 0: GRAF Precip (Standardized to 75mm)
            print(f"   Forcing Precip Max (idx 0) from {maxs[0]} to 75.0")
            maxs[0] = 75.0

            # Feature 1: Interaction (GRAF * TDiff)
            # Setting a safe upper bound prevents blowout.
            FORCED_MAX_INTERACTION = 35000.0 
            print(f"   Forcing Interaction Max (idx 1) from {maxs[1]} "
                  f"to {FORCED_MAX_INTERACTION}")
            maxs[1] = max(maxs[1], FORCED_MAX_INTERACTION)

            # Feature 2: TerrainDiff; Max local terrain diff. in CONUS 
            FORCED_MAX_TDIFF = 2500.0 
            print(f"   Forcing TerrainDiff Max (idx 2) from {maxs[2]} to "
                  f"{FORCED_MAX_TDIFF}")
            maxs[2] = max(maxs[2], FORCED_MAX_TDIFF)
    
            # Feature 3 & 4: Slopes
            FORCED_MAX_SLOPE = 0.02 
            print(f"   Forcing Slope Max (idx 3 & 4) to {FORCED_MAX_SLOPE}")
            maxs[3] = max(maxs[3], FORCED_MAX_SLOPE)
            maxs[4] = max(maxs[4], FORCED_MAX_SLOPE)
    
            self.stats = {'min': mins, 'max': maxs}
            print(f"   Stats Final: {self.stats}")
        else:
            self.stats = normalization_stats
            print("   Using provided normalization stats.")
    
        # --- Pre-normalization (In-Place to save RAM) ---
        print("   Pre-normalizing data in memory...")
        self.graf = self.normalize(self.graf, 0)
        self.terdiff_graf = self.normalize(self.terdiff_graf, 1)
        self.diff = self.normalize(self.diff, 2)
        self.dlon = self.normalize(self.dlon, 3)
        self.dlat = self.normalize(self.dlat, 4)

    def normalize(self, data, idx):
        """ Helper to normalize numpy arrays using stored stats """
        vmin = self.stats['min'][idx]
        vmax = self.stats['max'][idx]
        denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
        # Use float32 to save memory compared to float64
        return ((data - vmin) / denom).astype(np.float32)

    def __len__(self):
        return len(self.graf)
        
    def __getitem__(self, idx):
        # 1. Construct Input Tensor (x)
        x = np.stack([
            self.graf[idx],
            self.terdiff_graf[idx],
            self.diff[idx],
            self.dlon[idx],
            self.dlat[idx]
        ], axis=0)
    
        # 2. Construct Target Indices (y)
        y_raw = self.mrms[idx]
    
        # --- NEW: Retrieve Quality Mask ---
        # Assuming qual <= 0.01 is bad (matches your sampler logic)
        q_mask = self.qual[idx]
        is_bad = (q_mask <= 0.01)

        y_indices = np.searchsorted(self.thresholds, y_raw, side='right') - 1
        y_indices = np.clip(y_indices, 0, len(self.thresholds) - 2)
    
        # --- NEW: Apply Mask ---
        # Overwrite indices with -1 where data is bad.
        # The Loss function will now completely ignore these pixels.
        y_indices[is_bad] = -1
        
        # --- Random flipping so we don't overlearn to terrain
        
        if self.augment: 
            x, y_indices = self.apply_augmentation(x, y_indices)

        return torch.from_numpy(x), torch.from_numpy(y_indices).long()
        
    def apply_augmentation(self, x, y):
        
        """
        Applies random horizontal and vertical flips.
        x: (5, H, W) numpy array
           Channel 3: dlon (East-West slope)
           Channel 4: dlat (North-South slope)
        y: (H, W) target indices
        """
        # Random Horizontal Flip (Left-Right)
        if np.random.rand() > 0.5:
            # Flip pixels on width axis (axis 2)
            x = np.flip(x, axis=2) 
            y = np.flip(y, axis=1) # y is (H, W), so width is axis 1
            
            # NEGATE the dlon channel (Index 3)
            # Physical logic: Uphill East becomes Uphill West
            x[3, :, :] = -x[3, :, :]

        # Random Vertical Flip (Up-Down)
        if np.random.rand() > 0.5:
            # Flip pixels on height axis (axis 1)
            x = np.flip(x, axis=1) 
            y = np.flip(y, axis=0) # y is (H, W), so height is axis 0
            
            # NEGATE the dlat channel (Index 4)
            # Physical logic: Uphill North becomes Uphill South
            x[4, :, :] = -x[4, :, :]

        return x.copy(), y.copy()

    def __getitem__(self, idx):
        # 1. Construct Input Tensor (x)
        x = np.stack([
            self.graf[idx],
            self.terdiff_graf[idx],
            self.diff[idx],
            self.dlon[idx],
            self.dlat[idx]
        ], axis=0)
        
        # 2. Construct Target Indices (y)
        y_raw = self.mrms[idx]
        
        # --- Retrieve Quality Mask (Solution 1 from previous turn) ---
        q_mask = self.qual[idx]
        is_bad = (q_mask <= 0.01)

        y_indices = np.searchsorted(self.thresholds, y_raw, side='right') - 1
        y_indices = np.clip(y_indices, 0, len(self.thresholds) - 2)
        
        # Apply Mask (-1 ignores these pixels in loss)
        y_indices[is_bad] = -1
        
        # CHANGE 3: Conditionally Apply Augmentation
        if self.train:
            x, y_indices = self.apply_augmentation(x, y_indices)
        
        return torch.from_numpy(x), torch.from_numpy(y_indices).long()
            
# ------------------------------------------------------------------

# --- Loss Function (UPDATED) ---

class WeightedOrdinalWassersteinLoss(nn.Module):
    def __init__(self, num_classes, boundary_weights=None, 
                 class_weights=None, asymmetry_factor=1.0, 
                 ignore_index=-1, power=2.0):
        super(WeightedOrdinalWassersteinLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.power = power
        self.asymmetry_factor = asymmetry_factor
        
        if boundary_weights is not None:
            if not isinstance(boundary_weights, torch.Tensor):
                boundary_weights = torch.tensor(\
                    boundary_weights, dtype=torch.float32)
            self.register_buffer('boundary_weights', boundary_weights)
        else:
            self.register_buffer('boundary_weights', \
                torch.ones(num_classes - 1, dtype=torch.float32))

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) with values in [0, C-1] or ignore_index
        """

        B, C, H, W = logits.shape
        device = logits.device

        # ------------------------------------------------------------
        # 1. Predicted CDF
        # ------------------------------------------------------------
        probs = F.softmax(logits, dim=1)              
        pred_cdf = torch.cumsum(probs, dim=1)         

        # ------------------------------------------------------------
        # 2. Valid mask + safe targets (FIXED)
        # ------------------------------------------------------------
        valid_mask = (targets != self.ignore_index)
        
        # FIX: Clamp targets to valid range [0, C-1] to prevent index errors.
        # This handles negative ignore_indices AND wildly large garbage values.
        safe_targets = targets.clamp(0, C - 1) 

        # ------------------------------------------------------------
        # 3. Target CDF via broadcasting
        # ------------------------------------------------------------
        class_idx = torch.arange(
            C, device=device, dtype=safe_targets.dtype
        ).view(1, C, 1, 1)

        target_cdf = (class_idx >= safe_targets.unsqueeze(1)).float()

        # ------------------------------------------------------------
        # 4. Wasserstein distance 
        # ------------------------------------------------------------
        raw_diff = pred_cdf - target_cdf
        
        asym_weights = torch.ones_like(raw_diff)
        if self.asymmetry_factor != 1.0:
            asym_weights[raw_diff > 0] = self.asymmetry_factor
            
        diff = (torch.abs(raw_diff) ** self.power) * asym_weights

        w_bound = self.boundary_weights.view(1, C - 1, 1, 1)
        weighted_diff = diff[:, :-1, :, :] * w_bound

        pixel_loss = weighted_diff.sum(dim=1)         

        # ------------------------------------------------------------
        # 5. Optional class weighting (Sample Importance)
        # ------------------------------------------------------------
        if self.class_weights is not None:
            # This line caused the crash. safe_targets is now guaranteed safe.
            pixel_weights = self.class_weights[safe_targets]
            
            if self.ignore_index >= 0:
                # Zero out weights for ignored pixels
                pixel_weights = pixel_weights * valid_mask.float()
                
            pixel_loss = pixel_loss * pixel_weights

        # ------------------------------------------------------------
        # 6. Reduction
        # ------------------------------------------------------------
        if valid_mask.any():
            loss = pixel_loss.sum() / valid_mask.sum().clamp_min(1.0)
        else:
            loss = torch.zeros((), device=device, requires_grad=True)

        return loss
    
# ------------------------------------------------------------------

# --- DIAGNOSTIC PRINT FUNCTION  ---
#     once an epoch, it prints out some diagnostics that allow 
#     eyeball monitoring of changes in quality as it iterates.

def print_diagnostics(epoch, batch_idx, loss_val, outputs, \
        targets, model, stats):
        
    # 1. Save current mode and switch to eval for synthetic tests
    
    was_training = model.training
    model.eval() 

    with torch.no_grad():
        
        # --- REAL DATA STATS ---
        pdf = torch.softmax(outputs, dim=1)
        
        # --- SYNTHETIC TESTS ---
        def run_synthetic(precip_mm):
            # 1. Define physical values for the 5 features
            #    Feature 0: GRAF Precip
            #    Feature 1: GRAF * TerrainDiff (Assume 0 terrain diff)
            #    Feature 2: TerrainDiff
            #    Feature 3: dlon
            #    Feature 4: dlat
            
            p_val = precip_mm
            t_val = 0.0 # Average terrain diff
            
            f0 = p_val
            f1 = p_val * t_val 
            f2 = t_val
            f3 = 0.0
            f4 = 0.0
            
            phys_vals = [f0, f1, f2, f3, f4]
            
            # 2. Normalize
            norm_vals = []
            for i, val in enumerate(phys_vals):
                vmin = stats['min'][i]
                vmax = stats['max'][i]
                denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
                n_val = (val - vmin) / denom
                norm_vals.append(n_val)
            
            # 3. Create Tensor
            
            syn_x = torch.zeros((1, 5, 96, 96), device=DEVICE)
            for i, nv in enumerate(norm_vals):
                syn_x[:, i, :, :] = nv
            
            # Specify device_type='cuda' explicitly
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(syn_x)
                syn_p = torch.softmax(out, dim=1)
            
            return syn_p.mean(dim=(0,2,3)).cpu().numpy()

        syn_0mm_pdf = run_synthetic(0.0)
        syn_5mm_pdf = run_synthetic(5.0)

        # --- AGGREGATE STATS ---
        print(f"--- Epoch {epoch+1}, Batch {batch_idx} ---")
        print(f"Loss: {loss_val:.4f}")
        print(f"{'Classes':<10} | {'Range (mm)':<15} | {'Mean Prob':<9} | "
              f"{'Pred %':<6} | {'True %':<6} | {'Syn(0mm)':<8} | {'Syn(5mm)':<8}")
        print("-" * 80)
        
        def get_range_str(start_t_idx, end_t_idx):
            low = THRESHOLDS[start_t_idx] if start_t_idx < len(THRESHOLDS) else 200.0
            high = THRESHOLDS[end_t_idx] if end_t_idx < len(THRESHOLDS) else 200.0
            return f"{low:.2f} - {high:.2f}"

        # Target indices are passed directly now (no summation needed)
        target_indices = targets
        pred_indices = torch.argmax(pdf, dim=1)

        # 1. Class 0
        bin_prob = pdf[:, 0].mean().item()
        true_pct = (target_indices == 0).float().mean().item() * 100
        pred_pct = (pred_indices == 0).float().mean().item() * 100
        
        print(f"{'0':<10} | {'0.00':<15} | {bin_prob:.3f}     | "
              f"{pred_pct:6.3f} | {true_pct:6.3f} | {syn_0mm_pdf[0]:.3f}    "
              f"| {syn_5mm_pdf[0]:.3f}")

        # 2. Grouped Classes
        for i in range(1, NUM_CLASSES, 10):
            start = i
            end = min(i + 9, NUM_CLASSES - 1)
            
            bin_prob = pdf[:, start:end+1].sum(dim=1).mean().item()
            mask_true = (target_indices >= start) & (target_indices <= end)
            true_pct = mask_true.float().mean().item() * 100
            mask_pred = (pred_indices >= start) & (pred_indices <= end)
            pred_pct = mask_pred.float().mean().item() * 100
            
            syn0 = syn_0mm_pdf[start:end+1].sum()
            syn5 = syn_5mm_pdf[start:end+1].sum()
            label = f"{start}-{end}"
            rng = get_range_str(start - 1, end)
            
            print(f"{label:<10} | {rng:<15} | {bin_prob:.3f}     | "
                  f"{pred_pct:6.3f} | {true_pct:6.3f} | {syn0:.3f}    | {syn5:.3f}")

        print("-" * 80)

    # Restore training mode
    if was_training:
        model.train()


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# --- Main Training Loop ---

def train_model(date_str, lead_time_str):
    print("----------------------------------------------------------------")
    print(f"Starting Training for Date: {date_str}, Lead: {lead_time_str}")
    print(f"Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    print(f"Asymmetry Factor: {ASYMMETRY_FACTOR}")
    print("----------------------------------------------------------------")

    train_pattern = \
        f"{DATA_DIR}/GRAF_Unet_data_train_*{date_str}*_{lead_time_str}h.cPick*"
    val_pattern   = \
        f"{DATA_DIR}/GRAF_Unet_data_test_*{date_str}*_{lead_time_str}h.cPick*"
    
    try:
        train_files = glob.glob(train_pattern)
        val_files = glob.glob(val_pattern)
        
        if not train_files: \
            raise FileNotFoundError(f"No train files: {train_pattern}")
        if not val_files: \
            raise FileNotFoundError(f"No val files: {val_pattern}")

        train_file = train_files[0]
        val_file = val_files[0]
        print(f"Train File: {train_file}")
        print(f"Val File:   {val_file}")
        
    except Exception as e:
        print(f"ERROR Finding Files: {e}")
        sys.exit(1)
    
    # Pass train=True for training data.  This applies random 
    #    rotations so we don't overtrain to terrain info.
    train_dataset = GRAF_Dataset(train_file, THRESHOLDS, train=True)
    
    # Pass train=False for validation data.  Don't rotate.
    val_dataset = GRAF_Dataset(val_file, THRESHOLDS, \
        normalization_stats=train_dataset.stats, train=False)

    train_loader = DataLoader(train_dataset, \
        batch_size=BATCH_SIZE, shuffle=True, \
        num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, \
        batch_size=BATCH_SIZE, shuffle=False, \
        num_workers=NUM_WORKERS)
    
    model = AttnResUNet(in_channels=5, \
        num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() if USE_AMP else None

    # --- Initialize New Loss with Asymmetry and Weights ---
    criterion = WeightedOrdinalWassersteinLoss(
        num_classes=NUM_CLASSES, 
        boundary_weights=CLASS_WEIGHTS,
        class_weights=PIXEL_WEIGHTS, # Pass Pixel Weights
        asymmetry_factor=ASYMMETRY_FACTOR, # Pass Asymmetry
        ignore_index=-1
    ).to(DEVICE)

    # ====================================================================
    # --- RESUME LOGIC (Warm Start) ---
    # ====================================================================
    start_epoch = 0
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    # 1. Attempt to resume exact match (same lead time)
    resume_pattern = \
        f"{TRAIN_DIR}/resunet_ordinal_{date_str}_{lead_time_str}h_epoch_*.pth"
    existing_ckpts = glob.glob(resume_pattern)
    
    def get_epoch_from_filename(fname):
        match = re.search(r"_epoch_(\d+)\.pth", fname)
        return int(match.group(1)) if match else 0

    if existing_ckpts:
        print("   Found existing checkpoints for this lead time. Resuming...")
        existing_ckpts.sort(key=get_epoch_from_filename)
        latest_ckpt = existing_ckpts[-1]
        try:
            print(f"   Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"   Resuming from Epoch {start_epoch}")
        except Exception as e:
            print(f"   Error resuming: {e}. Starting fresh.")
            start_epoch = 0
    else:
        # 2. Attempt Warm Start from previous lead time (Lead - 3h)
        #    Note: We load weights, but start from Epoch 0 with fresh optimizer
        print("   No checkpoint. Checking previous lead (warm start)...")
        try:
            curr_lead = int(lead_time_str)
            prev_lead = curr_lead - 3
        except ValueError:
            prev_lead = -999
        
        warm_ckpts = []
        if prev_lead > 0:
            # Check both "3" and "03" formats to be robust
            possible_strs = [str(prev_lead), f"{prev_lead:02d}"]
            possible_strs = list(set(possible_strs)) # Dedup
            
            for p_str in possible_strs:
                 pat = f"{TRAIN_DIR}/resunet_ordinal_{date_str}_{p_str}h_epoch_*.pth"
                 warm_ckpts.extend(glob.glob(pat))
        
        if warm_ckpts:
            warm_ckpts.sort(key=get_epoch_from_filename)
            best_warm_ckpt = warm_ckpts[-1]
            print(f"   Found previous lead time checkpoint: {best_warm_ckpt}")
            try:
                checkpoint = torch.load(best_warm_ckpt, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("   Weights init. from previous lead. Start training from Epoch 0.")
            except Exception as e:
                print(f"   Error loading warm-start: {e}. Using random init.")
        else:
            print("   No previous lead time weights found. Using random init.")

    # ====================================================================

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # No summation needed; targets are already indices
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            if i % 250 == 0:
                print_diagnostics(epoch, i, loss.item(), outputs, \
                    targets, model, train_dataset.stats)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)                
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(inputs)
                    vloss = criterion(outputs, targets)
                val_loss += vloss.item()
        
        avg_train = running_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Finished. Train Loss: {avg_train:.4f}, "
              f"Val Loss: {avg_val:.4f}")

        stats_to_save = train_dataset.stats
        if stats_to_save is None:
            print("CRITICAL ERROR: Stats are None! Aborting save.")
            sys.exit(1)
            
        save_path = \
            f"{TRAIN_DIR}/resunet_ordinal_{date_str}_{lead_time_str}h_epoch_{epoch+1}.pth"
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train,
            'normalization_stats': stats_to_save
        }, save_path)
        print(f"   Saved {save_path}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("   Early stopping triggered.")
                break

if __name__ == "__main__":
    if len(sys.argv) == 3:
        date_str = sys.argv[1]
        lead_time = sys.argv[2]
    elif len(sys.argv) == 2:
        date_str = "*"
        lead_time = sys.argv[1]
    else:
        print("Usage: python pytorch_train_resunet.py <DATE> <LEAD>")
        sys.exit(1)
    
    train_model(date_str, lead_time)