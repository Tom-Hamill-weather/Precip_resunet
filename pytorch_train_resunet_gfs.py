"""
pytorch_train_resunet_gfs.py

Usage example:

$ python pytorch_train_resunet_gfs.py 2025120100 12

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

Coded by Tom Hamill with Claude Code assistance.

Latest version: Adapted from pytorch_train_resunet.py to include GFS features

Input features (8 channels):
(1) GRAF precipitation forecast
(2) Terrain interaction (GRAF × terrain height difference)
(3) Local terrain height difference
(4) Terrain gradient (longitude direction)
(5) Terrain gradient (latitude direction)
(6) GFS precipitable water (PWAT)
(7) GFS column-average relative humidity
(8) GFS CAPE

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

# Patch size configuration for 96x96
PATCH_SIZE = 96

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if DEVICE.type == 'cpu':
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    USE_AMP = False
else:
    BATCH_SIZE = 128
    NUM_WORKERS = 5
    USE_AMP = (DEVICE.type == 'cuda')

BASE_LEARNING_RATE = 7.e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
THRESHOLDS.append(200.0)
NUM_CLASSES = len(THRESHOLDS)

THRESHOLD_TENSOR = torch.tensor(THRESHOLDS[:-1], device=DEVICE, dtype=torch.float32)

weights_np = np.diff(THRESHOLDS)
weights_np = np.clip(weights_np, a_min=None, a_max=5.0)
weights_np[0] = 1.0
CLASS_WEIGHTS = torch.tensor(weights_np, device=DEVICE, dtype=torch.float32)

pixel_weights_np = np.ones((NUM_CLASSES), dtype=np.float32)
pixel_weights_np = np.clip(pixel_weights_np, a_min=None, a_max=3.0)
PIXEL_WEIGHTS = torch.tensor(pixel_weights_np, device=DEVICE, dtype=torch.float32)

ASYMMETRY_FACTOR = 1.0
TRAIN_DIR = '../resnet_data/trainings'
DATA_DIR = '../resnet_data'

# ====================================================================
# --- Model Architecture ---
# ====================================================================

class ResidualBlock(nn.Module):
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
    def __init__(self, in_channels=8, num_classes=NUM_CLASSES):
        super(AttnResUNet, self).__init__()
        # Initial dimensions: (Batch, 8, 96, 96)
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
                self.gfs_pwat = cPickle.load(f)
                self.gfs_r = cPickle.load(f)
                self.gfs_cape = cPickle.load(f)
        except Exception as e:
            print(f"CRITICAL ERROR loading pickle: {e}")
            sys.exit(1)

        # Validation: check if the loaded arrays match 96x96
        if self.graf.shape[1] != PATCH_SIZE or self.graf.shape[2] != PATCH_SIZE:
             print(f"WARNING: Data shape {self.graf.shape} does not match PATCH_SIZE {PATCH_SIZE}")

        self.thresholds = np.array(thresholds)
        feature_list = [self.graf, self.terdiff_graf, self.diff, self.dlon, self.dlat,
                       self.gfs_pwat, self.gfs_r, self.gfs_cape]

        if normalization_stats is None:
            mins = [float(np.min(arr)) for arr in feature_list]
            maxs = [float(np.max(arr)) for arr in feature_list]
            # Original 5 features
            maxs[0] = 75.0          # GRAF precip
            maxs[1] = max(maxs[1], 35000.0)  # terrain interaction
            maxs[2] = max(maxs[2], 2500.0)   # terrain diff
            maxs[3] = max(maxs[3], 0.02)     # dlon
            maxs[4] = max(maxs[4], 0.02)     # dlat
            # GFS features
            maxs[5] = max(maxs[5], 70.0)     # PWAT (kg/m²)
            maxs[6] = max(maxs[6], 100.0)    # RH (%)
            maxs[7] = max(maxs[7], 5000.0)   # CAPE (J/kg)
            self.stats = {'min': mins, 'max': maxs}
        else:
            self.stats = normalization_stats

        self.graf = self.normalize(self.graf, 0)
        self.terdiff_graf = self.normalize(self.terdiff_graf, 1)
        self.diff = self.normalize(self.diff, 2)
        self.dlon = self.normalize(self.dlon, 3)
        self.dlat = self.normalize(self.dlat, 4)
        self.gfs_pwat = self.normalize(self.gfs_pwat, 5)
        self.gfs_r = self.normalize(self.gfs_r, 6)
        self.gfs_cape = self.normalize(self.gfs_cape, 7)

    def normalize(self, data, idx):
        vmin = self.stats['min'][idx]
        vmax = self.stats['max'][idx]
        denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
        return ((data - vmin) / denom).astype(np.float32)

    def __len__(self):
        return len(self.graf)

    def apply_augmentation(self, x, y):
        # Horizontal flip (negate dlon channel, which is channel 3)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2); y = np.flip(y, axis=1)
            x[3, :, :] = -x[3, :, :]
        # Vertical flip (negate dlat channel, which is channel 4)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1); y = np.flip(y, axis=0)
            x[4, :, :] = -x[4, :, :]
        return x.copy(), y.copy()

    def __getitem__(self, idx):
        x = np.stack([self.graf[idx], self.terdiff_graf[idx], self.diff[idx],
                     self.dlon[idx], self.dlat[idx],
                     self.gfs_pwat[idx], self.gfs_r[idx], self.gfs_cape[idx]], axis=0)
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
    print(f"  Input channels: 8 (GRAF + terrain features + GFS features)")
    print(f"  Output classes: {NUM_CLASSES}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = AttnResUNet(in_channels=8, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # --- Scheduler Integration ---
    # Patience: wait 2 epochs of no improvement, then cut LR by half
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    scaler = GradScaler() if USE_AMP else None
    criterion = WeightedOrdinalWassersteinLoss(num_classes=NUM_CLASSES, boundary_weights=CLASS_WEIGHTS, class_weights=PIXEL_WEIGHTS, asymmetry_factor=ASYMMETRY_FACTOR).to(DEVICE)

    start_epoch = 0
    if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)

    # Resume Logic - use _gfs suffix to distinguish from non-GFS models
    resume_pattern = f"{TRAIN_DIR}/resunet_ordinal_gfs_{date_str}_{lead_time_str}h_epoch_*.pth"
    existing_ckpts = sorted(glob.glob(resume_pattern), key=lambda x: int(re.search(r"_epoch_(\d+)\.pth", x).group(1)))

    if existing_ckpts:
        print(f"Found existing checkpoint: {existing_ckpts[-1]}")
        checkpoint = torch.load(existing_ckpts[-1], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"   Resuming from Epoch {start_epoch}\n")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training from epoch {start_epoch+1}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            amp_device = 'cuda' if USE_AMP else 'cpu'
            with torch.amp.autocast(amp_device, enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if USE_AMP:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                amp_device = 'cuda' if USE_AMP else 'cpu'
                with torch.amp.autocast(amp_device, enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)

        # --- Update Scheduler ---
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save Checkpoint
        save_path = f"{TRAIN_DIR}/resunet_ordinal_gfs_{date_str}_{lead_time_str}h_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val,
            'normalization_stats': train_dataset.stats
        }, save_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n   Early stopping triggered after {epoch+1} epochs.")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {TRAIN_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        train_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python pytorch_train_resunet_gfs.py <DATE> <LEAD>")
        print("Example: python pytorch_train_resunet_gfs.py 2025120100 12")
