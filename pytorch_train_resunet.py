"""
python pytorch_train_resunet.py 2025120100 12
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob
import re

"""
ResUNet Training Script for Precipitation Forecasting
=====================================================

Updates:
1.  **Early Stopping**: Logic restored. Stops if val_loss doesn't improve 
    for 10 epochs.
2.  **Learning Rate**: Fixed at 3e-4.
3.  **MPS/Mac Fix**: Explicitly casts numpy scalars to python floats.
4.  **Checkpoint Fix**: Allows loading numpy stats from checkpoint.
"""

# --- 1. Device Selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') 
    else:
        return torch.device('cpu')

DEVICE = get_device()

# --- Configuration ---
if DEVICE.type == 'cpu':
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    USE_AMP = False
else:
    BATCH_SIZE = 64      
    NUM_WORKERS = 0      
    USE_AMP = (DEVICE.type == 'cuda') 

LEARNING_RATE = 3e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10

# 101 Classes (0.0 to 25.0 by 0.25, plus >25.0)
THRESHOLDS = np.arange(0.0, 25.01, 0.25).tolist()
THRESHOLDS.append(200.0)
NUM_CLASSES = len(THRESHOLDS)-1
THRESHOLD_TENSOR = torch.tensor(THRESHOLDS[:-1], device=DEVICE, dtype=torch.float32)

TRAIN_DIR = 'trainings'

# --- 2. Loss Function ---
class WeightedOrdinalWassersteinLoss(nn.Module):
    def __init__(self, num_classes, boundary_weights=None, 
                 class_weights=None, ignore_index=-1, power=2.0):
        super(WeightedOrdinalWassersteinLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.power = power
        
        if boundary_weights is not None:
            if not isinstance(boundary_weights, torch.Tensor):
                boundary_weights = torch.tensor(boundary_weights, dtype=torch.float32)
            self.register_buffer('boundary_weights', boundary_weights)
        else:
            self.register_buffer('boundary_weights', torch.ones(num_classes - 1, dtype=torch.float32))

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        
        target_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float() 
        target_cdf = torch.cumsum(target_one_hot, dim=1)
        
        diff = torch.abs(pred_cdf - target_cdf) ** self.power
        w_bound = self.boundary_weights.view(1, -1, 1, 1)
        
        weighted_diff = diff[:, :-1, :, :] * w_bound
        pixel_loss = torch.sum(weighted_diff, dim=1)
        
        if self.class_weights is not None:
            pixel_weights = self.class_weights[safe_targets]
            if self.ignore_index >= 0:
                pixel_weights[~valid_mask] = 0.0
            pixel_loss = pixel_loss * pixel_weights

        if valid_mask.sum() > 0:
            if self.class_weights is not None:
                loss = pixel_loss.sum() / (pixel_weights.sum() + 1e-8)
            else:
                loss = pixel_loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return loss

# --- 3. Model Architecture (ResUNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=8):
        super(ResUNet, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.enc1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2) 
        self.bridge = ResidualBlock(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(512, 256) 
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128) 
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(128, 64) 
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.relu(self.bn1(self.init_conv(x)))
        x2 = self.enc1(x1); p2 = self.pool1(x2) 
        x3 = self.enc2(p2); p3 = self.pool2(x3) 
        x4 = self.enc3(p3); p4 = self.pool3(x4) 
        b = self.bridge(p4)
        u1 = self.up1(b); u1 = torch.cat([u1, x4], dim=1); d1 = self.dec1(u1)
        u2 = self.up2(d1); u2 = torch.cat([u2, x3], dim=1); d2 = self.dec2(u2)
        u3 = self.up3(d2); u3 = torch.cat([u3, x2], dim=1); d3 = self.dec3(u3)
        return self.out_conv(d3)

# --- 4. Dataset with Pre-Stack Normalization ---
class GRAF_Dataset(torch.utils.data.Dataset):
    def __init__(self, pickle_file, thresholds, normalization_stats=None):
        print(f"Loading {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            try:
                graf = pickle.load(f)      
                mrms = pickle.load(f)      
                _ = pickle.load(f)         
                terrain = pickle.load(f)   
                diff = pickle.load(f)      
                dlon = pickle.load(f)      
                dlat = pickle.load(f)      
                time = pickle.load(f)      
            except EOFError:
                print("Warning: EOF reached.")

        feature_list = [graf, terrain, diff, dlon, dlat, time]
        self.stats = {}

        if normalization_stats is None:
            print("Computing normalization stats from training data...")
            mins = [np.min(arr) for arr in feature_list]
            maxs = [np.max(arr) for arr in feature_list]
            self.stats = {'min': mins, 'max': maxs}
        else:
            print("Applying provided normalization stats...")
            self.stats = normalization_stats

        norm_features = []
        for i, arr in enumerate(feature_list):
            vmin = self.stats['min'][i]
            vmax = self.stats['max'][i]
            denom = vmax - vmin
            if denom == 0: denom = 1e-8
            arr_norm = (arr - vmin) / denom
            norm_features.append(arr_norm)
        
        n_graf, n_terrain, n_diff, n_dlon, n_dlat, n_time = norm_features

        print("Stacking features...")
        self.inputs = np.stack(
            [n_graf, n_terrain, n_diff, n_dlon, n_dlat, n_time, n_graf], axis=1
        ).astype(np.float32)
        
        mrms_indices = np.digitize(mrms, thresholds) - 1
        mrms_indices = np.clip(mrms_indices, 0, len(thresholds)-2)
        self.targets = mrms_indices.astype(np.int64)
        print(f"Loaded {self.inputs.shape[0]} samples.")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]), 
                torch.from_numpy(self.targets[idx]))

# --- 5. Utilities ---
def calculate_expected_value(probs):
    thresholds = torch.tensor(THRESHOLDS, device=DEVICE, dtype=torch.float32)
    centers = (thresholds[:-1] + thresholds[1:]) / 2.0
    centers[-1] = 30.0 
    centers = centers.view(1, -1, 1, 1)
    expected_mm = torch.sum(probs * centers, dim=1)
    return expected_mm

def normalize_value(value, channel_idx, stats):
    """ 
    Helper to normalize a single scalar value using dataset stats.
    Returns a standard python float to avoid MPS numpy incompatibilities.
    """
    vmin = stats['min'][channel_idx]
    vmax = stats['max'][channel_idx]
    denom = vmax - vmin
    if denom == 0: denom = 1e-8
    
    val = float(value)
    vm = float(vmin)
    dm = float(denom)
    
    return (val - vm) / dm

def print_diagnostics(model, loader, epoch, batch_idx, loss_val, norm_stats):
    """
    Prints diagnostic table.
    Requires 'norm_stats' to correctly construct the synthetic 5mm input.
    """
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        
        # RMSE
        pred_mm = calculate_expected_value(probs)
        thresholds_dev = torch.tensor(THRESHOLDS, device=DEVICE)
        centers_dev = (thresholds_dev[:-1] + thresholds_dev[1:]) / 2.0
        centers_dev[-1] = 30.0
        target_mm = centers_dev[targets]
        mse = F.mse_loss(pred_mm, target_mm)
        rmse = torch.sqrt(mse).item()
        
        print(f"--- Epoch {epoch}, Batch {batch_idx} ---")
        print(f"Loss (EMD): {loss_val:.4f}  |  RMSE: {rmse:.3f} mm")

        # Class stats
        probs_flat = probs.permute(1, 0, 2, 3).reshape(NUM_CLASSES, -1)
        targets_flat = targets.view(-1)
        preds_flat = torch.argmax(probs, dim=1).view(-1)
        
        mean_probs = probs_flat.mean(dim=1).cpu().numpy()
        total_pixels = targets_flat.numel()

        # --- Synthetic Inputs Construction ---
        B, C, H, W = inputs.shape
        
        # 1. Zero Rain Case (0mm Precip, Flat Terrain)
        input_syn_0 = torch.zeros((1, C, H, W), device=DEVICE, dtype=inputs.dtype)
        
        # Normalize scalar values and ensure they are floats for MPS
        norm_0_precip = normalize_value(0.0, 0, norm_stats)
        norm_0_terrain = normalize_value(0.0, 1, norm_stats) 
        norm_0_diff = normalize_value(0.0, 2, norm_stats) 
        norm_0_slope = normalize_value(0.0, 3, norm_stats) 
        
        input_syn_0[:, 0, :, :] = norm_0_precip # GRAF
        input_syn_0[:, 1, :, :] = norm_0_terrain
        input_syn_0[:, 2, :, :] = norm_0_diff
        input_syn_0[:, 3, :, :] = norm_0_slope # dlon
        input_syn_0[:, 4, :, :] = norm_0_slope # dlat
        input_syn_0[:, 6, :, :] = norm_0_precip # GRAF duplicate
        
        # 2. 5mm Rain Case (5mm Precip, Flat Terrain)
        input_syn_5 = input_syn_0.clone()
        norm_5_precip = normalize_value(5.0, 0, norm_stats)
        
        input_syn_5[:, 0, :, :] = norm_5_precip
        input_syn_5[:, 6, :, :] = norm_5_precip
        
        # Run Synthetic Inference
        logits_0 = model(input_syn_0)
        probs_0 = F.softmax(logits_0, dim=1)
        dist_0 = probs_0.mean(dim=(0, 2, 3)).cpu().numpy()
        
        logits_5 = model(input_syn_5)
        probs_5 = F.softmax(logits_5, dim=1)
        dist_5 = probs_5.mean(dim=(0, 2, 3)).cpu().numpy()
        
        def get_group_stats(indices):
            m_prob = np.sum(mean_probs[indices])
            p_pct = sum([(preds_flat == c).sum().item() for c in indices]) / total_pixels * 100
            t_pct = sum([(targets_flat == c).sum().item() for c in indices]) / total_pixels * 100
            
            s0_prob = np.sum(dist_0[indices])
            s5_prob = np.sum(dist_5[indices])
            
            low_mm = THRESHOLDS[indices[0]]
            high_mm = THRESHOLDS[indices[-1] + 1]
            return m_prob, p_pct, t_pct, low_mm, high_mm, s0_prob, s5_prob
        
        print(f"{'Classes':<10} | {'Range (mm)':<15} | {'Mean Prob':<9} | "
              f"{'Pred %':<6} | {'True %':<6} | {'Syn(0mm)':<8} | {'Syn(5mm)':<8}")
        print("-" * 96)
        
        mp, pp, tp, lo, hi, s0, s5 = get_group_stats([0])
        print(f"{'0':<10} | {lo:>5.2f} - {hi:<7.2f} | {mp:<9.3f} | "
              f"{pp:<6.3f} | {tp:<6.3f} | {s0:<8.3f} | {s5:<8.3f}")
        
        remaining_classes = list(range(1, NUM_CLASSES))
        chunk_size = 10
        for i in range(0, len(remaining_classes), chunk_size):
            chunk = remaining_classes[i:i + chunk_size]
            if not chunk: continue
            mp, pp, tp, lo, hi, s0, s5 = get_group_stats(chunk)
            label = f"{chunk[0]}-{chunk[-1]}"
            print(f"{label:<10} | {lo:>5.2f} - {hi:<7.2f} | {mp:<9.3f} | "
                  f"{pp:<6.3f} | {tp:<6.3f} | {s0:<8.3f} | {s5:<8.3f}")
        print("-" * 96)

    model.train()

# --- 6. Main ---
def main():
    if len(sys.argv) < 3:
        print("Usage: python pytorch_train_resunet.py <YYYYMMDDHH> <lead>")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    
    print(f"Run Configuration: Date {cyyyymmddhh}, Lead {clead}")

    train_pkl = f'../resnet_data/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick'
    val_pkl   = f'../resnet_data/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick'
    
    if not os.path.exists(train_pkl):
        print(f"Error: {train_pkl} not found")
        sys.exit(1)

    # 1. Load and Normalize Training Data
    print("Initializing Training Dataset...")
    train_dataset = GRAF_Dataset(train_pkl, THRESHOLDS, normalization_stats=None)
    
    # 2. Load Validation Data using Training Stats
    print("Initializing Validation Dataset...")
    val_dataset = GRAF_Dataset(val_pkl, THRESHOLDS, normalization_stats=train_dataset.stats)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    
    # 3. Calculate Loss Weights
    print("Calculating class weights...")
    all_targets = train_dataset.targets.flatten()
    counts = np.bincount(all_targets, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    
    weights = 0.2 + 0.8*(1.0 / np.sqrt(counts))
    weights = weights / np.mean(weights)
    
    boundary_weights_calc = weights[:-1].astype(np.float32)
    imbalance_weights_calc = weights.astype(np.float32)

    # 4. Model Initialization
    model = ResUNet(in_channels=7, num_classes=NUM_CLASSES).to(DEVICE)
    
    # --- RESUME LOGIC ---
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    file_pattern = f"resunet_ordinal_{cyyyymmddhh}_{clead}h_epoch_*.pth"
    pattern = os.path.join(TRAIN_DIR, file_pattern)
    existing_checkpoints = glob.glob(pattern)
    
    start_epoch = 0
    if existing_checkpoints:
        checkpoints_with_epochs = []
        for ckpt in existing_checkpoints:
            match = re.search(r"epoch_(\d+)\.pth", ckpt)
            if match:
                ep = int(match.group(1))
                checkpoints_with_epochs.append((ep, ckpt))
        
        if checkpoints_with_epochs:
            checkpoints_with_epochs.sort(key=lambda x: x[0], reverse=True)
            latest_epoch, latest_ckpt_path = checkpoints_with_epochs[0]
            
            print(f"Resuming training from Epoch {latest_epoch}")
            print(f"Loading weights: {latest_ckpt_path}")
            
            try:
                # FIX: Set weights_only=False to allow numpy scalars in normalization_stats
                checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                start_epoch = latest_epoch 
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")
    else:
        print("No previous checkpoints found for this date/lead. Starting fresh.")

    criterion = WeightedOrdinalWassersteinLoss(
        num_classes=NUM_CLASSES,
        boundary_weights=boundary_weights_calc,
        class_weights=imbalance_weights_calc, 
        power=2.0
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # Early Stopping Variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting Training on {DEVICE}...")

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print_diagnostics(
                    model, val_loader, epoch+1, batch_idx, loss.item(), 
                    train_dataset.stats
                )

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"=== Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} ===")
        
        # --- EARLY STOPPING CHECK ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Stopping training.")
                break
        
        # Save Checkpoint
        save_filename = f"resunet_ordinal_{cyyyymmddhh}_{clead}h_epoch_{epoch+1}.pth"
        save_path = os.path.join(TRAIN_DIR, save_filename)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalization_stats': train_dataset.stats,
            'args': {'date': cyyyymmddhh, 'lead': clead}
        }, save_path)
        
        print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()


