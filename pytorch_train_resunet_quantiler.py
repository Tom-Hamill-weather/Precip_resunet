"""
pytorch_train_resunet_quantiler.py

Usage example:

$ python pytorch_train_resunet_quantiler.py 2025120100 12

where you supply the YYYYMMDDHH of initial condition and lead time in h.

This routine trains an Attention Residual U-Net for probabilistic precipitation
forecasting using QUANTILE REGRESSION instead of parametric distributions.

==============================================================================
MODEL OUTPUT: 13 Quantiles per pixel
==============================================================================

Instead of predicting distribution parameters, this model predicts precipitation
amounts at specific quantiles (percentiles):

Quantile levels: 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99

For each pixel, we get 13 precipitation values (in mm):
    q_0.01, q_0.05, q_0.10, ..., q_0.99

Examples:
- q_0.50 = 1.2mm means the median precipitation is 1.2mm
- q_0.90 = 5.0mm means 90% of the distribution is below 5mm (or 10% exceeds 5mm)
- If q_0.01 through q_0.30 all equal 0mm, this implies 30% probability of zero precip

==============================================================================
LOSS FUNCTION: Quantile Loss (Pinball Loss)
==============================================================================

For each quantile level τ ∈ [0, 1], the loss is:

    L_τ(y, q_τ) = {  τ × (y - q_τ)      if y ≥ q_τ
                  { (τ-1) × (y - q_τ)   if y < q_τ

Properties:
- Asymmetric: Over-prediction and under-prediction penalized differently
- When τ = 0.5 (median), reduces to mean absolute error
- Higher τ penalizes under-prediction more (important for extremes)
- Simple, differentiable everywhere, no special functions needed
- Works perfectly on MPS (no gradient issues)

Total loss = average over all 13 quantiles and all pixels

==============================================================================
MONOTONICITY ENFORCEMENT
==============================================================================

Quantiles must be monotonic: q_0.01 ≤ q_0.05 ≤ ... ≤ q_0.99

We enforce this using the cumulative sum trick:

    raw_outputs = model(x)                    # 13 unconstrained values
    deltas = softplus(raw_outputs)            # Force positive: Δ_i ≥ 0
    quantiles = cumsum(deltas, dim=1)         # Monotonic: q_i = Σ_{j≤i} Δ_j

This guarantees monotonicity by construction (no need for post-hoc sorting).

==============================================================================
ZERO INFLATION HANDLING
==============================================================================

Zero precipitation is naturally handled:
- If observed precipitation is frequently 0mm, the model learns to predict
  q_0.01 = q_0.05 = ... = q_0.30 = 0mm
- This implicitly represents P(X = 0) ≈ 0.30
- No special "fraction_zero" parameter needed

==============================================================================
ARCHITECTURE
==============================================================================

Same Attention Residual U-Net as other versions, but with 13 output
channels instead of 3 or 102.

Input features (7 channels):
(1) GRAF precipitation forecast
(2) Terrain elevation deviation (local terrain height difference)
(3) GFS column-average relative humidity
(4) Interaction: GRAF × terrain elevation deviation
(5) Interaction: GRAF × GFS relative humidity
(6) Terrain gradient (longitude direction)
(7) Terrain gradient (latitude direction)

Output: 13 channels representing quantiles at levels:
        [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

==============================================================================
ADVANTAGES OVER GAMMA MODEL
==============================================================================

1. **No gradient issues**: Pure arithmetic, works perfectly on MPS
2. **Simpler loss**: No lgamma, no exponentials, just max/min operations
3. **Flexible distribution**: Not constrained to Gamma shape
4. **Natural extreme handling**: High quantiles (0.95, 0.99) directly model tails
5. **Faster training**: Simpler loss function, faster backward pass

==============================================================================
INFERENCE
==============================================================================

At inference, to compute P(X > threshold):
1. Get 13 predicted quantiles for the pixel
2. Find quantiles bracketing the threshold
3. Interpolate to find percentile corresponding to threshold
4. P(X > threshold) = 1 - percentile

Example: If threshold = 2.5mm and model predicts:
    q_0.60 = 2.0mm, q_0.70 = 3.0mm
Then by interpolation, 2.5mm ≈ 65th percentile
So P(X > 2.5mm) ≈ 1 - 0.65 = 0.35 (35%)

==============================================================================
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
from configparser import ConfigParser

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

# Quantile levels to predict (13 quantiles)
QUANTILE_LEVELS = torch.tensor([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
                                dtype=torch.float32)
NUM_QUANTILES = len(QUANTILE_LEVELS)

# Device selection (MPS > CUDA > CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Running on: mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on: cuda")
else:
    DEVICE = torch.device("cpu")
    print("Running on: cpu")

# Move quantile levels to device
QUANTILE_LEVELS = QUANTILE_LEVELS.to(DEVICE)

# Hyperparameters
PATCH_SIZE = 96
BATCH_SIZE = 128 if DEVICE.type in ['cuda', 'mps'] else 16
LEARNING_RATE = 1.5e-3
MAX_EPOCHS = 30
PATIENCE = 5

# ---------------------------------------------------------------
# NEURAL NETWORK BUILDING BLOCKS
# ---------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections."""
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
    """Attention Residual U-Net for shifted quantile regression."""
    def __init__(self, in_channels=7, num_outputs=NUM_QUANTILES+1):
        super(AttnResUNet, self).__init__()

        # Encoder
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))

        # Bridge
        self.bridge = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))

        # Decoder with attention gates
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResidualBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec4 = ResidualBlock(128, 64)

        # Output layer: 1 shift + 13 quantiles = 14 channels
        self.outc = nn.Conv2d(64, num_outputs, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bridge
        x5 = self.bridge(x4)

        # Decoder with attention
        x = self.up1(x5)
        x4_att = self.att1(g=x, x=x4)
        x = torch.cat([x4_att, x], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x3_att = self.att2(g=x, x=x3)
        x = torch.cat([x3_att, x], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x2_att = self.att3(g=x, x=x2)
        x = torch.cat([x2_att, x], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x1_att = self.att4(g=x, x=x1)
        x = torch.cat([x1_att, x], dim=1)
        x = self.dec4(x)

        # Output: raw logits (unconstrained)
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------------
# LOSS FUNCTION
# ---------------------------------------------------------------

class ShiftedQuantileLoss(nn.Module):
    """
    Shifted Quantile Loss for censored quantile regression.

    For quantile level τ:
        L_τ(y, Q_τ) = max(τ × (y - Q_τ), (τ-1) × (y - Q_τ))

    where Q_τ = max(0, q_τ + s) are the censored quantiles.

    Includes regularization on shift parameter to prevent extreme values.
    """
    def __init__(self, quantile_levels, ignore_index=-1, shift_reg=0.01):
        super(ShiftedQuantileLoss, self).__init__()
        self.quantile_levels = quantile_levels  # [Q] tensor on device
        self.ignore_index = ignore_index
        self.shift_reg = shift_reg
        self.num_quantiles = len(quantile_levels)

    def forward(self, shift, quantile_preds, targets):
        """
        Args:
            shift: [B, 1, H, W] shift parameter
            quantile_preds: [B, Q, H, W] predicted censored quantiles
            targets: [B, H, W] target precipitation values

        Returns:
            loss: scalar
        """
        B, Q, H, W = quantile_preds.shape

        # Expand targets to match quantile dimension
        targets_expanded = targets.unsqueeze(1).expand(-1, Q, -1, -1)  # [B, Q, H, W]

        # Mask for valid pixels (not missing data)
        valid_mask = (targets != self.ignore_index)  # [B, H, W]
        valid_mask = valid_mask.unsqueeze(1).expand(-1, Q, -1, -1)  # [B, Q, H, W]

        # Compute errors: y - Q_τ
        errors = targets_expanded - quantile_preds  # [B, Q, H, W]

        # Quantile loss: max(τ × error, (τ-1) × error)
        tau = self.quantile_levels.view(1, Q, 1, 1)
        loss_quantile = torch.max(tau * errors, (tau - 1.0) * errors)  # [B, Q, H, W]

        # Apply mask
        loss_quantile = loss_quantile * valid_mask.float()

        # Mean over valid entries
        num_valid = valid_mask.float().sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=quantile_preds.device)

        loss_q = loss_quantile.sum() / num_valid

        # Shift regularization: discourage extreme shifts
        # Use L2 penalty on shift values
        valid_mask_shift = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
        shift_masked = shift * valid_mask_shift.float()
        num_valid_shift = valid_mask_shift.float().sum()

        if num_valid_shift > 0:
            loss_shift = self.shift_reg * (shift_masked ** 2).sum() / num_valid_shift
        else:
            loss_shift = torch.tensor(0.0, device=shift.device)

        return loss_q + loss_shift

# ---------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------

class PrecipitationDataset(Dataset):
    """Dataset for precipitation forecasting with 7-channel inputs."""
    def __init__(self, X, y, normalization_stats):
        """
        Args:
            X: [N, 7, H, W] features
            y: [N, H, W] targets
            normalization_stats: dict with 'min' and 'max' for each channel
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.normalization_stats = normalization_stats

        # Normalize features using min-max normalization: (x - min) / (max - min)
        mins = normalization_stats['min']
        maxes = normalization_stats['max']
        for i in range(7):
            vmin = mins[i]
            vmax = maxes[i]
            denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
            self.X[:, i, :, :] = (self.X[:, i, :, :] - vmin) / denom

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------

def enforce_monotonicity(raw_outputs, max_precip=150.0, max_shift=20.0, debug=False):
    """
    Shifted quantile regression: extract shift and compute censored quantiles.

    Args:
        raw_outputs: [B, 14, H, W] unconstrained model outputs (1 shift + 13 quantiles)
        max_precip: maximum precipitation value in mm (for scaling)
        max_shift: maximum absolute shift value in mm
        debug: if True, print shape diagnostics

    Returns:
        shift: [B, 1, H, W] shift parameter in mm (can be negative)
        quantiles: [B, 13, H, W] censored quantiles in mm, Q = max(0, q + shift)
    """
    if debug:
        print(f"  raw_outputs.shape: {raw_outputs.shape}")

    # Split outputs
    shift_raw = raw_outputs[:, 0:1, :, :]       # [B, 1, H, W]
    quantiles_raw = raw_outputs[:, 1:, :, :]    # [B, 13, H, W]

    if debug:
        print(f"  shift_raw.shape: {shift_raw.shape}")
        print(f"  quantiles_raw.shape: {quantiles_raw.shape}")

    # Shift parameter: use tanh to bound to [-max_shift, +max_shift]
    # Initialize bias to produce slightly negative shifts on average for dry forecasts
    shift = max_shift * torch.tanh(shift_raw)   # [B, 1, H, W]

    # Build monotonic underlying quantiles (before censoring)
    # Use ReLU for first increment to allow zeros
    q_first = F.relu(quantiles_raw[:, 0:1, :, :])  # [B, 1, H, W]

    # Use softplus for subsequent increments (smooth, always positive)
    if quantiles_raw.shape[1] > 1:
        deltas_rest = F.softplus(quantiles_raw[:, 1:, :, :])  # [B, 12, H, W]
        all_deltas = torch.cat([q_first, deltas_rest], dim=1)  # [B, 13, H, W]
        cumulative = torch.cumsum(all_deltas, dim=1)  # [B, 13, H, W]

        if debug:
            print(f"  q_first.shape: {q_first.shape}")
            print(f"  deltas_rest.shape: {deltas_rest.shape}")
            print(f"  all_deltas.shape: {all_deltas.shape}")
            print(f"  cumulative.shape: {cumulative.shape}")
    else:
        cumulative = q_first

    # Apply saturation using rational function
    q_underlying = max_precip * cumulative / (cumulative + max_precip)

    # Apply shift and censoring: Q = max(0, q + s)
    quantiles = torch.clamp(q_underlying + shift, min=0.0)  # [B, 13, H, W]

    if debug:
        print(f"  q_underlying.shape: {q_underlying.shape}")
        print(f"  quantiles.shape: {quantiles.shape}")

    return shift, quantiles

def read_config_file(config_file):
    """Read training data directory from config file."""
    config_object = ConfigParser()
    config_object.read(config_file)
    params = config_object['PARAMETERS']
    ndays_train = int(params['ndays_train'])
    return ndays_train

def load_training_data(cyyyymmddhh, clead):
    """Load training and validation data from pickle files."""
    train_file = f'../resnet_data/trainings/GRAF_Unet_data_train_{cyyyymmddhh}_{clead}h.cPick'
    val_file = f'../resnet_data/trainings/GRAF_Unet_data_test_{cyyyymmddhh}_{clead}h.cPick'

    print(f'\nLoading training data from: {train_file}')
    if not os.path.exists(train_file):
        print(f'ERROR: Training file not found: {train_file}')
        sys.exit(1)

    # Load sequential pickle dumps
    with open(train_file, 'rb') as f:
        graf_train = cPickle.load(f)
        mrms_train = cPickle.load(f)
        qual_train = cPickle.load(f)
        terdiff_graf_train = cPickle.load(f)
        diff_train = cPickle.load(f)
        dlon_train = cPickle.load(f)
        dlat_train = cPickle.load(f)
        init_times_train = cPickle.load(f)
        valid_times_train = cPickle.load(f)
        gfs_pwat_train = cPickle.load(f)
        gfs_r_train = cPickle.load(f)
        gfs_cape_train = cPickle.load(f)

    print(f'Loading validation data from: {val_file}')
    if not os.path.exists(val_file):
        print(f'ERROR: Validation file not found: {val_file}')
        sys.exit(1)

    with open(val_file, 'rb') as f:
        graf_val = cPickle.load(f)
        mrms_val = cPickle.load(f)
        qual_val = cPickle.load(f)
        terdiff_graf_val = cPickle.load(f)
        diff_val = cPickle.load(f)
        dlon_val = cPickle.load(f)
        dlat_val = cPickle.load(f)
        init_times_val = cPickle.load(f)
        valid_times_val = cPickle.load(f)
        gfs_pwat_val = cPickle.load(f)
        gfs_r_val = cPickle.load(f)
        gfs_cape_val = cPickle.load(f)

    # Build feature arrays (7 channels)
    # Compute GRAF × RH interaction
    graf_rh_train = graf_train * gfs_r_train
    graf_rh_val = graf_val * gfs_r_val

    # Stack features: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
    X_train = np.stack([graf_train, diff_train, gfs_r_train,
                        terdiff_graf_train, graf_rh_train,
                        dlon_train, dlat_train], axis=1)
    X_val = np.stack([graf_val, diff_val, gfs_r_val,
                      terdiff_graf_val, graf_rh_val,
                      dlon_val, dlat_val], axis=1)

    # Targets
    y_train = mrms_train
    y_val = mrms_val

    # Mask low-quality data
    y_train = np.where(qual_train <= 0.01, -1, y_train)
    y_val = np.where(qual_val <= 0.01, -1, y_val)

    # Compute normalization statistics from training data
    # Channel order: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
    mins = [0.0, 0.0, 0.0, 0.0, 0.0, -0.02, -0.02]
    maxes = [75.0, 2500.0, 100.0, 35000.0, 7500.0, 0.02, 0.02]
    normalization_stats = {'min': mins, 'max': maxes}

    train_data = {'X': X_train, 'y': y_train, 'normalization_stats': normalization_stats}
    val_data = {'X': X_val, 'y': y_val}

    return train_data, val_data

def print_diagnostics(epoch, batch_idx, loss_val, quantile_preds, targets):
    """
    Print training diagnostics including quantile predictions.

    Args:
        epoch: current epoch
        batch_idx: current batch
        loss_val: current loss value
        quantile_preds: [B, Q, H, W] predicted quantiles
        targets: [B, H, W] target values
    """
    valid_mask = targets != -1

    if not valid_mask.any():
        return

    # Extract valid targets
    valid_targets = targets[valid_mask].cpu().numpy()

    print(f'\n{"="*70}')
    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss_val:.6f}')
    print(f'{"="*70}')
    print(f'\nTarget (MRMS) Statistics:')
    print(f'  Mean:   {valid_targets.mean():.3f} mm')
    print(f'  Median: {np.median(valid_targets):.3f} mm')
    print(f'  Max:    {valid_targets.max():.3f} mm')
    print(f'  P(X=0): {100*(valid_targets==0).mean():.1f}%')

    # Check monotonicity violations (should be zero)
    violations = 0
    for b in range(quantile_preds.shape[0]):
        for h in range(quantile_preds.shape[2]):
            for w in range(quantile_preds.shape[3]):
                q_vals = quantile_preds[b, :, h, w].cpu().numpy()
                if not np.all(np.diff(q_vals) >= -1e-6):  # Allow small numerical errors
                    violations += 1

    if violations > 0:
        print(f'\nWARNING: {violations} monotonicity violations detected!')
    else:
        print(f'\nMonotonicity: OK (all quantiles properly ordered)')

    # Synthetic tests: What does model predict for specific inputs?
    print(f'\n{"="*70}')
    print(f'SYNTHETIC TESTS: Model predictions for idealized conditions')
    print(f'{"="*70}')

    # Test 1: Dry forecast (0mm GRAF, low RH = 20%)
    print(f'\n[TEST 1] DRY FORECAST: 0mm GRAF precipitation, 20% RH')
    print(f'-' * 70)

    graf_val = 0.0
    rh_val = 20.0
    syn_input = torch.zeros(1, 7, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
    syn_input[0, 0, :, :] = graf_val / 75.0  # GRAF precip (normalized by 75mm)
    syn_input[0, 2, :, :] = rh_val / 100.0  # RH (normalized by 100%)
    syn_input[0, 4, :, :] = (graf_val * rh_val) / 7500.0  # GRAF × RH interaction
    # Other channels (terrain, interactions, gradients) remain zero

    model.eval()  # Set to eval mode for stable batch norm statistics
    with torch.no_grad():
        syn_logits = model(syn_input)
        syn_shift, syn_quantiles = enforce_monotonicity(syn_logits)
        syn_s = syn_shift[0, 0, PATCH_SIZE//2, PATCH_SIZE//2].cpu().item()
        syn_q = syn_quantiles[0, :, PATCH_SIZE//2, PATCH_SIZE//2].cpu().numpy()
    model.train()  # Set back to training mode

    print(f'Predicted Shift: s = {syn_s:6.3f} mm')
    print(f'\nPredicted Quantiles (censored, Q = max(0, q + s)):')
    for i in range(NUM_QUANTILES):
        level = QUANTILE_LEVELS[i].item()
        val = syn_q[i]
        print(f'  Q_{level:.2f} = {val:6.3f} mm', end='')
        if (i+1) % 3 == 0:
            print()  # New line every 3 quantiles

    # Compute probabilities for standard thresholds
    def compute_prob(q_vals, threshold):
        """Interpolate to get P(X > threshold)."""
        if threshold < q_vals[0]:
            return 1.0
        if threshold > q_vals[-1]:
            return 0.0
        idx = np.searchsorted(q_vals, threshold)
        if idx == 0:
            return 1.0 - QUANTILE_LEVELS[0].item()
        q_low, q_high = QUANTILE_LEVELS[idx-1].item(), QUANTILE_LEVELS[idx].item()
        v_low, v_high = q_vals[idx-1], q_vals[idx]
        if abs(v_high - v_low) < 1e-8:
            percentile = q_low
        else:
            percentile = q_low + (q_high - q_low) * (threshold - v_low) / (v_high - v_low)
        return 1.0 - percentile

    print(f'\nDerived Probabilities:')
    print(f'  P(X ≥ 0.25mm) = {100*compute_prob(syn_q, 0.25):5.1f}%')
    print(f'  P(X ≥ 1.0mm)  = {100*compute_prob(syn_q, 1.0):5.1f}%')
    print(f'  P(X ≥ 2.5mm)  = {100*compute_prob(syn_q, 2.5):5.1f}%')
    print(f'  P(X ≥ 5.0mm)  = {100*compute_prob(syn_q, 5.0):5.1f}%')

    # Test 2: Light rain forecast (1mm GRAF, high RH = 80%)
    print(f'\n[TEST 2] LIGHT RAIN FORECAST: 1mm GRAF precipitation, 80% RH')
    print(f'-' * 70)

    graf_val = 1.0
    rh_val = 80.0
    syn_input[0, 0, :, :] = graf_val / 75.0  # GRAF precip = 1mm
    syn_input[0, 2, :, :] = rh_val / 100.0  # RH = 80%
    syn_input[0, 4, :, :] = (graf_val * rh_val) / 7500.0  # GRAF × RH interaction

    model.eval()  # Set to eval mode for stable batch norm statistics
    with torch.no_grad():
        syn_logits = model(syn_input)
        syn_shift, syn_quantiles = enforce_monotonicity(syn_logits)
        syn_s = syn_shift[0, 0, PATCH_SIZE//2, PATCH_SIZE//2].cpu().item()
        syn_q = syn_quantiles[0, :, PATCH_SIZE//2, PATCH_SIZE//2].cpu().numpy()
    model.train()  # Set back to training mode

    print(f'Predicted Shift: s = {syn_s:6.3f} mm')
    print(f'\nPredicted Quantiles (censored, Q = max(0, q + s)):')
    for i in range(NUM_QUANTILES):
        level = QUANTILE_LEVELS[i].item()
        val = syn_q[i]
        print(f'  Q_{level:.2f} = {val:6.3f} mm', end='')
        if (i+1) % 3 == 0:
            print()

    print(f'\nDerived Probabilities:')
    print(f'  P(X ≥ 0.25mm) = {100*compute_prob(syn_q, 0.25):5.1f}%')
    print(f'  P(X ≥ 1.0mm)  = {100*compute_prob(syn_q, 1.0):5.1f}%')
    print(f'  P(X ≥ 2.5mm)  = {100*compute_prob(syn_q, 2.5):5.1f}%')
    print(f'  P(X ≥ 5.0mm)  = {100*compute_prob(syn_q, 5.0):5.1f}%')

    print(f'\n{"="*70}\n')

# ---------------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python pytorch_train_resunet_quantiler.py <YYYYMMDDHH> <lead>")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    il = int(clead)

    print(f'\n{"="*70}')
    print(f'Training Quantile Regression U-Net')
    print(f'Initial Condition: {cyyyymmddhh}')
    print(f'Lead Time: {clead} hours')
    print(f'Device: {DEVICE}')
    print(f'Quantile Levels: {QUANTILE_LEVELS.cpu().numpy()}')
    print(f'{"="*70}\n')

    # Read configuration
    config_file = 'config_laptop.ini'
    ndays_train = read_config_file(config_file)
    print(f'Training days: {ndays_train}')

    # Adjust learning rate for longer lead times
    if il >= 12:
        LEARNING_RATE = LEARNING_RATE * 0.7
        print(f'Adjusted learning rate for {il}h lead time: {LEARNING_RATE:.6f}')

    # Load data
    train_data, val_data = load_training_data(cyyyymmddhh, clead)

    X_train = train_data['X']
    y_train = train_data['y']
    normalization_stats = train_data['normalization_stats']

    X_val = val_data['X']
    y_val = val_data['y']

    print(f'\nTraining samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Patch size: {PATCH_SIZE}x{PATCH_SIZE}')
    print(f'Input channels: 7')
    print(f'Output quantiles: {NUM_QUANTILES}')

    # Create datasets
    train_dataset = PrecipitationDataset(X_train, y_train, normalization_stats)
    val_dataset = PrecipitationDataset(X_val, y_val, normalization_stats)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = AttnResUNet(in_channels=7, num_outputs=NUM_QUANTILES+1).to(DEVICE)  # +1 for shift parameter

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel parameters: {total_params:,} (trainable: {trainable_params:,})')

    # Loss and optimizer
    criterion = ShiftedQuantileLoss(QUANTILE_LEVELS, ignore_index=-1, shift_reg=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2
    )

    # Check for existing checkpoints
    checkpoint_dir = '../resnet_data/trainings'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_pattern = f'resunet_quantiler_{cyyyymmddhh}_{clead}h_epoch_*.pth'
    import glob
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))

    start_epoch = 0
    best_val_loss = float('inf')

    if existing_checkpoints:
        # Find latest epoch
        epoch_nums = [int(f.split('epoch_')[1].split('.pth')[0]) for f in existing_checkpoints]
        latest_epoch = max(epoch_nums)
        latest_checkpoint = os.path.join(checkpoint_dir,
            f'resunet_quantiler_{cyyyymmddhh}_{clead}h_epoch_{latest_epoch}.pth')

        print(f'\nFound existing checkpoint: {latest_checkpoint}')
        print(f'Resuming from epoch {latest_epoch}...')

        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        print(f'Resumed: epoch={start_epoch}, best_val_loss={best_val_loss:.6f}')

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            raw_outputs = model(inputs)

            # Enforce monotonicity and apply shift
            # Debug first batch only
            debug_flag = (epoch == start_epoch and batch_idx == 0)
            if debug_flag:
                print(f"\n=== DEBUG: First batch shapes ===")
            shift, quantile_preds = enforce_monotonicity(raw_outputs, debug=debug_flag)
            if debug_flag:
                print(f"  shift.shape (returned): {shift.shape}")
                print(f"  quantile_preds.shape (returned): {quantile_preds.shape}")
                print(f"  targets.shape: {targets.shape}")
                print(f"=================================\n")

            # Compute loss
            loss = criterion(shift, quantile_preds, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print diagnostics every 100 batches
            if batch_idx % 100 == 0:
                print_diagnostics(epoch, batch_idx, loss.item(),
                                quantile_preds.detach(), targets.detach())

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                raw_outputs = model(inputs)
                shift, quantile_preds = enforce_monotonicity(raw_outputs)
                loss = criterion(shift, quantile_preds, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{MAX_EPOCHS}')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss:   {val_loss:.6f}')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'{"="*70}\n')

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir,
            f'resunet_quantiler_{cyyyymmddhh}_{clead}h_epoch_{epoch+1}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'normalization_stats': normalization_stats,
            'quantile_levels': QUANTILE_LEVELS.cpu().numpy(),
        }, checkpoint_path)

        print(f'Saved checkpoint: {checkpoint_path}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_path = os.path.join(checkpoint_dir,
                f'resunet_quantiler_{cyyyymmddhh}_{clead}h_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'normalization_stats': normalization_stats,
                'quantile_levels': QUANTILE_LEVELS.cpu().numpy(),
            }, best_path)

            print(f'*** New best model! Val loss: {val_loss:.6f} ***')
            print(f'Saved: {best_path}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{PATIENCE}')

            if patience_counter >= PATIENCE:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

    print(f'\n{"="*70}')
    print(f'Training Complete!')
    print(f'Best Validation Loss: {best_val_loss:.6f}')
    print(f'{"="*70}\n')
