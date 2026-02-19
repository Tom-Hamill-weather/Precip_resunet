"""
pytorch_train_resunet_gamma.py

Usage example:

$ python pytorch_train_resunet_gamma.py 2025120100 12

where you supply the YYYYMMDDHH of initial condition and lead time in h.

This routine trains an Attention Residual U-Net for probabilistic precipitation
forecasting using a ZERO-INFLATED GAMMA MIXTURE MODEL instead of categorical
probabilities. This is a more parsimonious and theoretically motivated approach.

==============================================================================
MODEL OUTPUT: 3 Parameters per pixel
==============================================================================

Instead of predicting 102 categorical probabilities, this model predicts:

(1) fraction_zero (p₀): Probability of exactly zero precipitation [0, 1]
(2) shape (α): Gamma distribution shape parameter (α > 0)
(3) scale (θ): Gamma distribution scale parameter (θ > 0)

For positive precipitation amounts, the distribution is Gamma(α, θ) with:
- Mean = α × θ
- Variance = α × θ²
- PDF(x) = (1/Γ(α)) × (x/θ)^(α-1) × exp(-x/θ) / θ

==============================================================================
LOSS FUNCTION: Negative Log-Likelihood for Zero-Inflated Gamma
==============================================================================

The Negative Log-Likelihood (NLL) is a proper scoring rule that measures
how well the predicted distribution explains the observation. Unlike CRPS,
NLL only requires the PDF (not CDF), making it computationally simpler with
full gradient support.

**Case 1: Observation y = 0**
    NLL(p₀, α, θ | y=0) = -log(p₀)

    Interpretation: Penalize low probability assigned to zero when zero is observed.

**Case 2: Observation y > 0**
    NLL(p₀, α, θ | y>0) = -log(1 - p₀) + NLL_gamma(α, θ | y)

    where:
    NLL_gamma(α, θ | y) = lgamma(α) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)

    This uses torch.lgamma which has full gradient support.

    Interpretation: The first term penalizes forecasting zero when observing
    precipitation. The second term is the NLL for the Gamma component.

==============================================================================
PARAMETER CONSTRAINTS
==============================================================================

The neural network outputs 3 unconstrained values which are transformed:

    fraction_zero = sigmoid(output[0])           → [0, 1]
    shape = shape_min + softplus(output[1])      → [shape_min, ∞)
    scale = scale_min + softplus(output[2])      → [scale_min, ∞)

where softplus(x) = log(1 + exp(x)) provides smooth, always-positive outputs.

The minimum bounds (shape_min, scale_min) are computed from training data
climatology to ensure numerical stability.

==============================================================================
INITIALIZATION STRATEGY
==============================================================================

The final layer is initialized using climatology from MRMS training data:

1. Compute fraction of zero pixels → initialize bias[0] = logit(fraction_zero)
2. Fit Gamma to wet pixels using Method of Moments (Thom's estimators):
   - Sample mean: μ = mean(wet_pixels)
   - Sample variance: σ² = var(wet_pixels)
   - Shape: α = μ² / σ²
   - Scale: θ = σ² / μ
3. Initialize bias[1] = inverse_softplus(α - shape_min)
4. Initialize bias[2] = inverse_softplus(θ - scale_min)

This ensures the model starts with climatologically reasonable predictions.

==============================================================================
ARCHITECTURE
==============================================================================

Same Attention Residual U-Net as categorical version, but with 3 output
channels instead of 102.

Input features (7 channels):
(1) GRAF precipitation forecast
(2) Terrain elevation deviation (local terrain height difference)
(3) GFS column-average relative humidity
(4) Interaction: GRAF × terrain elevation deviation
(5) Interaction: GRAF × GFS relative humidity
(6) Terrain gradient (longitude direction)
(7) Terrain gradient (latitude direction)

Coded by Tom Hamill with Claude Code assistance, February 2025
"""

import os
import sys
import glob
import re
import _pickle as cPickle
import numpy as np
from scipy import special, stats as scipy_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

# Enable CPU fallback for MPS unsupported operations
# This allows Gamma sampling/CDF to work, but those specific ops will run on CPU
# The rest (UNet forward/backward) still runs on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    USE_AMP = (DEVICE.type == 'cuda')

# Gradient accumulation to simulate larger effective batch size
ACCUMULATION_STEPS = 8  # Effective batch size = 16 * 8 = 128

# --- 3. Training Hyperparameters ---

PATCH_SIZE = 96
BASE_LEARNING_RATE = 9.e-4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 3

# --- 4. Loss Weighting (Initially disabled for unweighted NLL) ---

USE_WEIGHTED_LOSS = False
WEIGHT_BY_OBSERVATION = False  # If True, multiply NLL by f(observed_value)

_trainings_abs = '/data2/resnet_data/trainings'
TRAIN_DIR = _trainings_abs if os.path.exists(_trainings_abs) else '../resnet_data/trainings'
DATA_DIR  = TRAIN_DIR

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
    """
    Attention Residual U-Net with customizable output channels.

    For Gamma mixture model: num_outputs=3 (fraction_zero, shape, scale)
    """
    def __init__(self, in_channels=7, num_outputs=3):
        super(AttnResUNet, self).__init__()

        # Encoder
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))

        # Bridge
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))

        # Decoder with Attention Gates
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.upconv1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.upconv2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.upconv3 = ResidualBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.upconv4 = ResidualBlock(128, 64)

        # Final output layer - now outputs num_outputs channels (3 for Gamma)
        self.outc = nn.Conv2d(64, num_outputs, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention
        x = self.up1(x5)
        x4 = self.att1(g=x, x=x4)
        x = self.upconv1(torch.cat([x4, x], dim=1))

        x = self.up2(x)
        x3 = self.att2(g=x, x=x3)
        x = self.upconv2(torch.cat([x3, x], dim=1))

        x = self.up3(x)
        x2 = self.att3(g=x, x=x2)
        x = self.upconv3(torch.cat([x2, x], dim=1))

        x = self.up4(x)
        x1 = self.att4(g=x, x=x1)
        x = self.upconv4(torch.cat([x1, x], dim=1))

        # Output: 3 unconstrained values per pixel
        logits = self.outc(x)
        return logits

# ====================================================================
# --- LOSS FUNCTION: Negative Log-Likelihood for Zero-Inflated Gamma ---
# ====================================================================

class GammaNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for zero-inflated Gamma mixture model.

    This is a proper scoring rule (like CRPS) but computationally simpler,
    requiring only the PDF (not CDF), which has full gradient support.

    The NLL for a zero-inflated Gamma is:

    For each pixel, the forecast is:
    - With probability p₀: zero precipitation
    - With probability (1-p₀): Gamma(α, θ) distribution

    If observation y = 0:
        NLL = -log(p₀)

    If observation y > 0:
        NLL = -log(1 - p₀) + NLL_gamma(y; α, θ)

        where NLL_gamma is the negative log of the Gamma PDF:
        NLL_gamma(y; α, θ) = log(Γ(α)) + (1-α)×log(y/θ) + y/θ + log(θ)
                           = lgamma(α) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)

    This loss has full gradient support and is a proper scoring rule like CRPS.

    Parameters:
    -----------
    shape_min : float
        Minimum value for shape parameter (for numerical stability)
    scale_min : float
        Minimum value for scale parameter (for numerical stability)
    ignore_index : int
        Target value to ignore (bad quality pixels)
    epsilon : float
        Small constant added to avoid log(0) when y is very small
    """
    def __init__(self, shape_min=0.3, scale_min=0.01, ignore_index=-1, epsilon=1e-6):
        super(GammaNLLLoss, self).__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.ignore_index = ignore_index
        self.epsilon = epsilon  # For numerical stability in log(y)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 3, H, W) - raw network outputs
            targets: (B, H, W) - observed precipitation (mm)

        Returns:
            scalar NLL loss
        """
        # Transform outputs to constrained parameters
        # logits[:, 0] → fraction_zero via sigmoid
        # logits[:, 1] → shape via softplus + min
        # logits[:, 2] → scale via softplus + min

        fraction_zero = torch.sigmoid(logits[:, 0, :, :])  # [0, 1]
        shape = self.shape_min + F.softplus(logits[:, 1, :, :])  # [shape_min, ∞)
        scale = self.scale_min + F.softplus(logits[:, 2, :, :])  # [scale_min, ∞)

        # Create mask for valid pixels (not bad quality)
        valid_mask = (targets != self.ignore_index)

        # Separate zero and positive observations
        is_zero = (targets == 0.0) & valid_mask
        is_positive = (targets > 0.0) & valid_mask

        # Initialize NLL tensor
        nll = torch.zeros_like(targets)

        # ==========================================
        # Case 1: Observed zero precipitation
        # ==========================================
        # NLL = -log(p₀)
        # Add epsilon to avoid log(0)
        if is_zero.any():
            p0 = torch.clamp(fraction_zero[is_zero], min=self.epsilon, max=1.0 - self.epsilon)
            nll[is_zero] = -torch.log(p0)

        # ==========================================
        # Case 2: Observed positive precipitation
        # ==========================================
        if is_positive.any():
            y = targets[is_positive]
            p0 = fraction_zero[is_positive]
            alpha = shape[is_positive]
            theta = scale[is_positive]

            # Clamp p0 away from 1 to avoid log(0)
            p0_clamped = torch.clamp(p0, min=self.epsilon, max=1.0 - self.epsilon)

            # NLL for mixture: -log(1 - p₀) + NLL_gamma
            nll_mixture = -torch.log(1.0 - p0_clamped)

            # Gamma NLL: log(Γ(α)) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)
            # Simplified: lgamma(α) - (α-1)×log(y/θ) + y/θ + log(θ)

            # Add epsilon to y to avoid log(0) for very small precipitation
            y_safe = torch.clamp(y, min=self.epsilon)

            # Compute Gamma NLL using lgamma (has gradients!)
            lgamma_alpha = torch.lgamma(alpha)
            log_y = torch.log(y_safe)
            log_theta = torch.log(theta)

            # NLL_gamma = lgamma(α) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)
            nll_gamma = lgamma_alpha - (alpha - 1.0) * log_y + (alpha - 1.0) * log_theta + y / theta + log_theta

            # Total NLL for positive observations
            nll[is_positive] = nll_mixture + nll_gamma

        # Return mean NLL over valid pixels
        if valid_mask.sum() > 0:
            return nll[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=logits.device)

# ====================================================================
# --- DATASET ---
# ====================================================================

class GRAF_Dataset(Dataset):
    """
    Dataset loader for GRAF/MRMS patches with GFS features.

    Same as categorical version but returns continuous MRMS values instead
    of class indices for Gamma distribution fitting.
    """
    def __init__(self, pickle_file, normalization_stats=None, train=False):
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
                self.gfs_pwat = cPickle.load(f)  # Not used
                self.gfs_r = cPickle.load(f)
                self.gfs_cape = cPickle.load(f)  # Not used
        except Exception as e:
            print(f"CRITICAL ERROR loading pickle: {e}")
            sys.exit(1)

        if self.graf.shape[1] != PATCH_SIZE or self.graf.shape[2] != PATCH_SIZE:
             print(f"WARNING: Data shape {self.graf.shape} does not match PATCH_SIZE {PATCH_SIZE}")

        # Compute GRAF × RH interaction from raw values before normalization
        self.graf_rh_interaction = self.graf * self.gfs_r

        # Feature list: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
        feature_list = [self.graf, self.diff, self.gfs_r, self.terdiff_graf,
                       self.graf_rh_interaction, self.dlon, self.dlat]

        if normalization_stats is None:
            mins = [float(np.min(arr)) for arr in feature_list]
            maxs = [float(np.max(arr)) for arr in feature_list]
            maxs[0] = 75.0          # GRAF precip
            maxs[1] = max(maxs[1], 2500.0)   # terrain diff
            maxs[2] = max(maxs[2], 100.0)    # RH (%)
            maxs[3] = max(maxs[3], 35000.0)  # GRAF × terrain
            maxs[4] = max(maxs[4], 7500.0)   # GRAF × RH
            maxs[5] = max(maxs[5], 0.02)     # dlon
            maxs[6] = max(maxs[6], 0.02)     # dlat
            self.stats = {'min': mins, 'max': maxs}
        else:
            self.stats = normalization_stats

        # Normalize all features
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
        # Horizontal flip (negate dlon channel 5)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2); y = np.flip(y, axis=1)
            x[5, :, :] = -x[5, :, :]
        # Vertical flip (negate dlat channel 6)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1); y = np.flip(y, axis=0)
            x[6, :, :] = -x[6, :, :]
        return x.copy(), y.copy()

    def __getitem__(self, idx):
        # Stack in order: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
        x = np.stack([self.graf[idx], self.diff[idx], self.gfs_r[idx],
                     self.terdiff_graf[idx], self.graf_rh_interaction[idx],
                     self.dlon[idx], self.dlat[idx]], axis=0)

        # Return continuous MRMS values (not class indices)
        y_raw = self.mrms[idx]
        q_mask = self.qual[idx]

        # Mark bad quality pixels
        is_bad = (q_mask <= 0.01)
        y = y_raw.copy()
        y[is_bad] = -1  # ignore_index

        if self.train:
            x, y = self.apply_augmentation(x, y)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ====================================================================
# --- HELPER FUNCTIONS ---
# ====================================================================

def compute_gamma_climatology(train_dataset):
    """
    Compute climatological statistics from training data for initialization.

    Uses Thom's estimators (Method of Moments) for Gamma distribution:
    - Shape: α = μ² / σ²
    - Scale: θ = σ² / μ

    Returns:
    --------
    dict with keys:
        'fraction_zero': fraction of pixels with zero precipitation
        'shape_mean': mean shape parameter from wet pixels
        'scale_mean': mean scale parameter from wet pixels
        'shape_min': minimum shape for numerical stability (1st percentile)
        'scale_min': minimum scale for numerical stability (1st percentile)
    """
    print("\n" + "="*70)
    print("Computing Gamma climatology from training data...")
    print("="*70)

    # Sample up to 100,000 pixels for statistics
    n_samples = min(100000, len(train_dataset) * PATCH_SIZE * PATCH_SIZE)
    sample_indices = np.random.choice(len(train_dataset),
                                      size=min(1000, len(train_dataset)),
                                      replace=False)

    all_values = []
    for idx in sample_indices:
        _, y = train_dataset[idx]
        y_valid = y[y >= 0].numpy()  # Exclude bad quality
        all_values.append(y_valid)

    all_values = np.concatenate(all_values)

    # Compute fraction of zeros
    fraction_zero = (all_values == 0).sum() / len(all_values)

    # Extract wet pixels
    wet_values = all_values[all_values > 0]

    if len(wet_values) < 100:
        print("WARNING: Too few wet pixels for reliable Gamma fitting")
        return {
            'fraction_zero': fraction_zero,
            'shape_mean': 1.0,
            'scale_mean': 1.0,
            'shape_min': 0.3,
            'scale_min': 0.01
        }

    # Thom's estimators (Method of Moments)
    mu = np.mean(wet_values)
    sigma2 = np.var(wet_values)

    # Gamma parameters
    shape_clim = (mu ** 2) / sigma2
    scale_clim = sigma2 / mu

    # Compute minimum bounds (1st percentile of fitted distribution)
    # Fit Gamma to each patch separately, get distribution of parameters
    patch_shapes = []
    patch_scales = []

    for idx in sample_indices[:100]:  # Sample 100 patches
        _, y = train_dataset[idx]
        y_valid = y[y > 0].numpy()
        if len(y_valid) >= 10:
            mu_p = np.mean(y_valid)
            sigma2_p = np.var(y_valid)
            if sigma2_p > 0:
                shape_p = (mu_p ** 2) / sigma2_p
                scale_p = sigma2_p / mu_p
                patch_shapes.append(shape_p)
                patch_scales.append(scale_p)

    # Fixed minimum bounds (increased shape_min to improve light precip reliability)
    shape_min = 0.3
    scale_min = max(0.005, np.percentile(patch_scales, 1)) if patch_scales else 0.01

    print(f"  Fraction of zero pixels: {fraction_zero:.3f}")
    print(f"  Wet pixel statistics:")
    print(f"    Mean: {mu:.3f} mm")
    print(f"    Std: {np.sqrt(sigma2):.3f} mm")
    print(f"  Climatological Gamma parameters (Thom's estimators):")
    print(f"    Shape (α): {shape_clim:.3f}")
    print(f"    Scale (θ): {scale_clim:.3f}")
    print(f"  Minimum bounds:")
    print(f"    shape_min: {shape_min:.4f} (fixed)")
    print(f"    scale_min: {scale_min:.4f} (1st percentile)")
    print("="*70 + "\n")

    return {
        'fraction_zero': fraction_zero,
        'shape_mean': shape_clim,
        'scale_mean': scale_clim,
        'shape_min': shape_min,
        'scale_min': scale_min
    }

def initialize_output_layer(model, climatology):
    """
    Initialize the final output layer using climatology.

    The network outputs 3 unconstrained values per pixel:
    - logit[0] → fraction_zero via sigmoid
    - logit[1] → shape via softplus + shape_min
    - logit[2] → scale via softplus + scale_min

    We want initial predictions to match climatology:
    - sigmoid(bias[0]) ≈ fraction_zero
    - shape_min + softplus(bias[1]) ≈ shape_mean
    - scale_min + softplus(bias[2]) ≈ scale_mean

    Solving:
    - bias[0] = logit(fraction_zero) = log(p / (1-p))
    - bias[1] = inverse_softplus(shape_mean - shape_min)
    - bias[2] = inverse_softplus(scale_mean - scale_min)

    where inverse_softplus(y) = log(exp(y) - 1)
    """
    print("Initializing output layer with climatology...")

    # Get final layer
    final_layer = model.outc

    # Initialize weights with small random values
    nn.init.xavier_uniform_(final_layer.weight, gain=0.01)

    # Compute bias values
    p0 = climatology['fraction_zero']
    # Avoid extreme values
    p0 = np.clip(p0, 0.01, 0.99)
    bias_0 = np.log(p0 / (1 - p0))  # logit

    shape_target = climatology['shape_mean'] - climatology['shape_min']
    scale_target = climatology['scale_mean'] - climatology['scale_min']

    # inverse_softplus(y) = log(exp(y) - 1)
    # For numerical stability, if y is large, use log(exp(y)) = y
    def inverse_softplus(y):
        if y > 10:
            return y
        return np.log(np.exp(y) - 1)

    bias_1 = inverse_softplus(shape_target)
    bias_2 = inverse_softplus(scale_target)

    # Set biases (convert to torch tensors for device compatibility)
    with torch.no_grad():
        final_layer.bias[0] = torch.tensor(bias_0, dtype=torch.float32, device=DEVICE)
        final_layer.bias[1] = torch.tensor(bias_1, dtype=torch.float32, device=DEVICE)
        final_layer.bias[2] = torch.tensor(bias_2, dtype=torch.float32, device=DEVICE)

    print(f"  Initialized bias[0] = {bias_0:.3f} (fraction_zero)")
    print(f"  Initialized bias[1] = {bias_1:.3f} (shape)")
    print(f"  Initialized bias[2] = {bias_2:.3f} (scale)")
    print()

def print_diagnostics(epoch, batch_idx, loss_val, logits, targets,
                     shape_min, scale_min, model, stats):
    """
    Print diagnostics during training showing parameter distributions.
    Includes synthetic tests for 0mm and 1mm GRAF precipitation.
    """
    # Print explanation on first call
    if epoch == 0 and batch_idx == 0:
        print("\n" + "="*82)
        print("DIAGNOSTIC OUTPUT EXPLANATION")
        print("="*82)
        print("\nThis shows how the model predicts precipitation distributions.")
        print("\nFor real data, we show:")
        print("  - Average predicted parameters (fraction_zero, shape, scale)")
        print("  - Implied distribution characteristics")
        print("  - Comparison with observed statistics")
        print("\nFor synthetic tests, we show predicted distributions for:")
        print("  Syn(0mm): Dry conditions (GRAF=0mm, RH=20%, flat terrain)")
        print("  Syn(1mm): Light rain (GRAF=1mm, RH=80%, flat terrain)")
        print("\nWhat to look for:")
        print("  - Syn(0mm) should predict high fraction_zero (>0.8)")
        print("  - Syn(1mm) should predict moderate fraction_zero (~0.3-0.5)")
        print("  - Syn(1mm) should predict mean around 1-3mm")
        print("  - If Syn(0mm) predicts low fraction_zero, model has wet bias")
        print("="*82 + "\n")

    print(f"\n--- Epoch {epoch+1}, Batch {batch_idx} ---")
    print(f"Loss (NLL): {loss_val:.4f}")

    # Save training state
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # ==========================================
        # Real Data Statistics
        # ==========================================
        fraction_zero = torch.sigmoid(logits[:, 0, :, :])
        shape = shape_min + F.softplus(logits[:, 1, :, :])
        scale = shape_min + F.softplus(logits[:, 2, :, :])

        valid = (targets >= 0)

        if valid.sum() > 0:
            # Predicted statistics
            p0 = fraction_zero[valid].mean().item()
            alpha = shape[valid].mean().item()
            theta = scale[valid].mean().item()
            pred_mean = alpha * theta
            pred_std = np.sqrt(alpha * theta**2)

            # Observed statistics
            obs_zero_frac = (targets[valid] == 0).float().mean().item()
            obs_mean = targets[valid].mean().item()
            obs_std = targets[valid].std().item()

            print(f"\nReal data - Predicted parameters:")
            print(f"  Fraction zero: {p0:.3f}")
            print(f"  Shape (α):     {alpha:.3f}")
            print(f"  Scale (θ):     {theta:.3f}")
            print(f"  Implied mean:  {pred_mean:.3f} mm")
            print(f"  Implied std:   {pred_std:.3f} mm")

            print(f"\nReal data - Observed statistics:")
            print(f"  Fraction zero: {obs_zero_frac:.3f}")
            print(f"  Mean:          {obs_mean:.3f} mm")
            print(f"  Std:           {obs_std:.3f} mm")

        # ==========================================
        # Synthetic Tests
        # ==========================================
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

            # Forward pass
            amp_device = 'cuda' if USE_AMP else 'cpu'
            with torch.amp.autocast(amp_device, enabled=USE_AMP):
                syn_logits = model(syn_x)

            # Extract parameters
            p0 = torch.sigmoid(syn_logits[0, 0, :, :]).mean().item()
            alpha = (shape_min + F.softplus(syn_logits[0, 1, :, :])).mean().item()
            theta = (scale_min + F.softplus(syn_logits[0, 2, :, :])).mean().item()

            # Compute distribution characteristics
            mean = alpha * theta
            std = np.sqrt(alpha * theta**2)

            # Compute some probabilities using scipy for display
            # P(X > threshold) for positive component
            p_gt_025 = (1 - p0) * (1 - scipy_stats.gamma.cdf(0.25, alpha, scale=theta))
            p_gt_1 = (1 - p0) * (1 - scipy_stats.gamma.cdf(1.0, alpha, scale=theta))
            p_gt_5 = (1 - p0) * (1 - scipy_stats.gamma.cdf(5.0, alpha, scale=theta))

            return {
                'p0': p0,
                'alpha': alpha,
                'theta': theta,
                'mean': mean,
                'std': std,
                'p_gt_025': p_gt_025,
                'p_gt_1': p_gt_1,
                'p_gt_5': p_gt_5
            }

        # Dry case: 0mm GRAF, 20% RH
        syn_0mm = run_synthetic(0.0, 20.0)
        # Light rain case: 1mm GRAF, 80% RH
        syn_1mm = run_synthetic(1.0, 80.0)

        print(f"\nSynthetic test - Dry conditions (GRAF=0mm, RH=20%):")
        print(f"  P(zero):       {syn_0mm['p0']:.3f}")
        print(f"  Shape (α):     {syn_0mm['alpha']:.3f}")
        print(f"  Scale (θ):     {syn_0mm['theta']:.3f}")
        print(f"  Mean|wet:      {syn_0mm['mean']:.3f} mm")
        print(f"  P(>0.25mm):    {syn_0mm['p_gt_025']:.3f}")
        print(f"  P(>1mm):       {syn_0mm['p_gt_1']:.3f}")

        print(f"\nSynthetic test - Light rain (GRAF=1mm, RH=80%):")
        print(f"  P(zero):       {syn_1mm['p0']:.3f}")
        print(f"  Shape (α):     {syn_1mm['alpha']:.3f}")
        print(f"  Scale (θ):     {syn_1mm['theta']:.3f}")
        print(f"  Mean|wet:      {syn_1mm['mean']:.3f} mm")
        print(f"  P(>0.25mm):    {syn_1mm['p_gt_025']:.3f}")
        print(f"  P(>1mm):       {syn_1mm['p_gt_1']:.3f}")
        print(f"  P(>5mm):       {syn_1mm['p_gt_5']:.3f}")

        print("-" * 82)

    # Restore training state
    if was_training:
        model.train()

# ====================================================================
# --- TRAINING LOOP ---
# ====================================================================

def train_model(date_str, lead_time_str):
    """
    Main training function for Gamma mixture model.
    """
    print("\n" + "="*70)
    print(f"Training ResUNet GAMMA MODEL for {date_str} at {lead_time_str}h lead")
    print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | AMP: {USE_AMP}")
    print("="*70 + "\n")

    # Load data
    train_pickle = f"{DATA_DIR}/GRAF_Unet_data_train_{date_str}_{lead_time_str}h.cPick"
    val_pickle = f"{DATA_DIR}/GRAF_Unet_data_test_{date_str}_{lead_time_str}h.cPick"

    print(f"Loading training data from: {train_pickle}")
    print(f"Loading validation data from: {val_pickle}")

    train_dataset = GRAF_Dataset(train_pickle, train=True)
    val_dataset = GRAF_Dataset(val_pickle, normalization_stats=train_dataset.stats, train=False)

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input channels: 7 (GRAF, terrain, GFS RH, interactions, gradients)")
    print(f"  Output: 3 parameters (fraction_zero, shape, scale)")

    # Compute climatology for initialization
    climatology = compute_gamma_climatology(train_dataset)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS)

    # Create model with 3 outputs
    model = AttnResUNet(in_channels=7, num_outputs=3).to(DEVICE)

    # Initialize output layer with climatology
    initialize_output_layer(model, climatology)

    # Create loss function with climatological bounds
    criterion = GammaNLLLoss(
        shape_min=climatology['shape_min'],
        scale_min=climatology['scale_min'],
        ignore_index=-1
    ).to(DEVICE)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2
    )

    scaler = GradScaler() if USE_AMP else None

    # Setup checkpoint saving
    start_epoch = 0
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    checkpoint_path = f"{TRAIN_DIR}/resunet_gamma_{date_str}_{lead_time_str}h_best.pth"
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_checkpoint_path = None

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['loss']
            best_checkpoint_path = checkpoint_path
            print(f"   Resuming from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}\n")
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print(f"   WARNING: Checkpoint incompatible (different architecture)")
                print(f"   Starting fresh training\n")
            else:
                raise

    print(f"Starting training from epoch {start_epoch+1}...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Diagnostic output frequency: once per epoch\n")

    # Training loop
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
                loss = loss / ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS

            # Clear GPU cache periodically
            if i % 100 == 0 and DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        # Handle remaining gradients
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
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                amp_device = 'cuda' if USE_AMP else 'cpu'
                with torch.amp.autocast(amp_device, enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)

        # Print epoch-end diagnostics
        print_diagnostics(epoch, len(train_loader)-1, avg_train_loss,
                         outputs, targets, climatology['shape_min'],
                         climatology['scale_min'], model, train_dataset.stats)

        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = (new_lr != old_lr)

        # Track improvement
        improved = ""
        if avg_val < best_val_loss:
            improved = " *BEST*"

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val:.4f}{improved} | "
              f"LR: {new_lr:.2e}" +
              (" [reduced]" if lr_changed else ""))

        # Save checkpoint if validation improved
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0

            # Delete previous best
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)

            # Save new best
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'normalization_stats': train_dataset.stats,
                'climatology': climatology,  # Save for inference
            }
            torch.save(save_dict, checkpoint_path)
            best_checkpoint_path = checkpoint_path
            print(f"   Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    print(f"\nTraining complete!")
    print(f"Best validation NLL: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_path}")

# ====================================================================
# --- MAIN ---
# ====================================================================

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python pytorch_train_resunet_gamma.py YYYYMMDDHH lead_hours")
        print("Example: python pytorch_train_resunet_gamma.py 2025120100 12")
        sys.exit(1)

    train_model(sys.argv[1], sys.argv[2])
