"""
pytorch_train_resunet_gamma_mixture.py

Usage example:

$ python pytorch_train_resunet_gamma_mixture.py 2025120100 12

where you supply the YYYYMMDDHH of initial condition and lead time in h.

This routine trains an Attention Residual U-Net for probabilistic precipitation
forecasting using a ZERO-INFLATED 2-COMPONENT GAMMA MIXTURE MODEL instead of
a single Gamma distribution. This provides more flexibility to capture the
full conditional distribution of precipitation.

==============================================================================
MODEL OUTPUT: 6 Parameters per pixel
==============================================================================

Instead of predicting 102 categorical probabilities, this model predicts:

(1) fraction_zero (p₀): Probability of exactly zero precipitation [0, 1]
(2) weight (w): Mixing weight for first component [0, 1]
(3) shape1 (α₁): First Gamma shape parameter (α₁ > 0) - light precipitation
(4) scale1 (θ₁): First Gamma scale parameter (θ₁ > 0) - light precipitation
(5) shape2 (α₂): Second Gamma shape parameter (α₂ > α₁) - heavy precipitation
(6) scale2 (θ₂): Second Gamma scale parameter (θ₂ > 0) - heavy precipitation

For positive precipitation amounts, the distribution is a mixture of two Gammas:
    PDF(x) = w × Gamma(x; α₁, θ₁) + (1-w) × Gamma(x; α₂, θ₂)

Complete model:
    P(X = 0) = p₀
    P(X = x > 0) = (1 - p₀) × [w × Gamma(x; α₁, θ₁) + (1-w) × Gamma(x; α₂, θ₂)]

==============================================================================
LABEL SWITCHING PREVENTION
==============================================================================

To prevent label switching during training (where components swap roles), we
enforce a HARD ORDERING CONSTRAINT:

    α₂ = α₁ + softplus(shape_offset) + 0.5

This ensures that:
- Component 1 always represents lighter precipitation (lower shape parameter)
- Component 2 always represents heavier precipitation (higher shape parameter)
- The minimum separation (0.5) prevents degenerate solutions

==============================================================================
LOSS FUNCTION: Negative Log-Likelihood for 2-Component Mixture
==============================================================================

**Case 1: Observation y = 0**
    NLL(p₀, ... | y=0) = -log(p₀)

**Case 2: Observation y > 0**
    NLL = -log(1 - p₀) + NLL_mixture

    where:
    NLL_mixture = -log[w × pdf_gamma(y; α₁, θ₁) + (1-w) × pdf_gamma(y; α₂, θ₂)]

    pdf_gamma(y; α, θ) = exp(-NLL_gamma(y; α, θ))
    NLL_gamma(y; α, θ) = lgamma(α) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)

==============================================================================
INITIALIZATION STRATEGY
==============================================================================

The final layer is initialized using the EM algorithm (gamma_mixture_em.py)
applied to wet MRMS training pixels:

1. Compute fraction of zero pixels → initialize bias[0] = logit(fraction_zero)
2. Fit 2-component Gamma mixture to wet pixels using EM:
   - Returns: weights (w, 1-w), shapes (α₁, α₂), scales (θ₁, θ₂)
3. Initialize biases:
   - bias[1] = logit(w)
   - bias[2] = inverse_softplus(α₁ - shape_min)
   - bias[3] = inverse_softplus(θ₁ - scale_min)
   - bias[4] = inverse_softplus(α₂ - α₁ - 0.5)  # offset for hard constraint
   - bias[5] = inverse_softplus(θ₂ - scale_min)

This ensures the model starts with climatologically reasonable predictions.

==============================================================================
ARCHITECTURE
==============================================================================

Same Attention Residual U-Net as single Gamma version, but with 6 output
channels instead of 3.

Input features (7 channels):
(1) GRAF precipitation forecast
(2) Terrain elevation deviation (local terrain height difference)
(3) GFS column-average relative humidity
(4) Interaction: GRAF × terrain elevation deviation
(5) Interaction: GRAF × GFS relative humidity
(6) Terrain gradient (longitude direction)
(7) Terrain gradient (latitude direction)

Coded by Tom Hamill with Claude Code assistance, February 2026
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

# Import EM algorithm for initialization
from gamma_mixture_em import fit_gamma_mixture

# Enable CPU fallback for MPS unsupported operations
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
    AMP_DTYPE = torch.float32
else:
    BATCH_SIZE = 128  # L4 has 23 GB; batch=128 direct is faster than 16×8 accumulation
    NUM_WORKERS = 6
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16  # bfloat16: same dynamic range as float32, avoids lgamma overflow

# No gradient accumulation needed: batch=128 fits directly on the L4 GPU
ACCUMULATION_STEPS = 1

# --- 3. Training Hyperparameters ---

PATCH_SIZE = 96
BASE_LEARNING_RATE = 7.e-4  # Slightly lower for mixture model
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7

# --- 3a. Power Transformation Parameter ---
# Apply power transformation to GRAF precipitation: precip^POWER_TRANSFORM
# Default 0.5 = square root transformation for variance stabilization
# Set to 1.0 to disable transformation (use raw precipitation)
POWER_TRANSFORM = 0.5

# --- 3b. Numerical Stability Parameters (auto-adjusted based on power transform) ---
def get_stability_params(power_transform):
    """
    Return appropriate numerical stability parameters based on power transformation.

    With power transformation (< 1.0), values are compressed to smaller range,
    requiring more conservative epsilon and bounds.

    Without transformation (1.0), can use less conservative values.
    """
    if power_transform < 1.0:
        # Power transformation: more conservative for numerical stability
        return {
            'epsilon': 1e-4,           # Larger epsilon for log protection
            'shape_max': 100.0,        # Cap shape parameter
            'scale_max': 100.0,        # Cap scale parameter
            'nll_max': 100.0,          # Cap NLL loss
            'grad_clip': 1.0           # Gradient clipping threshold
        }
    else:
        # No transformation: can be less conservative
        return {
            'epsilon': 1e-6,           # Smaller epsilon sufficient
            'shape_max': 200.0,        # Allow larger shape
            'scale_max': 200.0,        # Allow larger scale
            'nll_max': 200.0,          # Allow larger NLL
            'grad_clip': 2.0           # Less aggressive clipping
        }

STABILITY_PARAMS = get_stability_params(POWER_TRANSFORM)

# --- 4. Loss Weighting (Initially disabled for unweighted NLL) ---

USE_WEIGHTED_LOSS = False
WEIGHT_BY_OBSERVATION = False  # If True, multiply NLL by f(observed_value)

# AWS path detection
# DATA_DIR: Location of training data pickle files
# TRAIN_DIR: Location to save model checkpoints (.pth files)
_base_aws1 = '/data/resnet_data'
_base_aws2 = '/data2/resnet_data'
_base_laptop = '../resnet_data'

if os.path.exists(_base_aws1):
    BASE_DIR = _base_aws1
elif os.path.exists(_base_aws2):
    BASE_DIR = _base_aws2
else:
    BASE_DIR = _base_laptop

# Check for training data in patch_data/ first, fall back to trainings/
# G5 GPU instance uses patch_data/, CPU instance uses trainings/
_patch_data = f"{BASE_DIR}/patch_data"
_trainings = f"{BASE_DIR}/trainings"

if os.path.exists(_patch_data) and len(os.listdir(_patch_data)) > 0:
    DATA_DIR = _patch_data
else:
    DATA_DIR = _trainings

TRAIN_DIR = _trainings  # Model checkpoints always go in trainings/

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

    For 2-component Gamma mixture: num_outputs=6
    (fraction_zero, weight, shape1, scale1, shape2_offset, scale2)
    """
    def __init__(self, in_channels=7, num_outputs=6):
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

        # Final output layer - now outputs 6 channels for mixture model
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

        # Output: 6 unconstrained values per pixel
        logits = self.outc(x)
        return logits

# ====================================================================
# --- LOSS FUNCTION: Negative Log-Likelihood for 2-Component Mixture ---
# ====================================================================

class GammaMixtureNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for zero-inflated 2-component Gamma mixture.

    The NLL for a zero-inflated 2-component Gamma mixture is:

    For each pixel, the forecast is:
    - With probability p₀: zero precipitation
    - With probability (1-p₀): Mixture of two Gammas
        - With weight w: Gamma(α₁, θ₁) - light precipitation
        - With weight (1-w): Gamma(α₂, θ₂) - heavy precipitation

    If observation y = 0:
        NLL = -log(p₀)

    If observation y > 0:
        NLL = -log(1 - p₀) + NLL_mixture

        where:
        NLL_mixture = -log[w × pdf_gamma(y; α₁, θ₁) + (1-w) × pdf_gamma(y; α₂, θ₂)]

    Parameters:
    -----------
    shape_min : float
        Minimum value for shape parameters (for numerical stability)
    scale_min : float
        Minimum value for scale parameters (for numerical stability)
    ignore_index : int
        Target value to ignore (bad quality pixels)
    epsilon : float
        Small constant added to avoid log(0)
    min_separation : float
        Minimum separation between shape1 and shape2 (for hard ordering)
    """
    def __init__(self, shape_min=0.3, scale_min=0.01, ignore_index=-1,
                 epsilon=1e-4, shape_max=100.0, scale_max=100.0, nll_max=100.0,
                 min_separation=0.5):
        super(GammaMixtureNLLLoss, self).__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.shape_max = shape_max
        self.scale_max = scale_max
        self.nll_max = nll_max
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        self.min_separation = min_separation

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 6, H, W) - raw network outputs
            targets: (B, H, W) - observed precipitation (mm)

        Returns:
            scalar NLL loss
        """
        # Transform outputs to constrained parameters
        # logits[:, 0] → fraction_zero via sigmoid
        # logits[:, 1] → weight via sigmoid
        # logits[:, 2] → shape1 via softplus + min
        # logits[:, 3] → scale1 via softplus + min
        # logits[:, 4] → shape2_offset via softplus (enforces shape2 > shape1)
        # logits[:, 5] → scale2 via softplus + min

        fraction_zero = torch.sigmoid(logits[:, 0, :, :])  # [0, 1]
        weight = torch.sigmoid(logits[:, 1, :, :])  # [0, 1]
        shape1 = self.shape_min + F.softplus(logits[:, 2, :, :])  # [shape_min, ∞)
        scale1 = self.scale_min + F.softplus(logits[:, 3, :, :])  # [scale_min, ∞)

        # Hard ordering constraint: shape2 = shape1 + softplus(offset) + min_separation
        shape2_offset = F.softplus(logits[:, 4, :, :])
        shape2 = shape1 + shape2_offset + self.min_separation  # Ensures shape2 > shape1

        scale2 = self.scale_min + F.softplus(logits[:, 5, :, :])  # [scale_min, ∞)

        # Clamp parameters to prevent numerical issues
        shape1 = torch.clamp(shape1, min=self.shape_min, max=self.shape_max)
        scale1 = torch.clamp(scale1, min=self.scale_min, max=self.scale_max)
        shape2 = torch.clamp(shape2, min=self.shape_min, max=self.shape_max)
        scale2 = torch.clamp(scale2, min=self.scale_min, max=self.scale_max)

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
        if is_zero.any():
            p0 = torch.clamp(fraction_zero[is_zero], min=self.epsilon, max=1.0 - self.epsilon)
            nll[is_zero] = -torch.log(p0)

        # ==========================================
        # Case 2: Observed positive precipitation
        # ==========================================
        if is_positive.any():
            y = targets[is_positive]
            p0 = fraction_zero[is_positive]
            w = weight[is_positive]
            alpha1 = shape1[is_positive]
            theta1 = scale1[is_positive]
            alpha2 = shape2[is_positive]
            theta2 = scale2[is_positive]

            # Clamp p0 away from 1 to avoid log(0)
            p0_clamped = torch.clamp(p0, min=self.epsilon, max=1.0 - self.epsilon)

            # NLL for zero-inflation: -log(1 - p₀)
            nll_zero_inflation = -torch.log(1.0 - p0_clamped)

            # Add epsilon to y to avoid log(0) for very small precipitation
            y_safe = torch.clamp(y, min=self.epsilon)

            # Compute NLL for each Gamma component (negative log of PDF)
            # NLL_gamma(y; α, θ) = lgamma(α) - (α-1)×log(y) + (α-1)×log(θ) + y/θ + log(θ)

            # Component 1 (light precipitation)
            lgamma_alpha1 = torch.lgamma(alpha1)
            log_y = torch.log(y_safe)
            log_theta1 = torch.log(theta1)
            nll_gamma1 = lgamma_alpha1 - (alpha1 - 1.0) * log_y + (alpha1 - 1.0) * log_theta1 + y / theta1 + log_theta1
            nll_gamma1 = torch.clamp(nll_gamma1, max=self.nll_max)

            # Component 2 (heavy precipitation)
            lgamma_alpha2 = torch.lgamma(alpha2)
            log_theta2 = torch.log(theta2)
            nll_gamma2 = lgamma_alpha2 - (alpha2 - 1.0) * log_y + (alpha2 - 1.0) * log_theta2 + y / theta2 + log_theta2
            nll_gamma2 = torch.clamp(nll_gamma2, max=self.nll_max)

            # Convert NLL to PDF (in log space for numerical stability)
            # log_pdf = -nll
            log_pdf1 = -nll_gamma1
            log_pdf2 = -nll_gamma2

            # Compute mixture PDF in log space using logsumexp trick
            # log(w × pdf1 + (1-w) × pdf2) = log(exp(log(w) + log_pdf1) + exp(log(1-w) + log_pdf2))
            w_clamped = torch.clamp(w, min=self.epsilon, max=1.0 - self.epsilon)
            log_w = torch.log(w_clamped)
            log_1_minus_w = torch.log(1.0 - w_clamped)

            # logsumexp([log_w + log_pdf1, log_1_minus_w + log_pdf2])
            log_mixture_pdf = torch.logsumexp(
                torch.stack([log_w + log_pdf1, log_1_minus_w + log_pdf2], dim=0),
                dim=0
            )

            # NLL for mixture: -log_mixture_pdf
            nll_mixture = -log_mixture_pdf
            nll_mixture = torch.clamp(nll_mixture, max=self.nll_max)

            # Total NLL for positive observations
            nll[is_positive] = nll_zero_inflation + nll_mixture

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

    Same as single Gamma version but for 2-component mixture model.
    Now supports both pickle (.cPick) and NetCDF (.nc) formats.
    """
    def __init__(self, pickle_file, normalization_stats=None, train=False, power_transform=1.0):
        self.train = train
        self.power_transform = power_transform
        try:
            # Load data using format-agnostic loader (supports .cPick and .nc)
            from data_loader_utils import load_training_data
            data = load_training_data(pickle_file)

            self.graf = data['GRAF']
            self.mrms = data['MRMS']
            self.qual = data['MRMS_qual']
            self.terdiff_graf = data['terdiff_x_GRAF']
            self.diff = data['terrain_diff']
            self.dlon = data['dt_dlon']
            self.dlat = data['dt_dlat']
            self.init_times = data['init_times']
            self.valid_times = data['valid_times']
            self.gfs_pwat = data['GFS_pwat']  # Not used
            self.gfs_r = data['GFS_r']
            self.gfs_cape = data['GFS_cape']  # Not used
        except Exception as e:
            print(f"CRITICAL ERROR loading data: {e}")
            sys.exit(1)

        if self.graf.shape[1] != PATCH_SIZE or self.graf.shape[2] != PATCH_SIZE:
             print(f"WARNING: Data shape {self.graf.shape} does not match PATCH_SIZE {PATCH_SIZE}")

        # Apply power transformation to GRAF precipitation BEFORE computing interactions
        if self.power_transform != 1.0:
            print(f"  Applying power transformation: GRAF^{self.power_transform}")
            self.graf = np.power(self.graf, self.power_transform)
            # Recompute interaction terms with transformed GRAF
            self.terdiff_graf = self.graf * self.diff

        # Compute GRAF × RH interaction from raw/transformed values before normalization
        self.graf_rh_interaction = self.graf * self.gfs_r

        # Feature list: GRAF, diff, RH, GRAF×diff, GRAF×RH, dlon, dlat
        feature_list = [self.graf, self.diff, self.gfs_r, self.terdiff_graf,
                       self.graf_rh_interaction, self.dlon, self.dlat]

        if normalization_stats is None:
            mins = [float(np.min(arr)) for arr in feature_list]
            maxs = [float(np.max(arr)) for arr in feature_list]
            # Adjust max for GRAF based on power transformation
            if self.power_transform == 1.0:
                maxs[0] = 75.0          # GRAF precip (raw)
            else:
                maxs[0] = np.power(75.0, self.power_transform)  # transformed
            maxs[1] = max(maxs[1], 2500.0)   # terrain diff
            maxs[2] = max(maxs[2], 100.0)    # RH (%)
            # Interaction terms also affected by power transformation
            if self.power_transform == 1.0:
                maxs[3] = max(maxs[3], 35000.0)  # GRAF × terrain
                maxs[4] = max(maxs[4], 7500.0)   # GRAF × RH
            else:
                maxs[3] = max(maxs[3], np.power(75.0, self.power_transform) * 2500.0)
                maxs[4] = max(maxs[4], np.power(75.0, self.power_transform) * 100.0)
            maxs[5] = max(maxs[5], 0.02)     # dlon
            maxs[6] = max(maxs[6], 0.02)     # dlat
            self.stats = {'min': mins, 'max': maxs}
        else:
            self.stats = normalization_stats

        # Normalize all features, pre-stack into (N, 7, H, W), then free individual arrays
        self.graf              = self.normalize(self.graf, 0)
        self.diff              = self.normalize(self.diff, 1)
        self.gfs_r             = self.normalize(self.gfs_r, 2)
        self.terdiff_graf      = self.normalize(self.terdiff_graf, 3)
        self.graf_rh_interaction = self.normalize(self.graf_rh_interaction, 4)
        self.dlon              = self.normalize(self.dlon, 5)
        self.dlat              = self.normalize(self.dlat, 6)
        self.features = np.stack(
            [self.graf, self.diff, self.gfs_r, self.terdiff_graf,
             self.graf_rh_interaction, self.dlon, self.dlat], axis=1
        ).astype(np.float32)  # (N, 7, 96, 96)
        del self.graf, self.diff, self.gfs_r, self.terdiff_graf, self.graf_rh_interaction, self.dlon, self.dlat

    def normalize(self, data, idx):
        vmin = self.stats['min'][idx]
        vmax = self.stats['max'][idx]
        denom = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
        return ((data - vmin) / denom).astype(np.float32)

    def __len__(self):
        return len(self.features)

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
        x = self.features[idx].copy()  # (7, 96, 96) — copy avoids writing into the shared array

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

def compute_gamma_mixture_climatology(train_dataset):
    """
    Compute climatological statistics from training data for initialization.

    Uses EM algorithm (gamma_mixture_em.py) to fit 2-component Gamma mixture
    to wet pixels.

    Returns:
    --------
    dict with keys:
        'fraction_zero': fraction of pixels with zero precipitation
        'weight': mixing weight for component 1
        'shape1': shape parameter for component 1 (light)
        'scale1': scale parameter for component 1 (light)
        'shape2': shape parameter for component 2 (heavy)
        'scale2': scale parameter for component 2 (heavy)
        'shape_min': minimum shape for numerical stability
        'scale_min': minimum scale for numerical stability
    """
    print("\n" + "="*70)
    print("Computing 2-component Gamma mixture climatology using EM algorithm...")
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

    # Extract wet pixels for mixture fitting
    wet_values = all_values[all_values > 0]

    if len(wet_values) < 100:
        print("WARNING: Too few wet pixels for reliable Gamma mixture fitting")
        return {
            'fraction_zero': fraction_zero,
            'weight': 0.5,
            'shape1': 0.8,
            'scale1': 1.0,
            'shape2': 2.0,
            'scale2': 3.0,
            'shape_min': 0.3,
            'scale_min': 0.01
        }

    # Sub-sample wet pixels for faster EM (max 50,000 pixels)
    max_wet_pixels = 50000
    if len(wet_values) > max_wet_pixels:
        print(f"\n  Sub-sampling {max_wet_pixels} wet pixels from {len(wet_values)} for EM...")
        subsample_indices = np.random.choice(len(wet_values), size=max_wet_pixels, replace=False)
        wet_values = wet_values[subsample_indices]

    # Fit 2-component Gamma mixture using EM algorithm
    print(f"  Fitting 2-component Gamma mixture to {len(wet_values)} wet pixels...")
    print(f"  Using EM algorithm (gamma_mixture_em.py)...")

    try:
        weights, shapes, scales, model = fit_gamma_mixture(
            wet_values,
            n_components=2,
            init_method='moments',
            verbose=False,
            max_iter=500
        )

        # Sort components by shape parameter (light → heavy)
        # This ensures component 1 is always the lighter distribution
        sort_idx = np.argsort(shapes)
        weights = weights[sort_idx]
        shapes = shapes[sort_idx]
        scales = scales[sort_idx]

        weight1 = float(weights[0])
        shape1 = float(shapes[0])
        scale1 = float(scales[0])
        shape2 = float(shapes[1])
        scale2 = float(scales[1])

        print(f"\n  EM converged after {model.n_iter_} iterations")
        print(f"  Log-likelihood: {model.loglik_:.2f}")

    except Exception as e:
        print(f"  WARNING: EM algorithm failed: {e}")
        print(f"  Using fallback: simple moment-based initialization")
        # Fallback to simple initialization
        weight1 = 0.5
        shape1 = 0.8
        scale1 = np.percentile(wet_values, 25)
        shape2 = 2.0
        scale2 = np.percentile(wet_values, 75)

    # Fixed minimum bounds
    shape_min = 0.3
    scale_min = 0.01

    print(f"\n  Fraction of zero pixels: {fraction_zero:.3f}")
    print(f"  Component 1 (light precipitation):")
    print(f"    Weight: {weight1:.3f}")
    print(f"    Shape (α₁): {shape1:.3f}")
    print(f"    Scale (θ₁): {scale1:.3f}")
    print(f"    Mean: {shape1 * scale1:.3f} mm")
    print(f"  Component 2 (heavy precipitation):")
    print(f"    Weight: {1-weight1:.3f}")
    print(f"    Shape (α₂): {shape2:.3f}")
    print(f"    Scale (θ₂): {scale2:.3f}")
    print(f"    Mean: {shape2 * scale2:.3f} mm")
    print(f"  Minimum bounds:")
    print(f"    shape_min: {shape_min:.4f} (fixed)")
    print(f"    scale_min: {scale_min:.4f} (fixed)")
    print("="*70 + "\n")

    return {
        'fraction_zero': fraction_zero,
        'weight': weight1,
        'shape1': shape1,
        'scale1': scale1,
        'shape2': shape2,
        'scale2': scale2,
        'shape_min': shape_min,
        'scale_min': scale_min
    }

def initialize_output_layer(model, climatology):
    """
    Initialize the final output layer using climatology from EM algorithm.

    The network outputs 6 unconstrained values per pixel:
    - logit[0] → fraction_zero via sigmoid
    - logit[1] → weight via sigmoid
    - logit[2] → shape1 via softplus + shape_min
    - logit[3] → scale1 via softplus + scale_min
    - logit[4] → shape2_offset via softplus (for hard ordering)
    - logit[5] → scale2 via softplus + scale_min

    We want initial predictions to match climatology:
    - sigmoid(bias[0]) ≈ fraction_zero
    - sigmoid(bias[1]) ≈ weight
    - shape_min + softplus(bias[2]) ≈ shape1
    - scale_min + softplus(bias[3]) ≈ scale1
    - shape1 + softplus(bias[4]) + 0.5 ≈ shape2
    - scale_min + softplus(bias[5]) ≈ scale2

    Solving:
    - bias[0] = logit(fraction_zero)
    - bias[1] = logit(weight)
    - bias[2] = inverse_softplus(shape1 - shape_min)
    - bias[3] = inverse_softplus(scale1 - scale_min)
    - bias[4] = inverse_softplus(shape2 - shape1 - 0.5)
    - bias[5] = inverse_softplus(scale2 - scale_min)

    where inverse_softplus(y) = log(exp(y) - 1)
    """
    print("Initializing output layer with 2-component Gamma mixture climatology...")

    # Get final layer
    final_layer = model.outc

    # Initialize weights with small random values
    nn.init.xavier_uniform_(final_layer.weight, gain=0.01)

    # Compute bias values
    p0 = climatology['fraction_zero']
    w = climatology['weight']

    # Avoid extreme values
    p0 = np.clip(p0, 0.01, 0.99)
    w = np.clip(w, 0.01, 0.99)

    bias_0 = np.log(p0 / (1 - p0))  # logit(fraction_zero)
    bias_1 = np.log(w / (1 - w))    # logit(weight)

    shape1_target = climatology['shape1'] - climatology['shape_min']
    scale1_target = climatology['scale1'] - climatology['scale_min']
    shape2_offset_target = climatology['shape2'] - climatology['shape1'] - 0.5
    scale2_target = climatology['scale2'] - climatology['scale_min']

    # Ensure targets are positive
    shape1_target = max(shape1_target, 0.1)
    scale1_target = max(scale1_target, 0.01)
    shape2_offset_target = max(shape2_offset_target, 0.1)
    scale2_target = max(scale2_target, 0.01)

    # inverse_softplus(y) = log(exp(y) - 1)
    # For numerical stability, if y is large, use log(exp(y)) = y
    def inverse_softplus(y):
        if y > 10:
            return y
        return np.log(np.exp(y) - 1)

    bias_2 = inverse_softplus(shape1_target)
    bias_3 = inverse_softplus(scale1_target)
    bias_4 = inverse_softplus(shape2_offset_target)
    bias_5 = inverse_softplus(scale2_target)

    # Set biases (convert to torch tensors for device compatibility)
    with torch.no_grad():
        final_layer.bias[0] = torch.tensor(bias_0, dtype=torch.float32, device=DEVICE)
        final_layer.bias[1] = torch.tensor(bias_1, dtype=torch.float32, device=DEVICE)
        final_layer.bias[2] = torch.tensor(bias_2, dtype=torch.float32, device=DEVICE)
        final_layer.bias[3] = torch.tensor(bias_3, dtype=torch.float32, device=DEVICE)
        final_layer.bias[4] = torch.tensor(bias_4, dtype=torch.float32, device=DEVICE)
        final_layer.bias[5] = torch.tensor(bias_5, dtype=torch.float32, device=DEVICE)

    print(f"  Initialized bias[0] = {bias_0:.3f} (fraction_zero = {p0:.3f})")
    print(f"  Initialized bias[1] = {bias_1:.3f} (weight = {w:.3f})")
    print(f"  Initialized bias[2] = {bias_2:.3f} (shape1 = {climatology['shape1']:.3f})")
    print(f"  Initialized bias[3] = {bias_3:.3f} (scale1 = {climatology['scale1']:.3f})")
    print(f"  Initialized bias[4] = {bias_4:.3f} (shape2 offset = {shape2_offset_target:.3f})")
    print(f"  Initialized bias[5] = {bias_5:.3f} (scale2 = {climatology['scale2']:.3f})")
    print()

def print_diagnostics(epoch, batch_idx, loss_val, logits, targets,
                     shape_min, scale_min, model, stats):
    """
    Print diagnostics during training showing parameter distributions for mixture model.
    Includes synthetic tests for 0mm and 1mm GRAF precipitation.
    """
    # Print explanation on first call
    if epoch == 0 and batch_idx == 0:
        print("\n" + "="*82)
        print("DIAGNOSTIC OUTPUT EXPLANATION - 2-COMPONENT MIXTURE MODEL")
        print("="*82)
        print("\nThis shows how the model predicts 2-component Gamma mixture distributions.")
        print("\nFor real data, we show:")
        print("  - Average predicted parameters (fraction_zero, weight, shape1/2, scale1/2)")
        print("  - Implied distribution characteristics for each component")
        print("  - Comparison with observed statistics")
        print("\nFor synthetic tests, we show predicted distributions for:")
        print("  Syn(0mm): Dry conditions (GRAF=0mm, RH=20%, flat terrain)")
        print("  Syn(1mm): Light rain (GRAF=1mm, RH=80%, flat terrain)")
        print("\nWhat to look for:")
        print("  - Syn(0mm) should predict high fraction_zero (>0.8)")
        print("  - Syn(1mm) should predict moderate fraction_zero (~0.3-0.5)")
        print("  - Component 1 should capture light precipitation")
        print("  - Component 2 should capture heavy precipitation")
        print("  - Shape2 should always be > Shape1 (hard ordering constraint)")
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
        weight = torch.sigmoid(logits[:, 1, :, :])
        shape1 = shape_min + F.softplus(logits[:, 2, :, :])
        scale1 = scale_min + F.softplus(logits[:, 3, :, :])
        shape2_offset = F.softplus(logits[:, 4, :, :])
        shape2 = shape1 + shape2_offset + 0.5
        scale2 = scale_min + F.softplus(logits[:, 5, :, :])

        valid = (targets >= 0)

        if valid.sum() > 0:
            # Predicted statistics
            p0 = fraction_zero[valid].mean().item()
            w = weight[valid].mean().item()
            alpha1 = shape1[valid].mean().item()
            theta1 = scale1[valid].mean().item()
            alpha2 = shape2[valid].mean().item()
            theta2 = scale2[valid].mean().item()

            # Component means
            mean1 = alpha1 * theta1
            std1 = np.sqrt(alpha1 * theta1**2)
            mean2 = alpha2 * theta2
            std2 = np.sqrt(alpha2 * theta2**2)

            # Overall mixture mean (for wet pixels)
            mixture_mean = w * mean1 + (1 - w) * mean2

            # Observed statistics
            obs_zero_frac = (targets[valid] == 0).float().mean().item()
            obs_mean = targets[valid].mean().item()
            obs_std = targets[valid].std().item()

            print(f"\nReal data - Predicted parameters:")
            print(f"  Fraction zero: {p0:.3f}")
            print(f"  Component 1 (light) - weight: {w:.3f}")
            print(f"    Shape (α₁):  {alpha1:.3f}")
            print(f"    Scale (θ₁):  {theta1:.3f}")
            print(f"    Mean:        {mean1:.3f} mm")
            print(f"    Std:         {std1:.3f} mm")
            print(f"  Component 2 (heavy) - weight: {1-w:.3f}")
            print(f"    Shape (α₂):  {alpha2:.3f}")
            print(f"    Scale (θ₂):  {theta2:.3f}")
            print(f"    Mean:        {mean2:.3f} mm")
            print(f"    Std:         {std2:.3f} mm")
            print(f"  Mixture mean (wet pixels): {mixture_mean:.3f} mm")

            print(f"\nReal data - Observed statistics:")
            print(f"  Fraction zero: {obs_zero_frac:.3f}")
            print(f"  Mean:          {obs_mean:.3f} mm")
            print(f"  Std:           {obs_std:.3f} mm")

        # ==========================================
        # Synthetic Tests
        # ==========================================
        def run_synthetic(precip_mm, rh_pct):
            """Run model on synthetic uniform input."""
            # Apply power transformation to synthetic GRAF value
            p_val = np.power(precip_mm, POWER_TRANSFORM)
            t_val = 0.0  # Flat terrain

            f0 = p_val              # GRAF precip (transformed)
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
            w = torch.sigmoid(syn_logits[0, 1, :, :]).mean().item()
            alpha1 = (shape_min + F.softplus(syn_logits[0, 2, :, :])).mean().item()
            theta1 = (scale_min + F.softplus(syn_logits[0, 3, :, :])).mean().item()
            offset = F.softplus(syn_logits[0, 4, :, :]).mean().item()
            alpha2 = alpha1 + offset + 0.5
            theta2 = (scale_min + F.softplus(syn_logits[0, 5, :, :])).mean().item()

            # Compute distribution characteristics
            mean1 = alpha1 * theta1
            mean2 = alpha2 * theta2
            mixture_mean = w * mean1 + (1 - w) * mean2

            # Compute some probabilities using scipy for display
            # P(X > threshold) for mixture
            def mixture_sf(thresh):
                # Survival function: P(X > thresh) = (1 - p0) * [w * SF1 + (1-w) * SF2]
                sf1 = scipy_stats.gamma.sf(thresh, alpha1, scale=theta1)
                sf2 = scipy_stats.gamma.sf(thresh, alpha2, scale=theta2)
                return (1 - p0) * (w * sf1 + (1 - w) * sf2)

            p_gt_025 = mixture_sf(0.25)
            p_gt_1 = mixture_sf(1.0)
            p_gt_5 = mixture_sf(5.0)

            return {
                'p0': p0,
                'w': w,
                'alpha1': alpha1,
                'theta1': theta1,
                'alpha2': alpha2,
                'theta2': theta2,
                'mean1': mean1,
                'mean2': mean2,
                'mixture_mean': mixture_mean,
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
        print(f"  Weight (w):    {syn_0mm['w']:.3f}")
        print(f"  Comp1: α₁={syn_0mm['alpha1']:.2f}, θ₁={syn_0mm['theta1']:.2f}, mean={syn_0mm['mean1']:.2f}mm")
        print(f"  Comp2: α₂={syn_0mm['alpha2']:.2f}, θ₂={syn_0mm['theta2']:.2f}, mean={syn_0mm['mean2']:.2f}mm")
        print(f"  Mixture mean:  {syn_0mm['mixture_mean']:.3f} mm")
        print(f"  P(>0.25mm):    {syn_0mm['p_gt_025']:.3f}")
        print(f"  P(>1mm):       {syn_0mm['p_gt_1']:.3f}")

        print(f"\nSynthetic test - Light rain (GRAF=1mm, RH=80%):")
        print(f"  P(zero):       {syn_1mm['p0']:.3f}")
        print(f"  Weight (w):    {syn_1mm['w']:.3f}")
        print(f"  Comp1: α₁={syn_1mm['alpha1']:.2f}, θ₁={syn_1mm['theta1']:.2f}, mean={syn_1mm['mean1']:.2f}mm")
        print(f"  Comp2: α₂={syn_1mm['alpha2']:.2f}, θ₂={syn_1mm['theta2']:.2f}, mean={syn_1mm['mean2']:.2f}mm")
        print(f"  Mixture mean:  {syn_1mm['mixture_mean']:.3f} mm")
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
    Main training function for 2-component Gamma mixture model.
    """
    print("\n" + "="*70)
    print(f"Training ResUNet 2-COMPONENT GAMMA MIXTURE MODEL")
    print(f"Date: {date_str} | Lead time: {lead_time_str}h")
    print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | AMP: {USE_AMP}")
    print(f"Power Transform: GRAF^{POWER_TRANSFORM}")
    print(f"Stability params: epsilon={STABILITY_PARAMS['epsilon']:.0e}, "
          f"grad_clip={STABILITY_PARAMS['grad_clip']}, "
          f"shape_max={STABILITY_PARAMS['shape_max']}, "
          f"nll_max={STABILITY_PARAMS['nll_max']}")
    print("="*70 + "\n")

    # Load data
    train_pickle = f"{DATA_DIR}/GRAF_Unet_data_train_{date_str}_{lead_time_str}h.cPick"
    val_pickle = f"{DATA_DIR}/GRAF_Unet_data_test_{date_str}_{lead_time_str}h.cPick"

    print(f"Loading training data from: {train_pickle}")
    print(f"Loading validation data from: {val_pickle}")

    train_dataset = GRAF_Dataset(train_pickle, train=True, power_transform=POWER_TRANSFORM)
    val_dataset = GRAF_Dataset(val_pickle, normalization_stats=train_dataset.stats, train=False, power_transform=POWER_TRANSFORM)

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input channels: 7 (GRAF, terrain, GFS RH, interactions, gradients)")
    print(f"  Output: 6 parameters (p0, w, α₁, θ₁, α₂, θ₂)")

    # Compute climatology for initialization using EM algorithm
    climatology = compute_gamma_mixture_climatology(train_dataset)

    # Create dataloaders
    pin = (DEVICE.type != 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=pin, persistent_workers=False)

    # Create model with 6 outputs
    model = AttnResUNet(in_channels=7, num_outputs=6).to(DEVICE)

    # Initialize output layer with climatology from EM
    initialize_output_layer(model, climatology)

    # Create loss function with climatological bounds and stability parameters
    criterion = GammaMixtureNLLLoss(
        shape_min=climatology['shape_min'],
        scale_min=climatology['scale_min'],
        ignore_index=-1,
        epsilon=STABILITY_PARAMS['epsilon'],
        shape_max=STABILITY_PARAMS['shape_max'],
        scale_max=STABILITY_PARAMS['scale_max'],
        nll_max=STABILITY_PARAMS['nll_max'],
        min_separation=0.5
    ).to(DEVICE)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=2
    )

    scaler = None  # bfloat16 has full float32 dynamic range; no GradScaler needed

    # Setup checkpoint saving
    start_epoch = 0
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    checkpoint_path = f"{TRAIN_DIR}/resunet_gamma_mixture_{date_str}_{lead_time_str}h_best.pth"
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
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Forward pass: conv layers run in bfloat16; loss in float32
            # (lgamma/log in the NLL loss needs float32 precision)
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                output = model(data)
            loss = criterion(output.float(), target) / ACCUMULATION_STEPS

            # Backward pass
            if USE_AMP and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=STABILITY_PARAMS['grad_clip']
                )

                if USE_AMP and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS

            # Print diagnostics once per epoch (first batch)
            if batch_idx == 0:
                print_diagnostics(
                    epoch, batch_idx, loss.item() * ACCUMULATION_STEPS,
                    output.float(), target,
                    climatology['shape_min'], climatology['scale_min'],
                    model, train_dataset.stats
                )

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                    output = model(data)
                loss = criterion(output.float(), target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'climatology': climatology,
                'power_transform': POWER_TRANSFORM,
                'normalization_stats': train_dataset.stats
            }
            torch.save(checkpoint, checkpoint_path)
            best_checkpoint_path = checkpoint_path
            print(f"  → Saved best model: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement for {epochs_no_improve} epochs)")

        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if best_checkpoint_path:
        print(f"Best model saved at: {best_checkpoint_path}")
    print("="*70 + "\n")

# ====================================================================
# --- MAIN ---
# ====================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pytorch_train_resunet_gamma_mixture.py YYYYMMDDHH lead_time")
        print("Example: python pytorch_train_resunet_gamma_mixture.py 2025120100 12")
        sys.exit(1)

    date_str = sys.argv[1]
    lead_time_str = sys.argv[2]

    train_model(date_str, lead_time_str)
