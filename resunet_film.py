"""resunet_film.py

FiLM (Feature-wise Linear Modulation) conditioning for AttnResUNet, ported
from HRRRcal/pytorch_train_hrrr_gamma_mixture.py:919-1130. Lets one model
pool training data across many lead times (and day-of-year / solar-hour
phase) instead of training a separate model per lead time.

AttnResUNetFiLM subclasses the existing (unconditioned) AttnResUNet from
pytorch_train_resunet_gamma_mixture_v2.py, reusing its encoder/decoder/output
layers unchanged and adding a small conditioning MLP whose (gamma, beta)
outputs are applied at each of the 9 stages. Zero-initialized so FiLM starts
as the identity -- an existing non-FiLM checkpoint's backbone weights can be
loaded into this model (via load_state_dict(..., strict=False) for the
cond_mlp-less checkpoint) and warm-started.
"""

import torch
import torch.nn as nn

from pytorch_train_resunet_gamma_mixture_v2 import AttnResUNet

# Stage order: inc, down1-4, upconv1-4 -- same channel widths as the encoder/
# decoder in AttnResUNet.forward().
STAGE_CHANNELS = [64, 128, 256, 512, 1024, 512, 256, 128, 64]


class ConditioningMLP(nn.Module):
    """Maps `cond_dim` conditioning scalars to per-stage FiLM (gamma, beta)."""

    def __init__(self, cond_dim=5, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in STAGE_CHANNELS])
        # Zero-init so conditioning starts as identity (no effect at init).
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, cond):
        """cond: (B, cond_dim) -> list of 9 (gamma, beta) tensors, each (B, C, 1, 1)."""
        h = self.shared(cond)
        result = []
        for head, c in zip(self.heads, STAGE_CHANNELS):
            params = head(h)
            gamma = params[:, :c].view(-1, c, 1, 1)
            beta  = params[:, c:].view(-1, c, 1, 1)
            result.append((gamma, beta))
        return result


def film_apply(x, gamma, beta):
    """x * (1 + gamma) + beta -- identity when gamma=0, beta=0."""
    return x * (1.0 + gamma) + beta


class AttnResUNetFiLM(AttnResUNet):
    """AttnResUNet with FiLM conditioning at each encoder/decoder stage.

    forward(x, cond): x is (B, in_channels, 96, 96); cond is (B, cond_dim).
    """

    def __init__(self, in_channels=7, num_outputs=6, cond_dim=5):
        super().__init__(in_channels=in_channels, num_outputs=num_outputs)
        self.cond_dim = cond_dim
        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim)

    def forward(self, x, cond, return_attention=False):
        film = self.cond_mlp(cond)  # list of 9 (gamma, beta) pairs

        x1 = self.inc(x)
        x1 = film_apply(x1, *film[0])

        x2 = self.down1(x1)
        x2 = film_apply(x2, *film[1])

        x3 = self.down2(x2)
        x3 = film_apply(x3, *film[2])

        x4 = self.down3(x3)
        x4 = film_apply(x4, *film[3])

        x5 = self.down4(x4)
        x5 = film_apply(x5, *film[4])

        x  = self.up1(x5)
        x4 = self.att1(g=x, x=x4)
        x  = self.upconv1(torch.cat([x4, x], dim=1))
        x  = film_apply(x, *film[5])

        x  = self.up2(x)
        x3 = self.att2(g=x, x=x3)
        x  = self.upconv2(torch.cat([x3, x], dim=1))
        x  = film_apply(x, *film[6])

        x  = self.up3(x)
        x2 = self.att3(g=x, x=x2)
        x  = self.upconv3(torch.cat([x2, x], dim=1))
        x  = film_apply(x, *film[7])

        x  = self.up4(x)
        x1 = self.att4(g=x, x=x1)
        x  = self.upconv4(torch.cat([x1, x], dim=1))
        x  = film_apply(x, *film[8])

        logits = self.outc(x)
        if return_attention:
            attention_maps = [self.att1._alpha, self.att2._alpha,
                              self.att3._alpha, self.att4._alpha]
            return logits, attention_maps
        return logits
