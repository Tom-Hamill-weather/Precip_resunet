# 2Δx Noise in ResUNet Output

## Observation

Panel b of Figure 4 in method_6h.tex (6h_MLP_4panel_IC2025120412_lead24h.png) shows
2Δx-scale noise artifacts, e.g., top-center of the domain in the P(≥0.25mm) panel.
The 6-h MLP inherits this noise from the 1-h ResUNet probabilities.

## Primary Cause: ConvTranspose2d

The decoder in `pytorch_train_resunet_gamma_mixture.py` uses:

```python
self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
```

With `kernel_size=2, stride=2`, every other output pixel in both directions is written
by a different subset of kernel weights. When those weights become unequal during
training (which they almost always do), the result is a periodic checkerboard pattern
at exactly 2Δx. This artifact is present at both training and inference time — it is
structurally embedded in the learned weights, not introduced post-hoc.

## Other Contributing Factors

- **Patch stitching**: Manhattan-distance blending reduces boundary discontinuities but
  does not suppress within-patch high-frequency noise.
- **Residual shortcuts**: Identity paths in ResidualBlocks allow high-frequency
  components to bypass the smoothing effect of 3×3 convolutions.
- **Attention gate 1×1 convolutions**: Pointwise ops provide zero spatial smoothing;
  spatially noisy attention weights modulate skip features pixel-by-pixel.

## Recommended Fix

Replace each `ConvTranspose2d` with bilinear upsampling + regular convolution:

```python
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
)
```

Bilinear interpolation has uniform spatial coverage (no unequal-overlap problem),
and the subsequent 3×3 conv handles channel mixing without spatial periodicity.

**Requires retraining from scratch** — the artifact is baked into the learned weights,
so changing the architecture mid-training would not help.

## Related Questions Explored

- Max pooling (used in encoder downsampling) is not the cause here — it does not
  introduce periodic spatial artifacts, though strided convolutions may be preferable
  for other reasons (avoids wet bias from always selecting the maximum pixel value).
