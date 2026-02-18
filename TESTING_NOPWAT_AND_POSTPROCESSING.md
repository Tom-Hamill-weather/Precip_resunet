# Testing No-PWAT Model & Post-Processing Calibration

## Overview

Based on our discussion, we're pursuing two parallel approaches:
1. **Test if removing PWAT fixes the wet bias** (7-channel model: RH + CAPE only)
2. **Keep current model, apply post-processing calibration** (if PWAT removal doesn't help)

---

## Approach 1: No-PWAT Model (7 channels)

### Rationale
PWAT shows poor discriminatory power between dry and light precipitation:
- Dry patches: PWAT = 12.34 kg/m² (median: 10.35)
- Light precip patches: PWAT = 13.65 kg/m² (median: 11.76)
- **Only +1.3 kg/m² difference!**

Plus, PWAT is climatologically dependent:
- High PWAT in warm regions doesn't guarantee precipitation
- Low PWAT in cool regions can still have precipitation

RH (relative humidity) is more physically meaningful:
- Dry: RH = 21%
- Light: RH = 36%
- Moderate: RH = 45%
- Heavy: RH = 49%
- **Clear progression with good separation**

### Training the No-PWAT Model

```bash
# Train 7-channel model (uses existing GFS training data, just drops PWAT channel)
python pytorch_train_resunet_gfs_nopwat.py 2025120100 12

# Model weights saved as: resunet_ordinal_gfs_nopwat_*.pth
```

**Advantages:**
- Uses same training data (no regeneration needed)
- Tests PWAT hypothesis cleanly
- If it works, no post-processing needed

**Note:** You'll need to create inference/plotting scripts for the 7-channel model (similar to what we did for GFS).

### Expected Outcome
If PWAT was the problem, the 7-channel model should:
- Show reduced false alarms in dry regions
- Maintain skill for moderate/heavy precipitation (RH and CAPE are good predictors there)
- Similar or better performance than 8-channel version

---

## Approach 2: Post-Processing Calibration (Keep 8-channel model)

If the no-PWAT model doesn't fix the issue, or if you want to salvage the existing 8-channel model, apply calibration.

### Option 2A: Conditional Thresholding

Apply a GRAF-conditional threshold to suppress false alarms:

```python
def apply_conditional_threshold(dl_probs, graf_precip, threshold_map):
    """
    Suppress low probabilities where GRAF shows little/no precipitation.

    Parameters:
    -----------
    dl_probs : dict
        Deep learning probabilities for each threshold (e.g., '0p25', '1', etc.)
    graf_precip : ndarray
        GRAF precipitation field (mm)
    threshold_map : dict
        Mapping of probability threshold to GRAF precip threshold
        e.g., {'0p25': (0.01, 0.10)} means:
              "If GRAF < 0.01mm, set prob < 0.10 to zero"

    Returns:
    --------
    calibrated_probs : dict
        Calibrated probabilities
    """
    calibrated_probs = {}

    for key, prob_field in dl_probs.items():
        graf_thresh, prob_thresh = threshold_map.get(key, (0.0, 0.0))

        # Where GRAF is very low, suppress low probabilities
        mask = (graf_precip < graf_thresh) & (prob_field < prob_thresh)

        calibrated_field = prob_field.copy()
        calibrated_field[mask] = 0.0

        calibrated_probs[key] = calibrated_field

    return calibrated_probs

# Example usage in resunet_inference_gfs.py:
threshold_map = {
    '0p25': (0.01, 0.10),   # If GRAF < 0.01mm, zero out prob < 0.10
    '1':    (0.10, 0.05),   # If GRAF < 0.10mm, zero out prob < 0.05
    '2p5':  (0.25, 0.03),   # If GRAF < 0.25mm, zero out prob < 0.03
    '5':    (0.50, 0.02),   # etc.
    '10':   (1.00, 0.01),
    '25':   (2.50, 0.01)
}

dl_probs_calibrated = apply_conditional_threshold(dl_probs, precipitation_GRAF, threshold_map)
```

**Advantages:**
- Simple to implement
- Physically motivated (trust GRAF's dry signal)
- No retraining needed

**Disadvantages:**
- Partially defeats purpose of adding GFS features
- Requires tuning threshold_map parameters
- Doesn't fix underlying model issue

### Option 2B: Isotonic Regression Calibration

Train a calibration mapping using validation data:

```python
from sklearn.isotonic import IsotonicRegression

def train_calibration(dl_probs_valid, mrms_valid, precip_threshold):
    """
    Train isotonic regression to calibrate probabilities.

    Parameters:
    -----------
    dl_probs_valid : ndarray
        Model-predicted probabilities on validation set
    mrms_valid : ndarray
        Observed MRMS precipitation on validation set
    precip_threshold : float
        Threshold for binary outcome (e.g., 0.25mm)

    Returns:
    --------
    calibrator : IsotonicRegression
        Trained calibration function
    """
    # Binary outcome: did precipitation exceed threshold?
    y_binary = (mrms_valid >= precip_threshold).astype(int)

    # Flatten arrays
    probs_flat = dl_probs_valid.ravel()
    y_flat = y_binary.ravel()

    # Remove bad data
    valid_mask = (mrms_valid.ravel() >= 0)
    probs_flat = probs_flat[valid_mask]
    y_flat = y_flat[valid_mask]

    # Train calibrator
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(probs_flat, y_flat)

    return calibrator

def apply_calibration(dl_probs, calibrator):
    """Apply calibration to probability field."""
    shape = dl_probs.shape
    probs_flat = dl_probs.ravel()
    calibrated_flat = calibrator.transform(probs_flat)
    return calibrated_flat.reshape(shape)
```

**Advantages:**
- Data-driven, optimal calibration
- Maintains probability interpretation
- Can improve reliability diagrams

**Disadvantages:**
- Requires validation data with MRMS truth
- More complex implementation
- Needs separate calibrator for each threshold

### Option 2C: Stratified Calibration by GFS Features

Recognize that calibration needs vary by atmospheric regime:

```python
def stratified_calibration(dl_probs, gfs_pwat, gfs_rh):
    """
    Apply different calibration based on atmospheric moisture regime.

    Hypothesis: In dry atmospheres (low RH, moderate PWAT),
    model over-forecasts due to training data bias.
    """
    calibrated = dl_probs.copy()

    # Define regimes
    dry_regime = (gfs_rh < 30) & (gfs_pwat < 20)
    moist_regime = (gfs_rh >= 50)

    # Aggressive suppression in dry regime
    calibrated[dry_regime] *= 0.3  # Reduce by 70%

    # Slight adjustment in moist regime (model is well-calibrated here)
    calibrated[moist_regime] *= 1.0  # No change

    # Moderate adjustment in transition regime
    transition = ~dry_regime & ~moist_regime
    calibrated[transition] *= 0.7  # Reduce by 30%

    return np.clip(calibrated, 0, 1)
```

**Advantages:**
- Targets the specific problem (dry regime false alarms)
- Uses GFS features constructively
- Can be tuned based on verification

**Disadvantages:**
- Requires careful regime definition
- Hard to optimize without extensive validation

---

## Recommended Testing Sequence

### Phase 1: Test No-PWAT Model (1-2 days)
1. Train 7-channel model on one lead time (12h): `python pytorch_train_resunet_gfs_nopwat.py 2025120100 12`
2. Create inference script for 7-channel model
3. Run inference on test case: 2025120412
4. Visually compare:
   - 5-channel (no GFS)
   - 8-channel (PWAT + RH + CAPE)
   - 7-channel (RH + CAPE only)

**Decision point:** If 7-channel model looks good → train all lead times and proceed
**If not** → Move to Phase 2

### Phase 2: Implement Post-Processing (1 day)
1. Start with Option 2A (Conditional Thresholding) - simplest
2. Tune threshold_map parameters using 2-3 test cases
3. If results acceptable → deploy
4. If not → Try Option 2C (Stratified Calibration)

### Phase 3: Validation (ongoing)
Once you have a working solution:
1. Compute verification metrics on multiple cases
2. Check stratified performance (dry days, wet days, transition)
3. Compare Brier Score, ROC AUC across different solutions
4. Generate reliability diagrams

---

## Quick Implementation: Conditional Thresholding in Inference

Here's a drop-in modification for `resunet_inference_gfs.py`:

```python
# Add after line 393 (after dl_probs are computed):

# Apply conditional thresholding calibration
if True:  # Set to True to enable, False to disable
    print("Applying conditional threshold calibration...")

    threshold_map = {
        '0p25': (0.01, 0.12),
        '1':    (0.10, 0.08),
        '2p5':  (0.25, 0.05),
        '5':    (0.50, 0.03),
        '10':   (1.00, 0.02),
        '25':   (2.50, 0.01)
    }

    for key in dl_probs.keys():
        if key in threshold_map:
            graf_thresh, prob_thresh = threshold_map[key]
            mask = (precipitation_GRAF < graf_thresh) & (dl_probs[key] < prob_thresh)
            dl_probs[key][mask] = 0.0
            print(f"  {key}mm: Zeroed {np.sum(mask)} pixels (GRAF<{graf_thresh}, prob<{prob_thresh})")
```

**Tuning the thresholds:**
- First value (graf_thresh): GRAF amount below which to apply suppression
- Second value (prob_thresh): Probability threshold to suppress
- Start conservative (only suppress very low probs where GRAF is near-zero)
- Gradually increase suppression until visual results look reasonable
- Verify you're not degrading skill on wet events

---

## My Recommendation

1. **Try the no-PWAT model first** - cleanest solution if PWAT is the culprit
2. **If that doesn't work**, implement conditional thresholding (Option 2A) as a quick fix
3. **For production**, train a larger dataset with better class balance (future work)

The post-processing approaches are pragmatic and can be deployed quickly, but they're band-aids. The no-PWAT experiment will tell us if the feature set is fundamentally flawed, which is valuable scientific insight regardless of the outcome.

---

## Next Steps

Would you like me to:
1. Create the 7-channel inference script (`resunet_inference_gfs_nopwat.py`)?
2. Implement the conditional thresholding in the existing 8-channel inference script?
3. Both?

Let me know which direction you want to pursue first!
