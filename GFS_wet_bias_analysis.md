# Analysis: Widespread Light Precipitation Bias in GFS-Enhanced Model

## Problem Statement
The GFS-enhanced model (panel b in the first plot) shows widespread low probabilities (2-10%) of light precipitation (>0.25 mm/h) across nearly the entire domain, including dry regions. This is highly undesirable compared to the non-GFS model which correctly concentrates probabilities near actual precipitation features.

## Key Findings from Investigation

### 1. **CRITICAL: Insufficient Dry Patches in Training Data**
- **Only 5.7%** of training patches have essentially zero GRAF precipitation (< 0.01mm)
- **Only 7.5%** of training patches have essentially zero MRMS precipitation
- **Root cause**: The preferential wet sampling strategy in `save_patched_GRAF_MRMS_GFS.py` (lines 154-179) heavily biases toward wet patches
- **Impact**: The model has inadequate training examples to learn what "truly dry" atmospheric conditions look like

### 2. **GFS Feature Values in Dry vs. Wet Conditions**

**Training data - "Dry" patches (GRAF < 0.01mm):**
- PWAT: mean = 12.34 kg/m², median = 10.35 kg/m²
- RH: mean = 21.49%, median = 20.19%
- CAPE: mean = 7.38 J/kg, median = 0.00 J/kg

**Inference case (December 2025, showing widespread false alarms):**
- PWAT: mean = 20.58 kg/m², median = 18.53 kg/m²
- RH: mean = 37.20%, median = 36.90%
- CAPE: mean = 164.47 J/kg, median = 1.00 J/kg

**After normalization:**
- Inference PWAT is **1.7x higher** than training dry patches
- Inference RH is **1.7x higher** than training dry patches
- Inference CAPE is **22.3x higher** than training dry patches

### 3. **Model Learning Problem**
The model has learned an incorrect association:
- **What it should learn**: "Low GRAF + moderate GFS values → low precipitation probability"
- **What it actually learned**: "Moderate GFS values → some precipitation probability" (because it rarely saw moderate GFS values paired with truly dry conditions)

### 4. **Additional Observations**
- Training data has suspicious MRMS values (max = 9.97e36), indicating potential data quality issues
- Mean GRAF in training: 0.69 mm (median: 0.00 mm)
- Mean MRMS in training: unknown due to corrupt values, but has similar dry patch percentage

## Root Cause Hypotheses (Ranked by Likelihood)

### **Hypothesis 1: Training Data Sampling Bias (MOST LIKELY)**
**Evidence:**
- Only 5.7% dry patches despite atmosphere being dry much more often
- GFS values in rare "dry" training patches are substantially lower than typical dry conditions
- Model extrapolating beyond training distribution

**Mechanism:**
The preferential wet sampling (lines 154-179 in `save_patched_GRAF_MRMS_gemini.py`) was designed to ensure enough heavy precipitation events for training. However, this created an extreme imbalance:
1. Wet day: 50 patches per day
2. Normal day: 35 patches per day
3. Dry day: 20 patches per day
4. Within each day, patches are weighted toward wetter locations

This double-bias (macro and micro) results in almost no truly dry examples.

### **Hypothesis 2: GFS Features Lack Clear Dry Signal (LIKELY)**
**Evidence:**
- Even in "dry" training patches, GFS features are non-zero (PWAT ~12 kg/m², RH ~21%)
- Atmosphere always has some moisture; GFS features don't have a clear "dry threshold"
- GRAF precipitation (0.00 mm) provides a clear dry signal that GFS lacks

**Mechanism:**
Unlike GRAF which can be exactly 0.00 mm, GFS moisture fields are continuous:
- PWAT is always > 0 (column-integrated water vapor)
- RH is always > 0 (atmospheric moisture)
- CAPE can be 0, but mean is non-zero even in dry patches

The model may not know what GFS threshold distinguishes "moisture but no rain" from "moisture with rain."

### **Hypothesis 3: Class Imbalance and Loss Function (MODERATE)**
**Evidence:**
- WeightedOrdinalWassersteinLoss emphasizes heavy precipitation classes
- Class weights: `weights_np[0] = 1.0` (explicitly set), but others up to 5.0
- Model may optimize for wet events at expense of dry prediction skill

**Mechanism:**
The loss function (lines 303-327 in `pytorch_train_resunet_gfs.py`) is designed to penalize errors on rare heavy events more than common light/no events. This is appropriate for the problem, but combined with sampling bias, may lead to:
- Model "plays it safe" by forecasting low probabilities everywhere
- Better to slightly overforecast light precip than miss heavy events
- No-precip class (class 0) not sufficiently rewarded

### **Hypothesis 4: Feature Normalization Issues (LESS LIKELY)**
**Evidence:**
- Training uses fixed max values: PWAT=70, RH=100, CAPE=5000
- Actual training data maxes: PWAT=69.51, RH=79.06, CAPE=3300.88
- Inference case maxes: PWAT=61.31, RH=78.50, CAPE=2471.00

**Assessment:**
Normalization appears reasonable. The fixed maxes are close to actual maxes, and inference values are within training range. This is unlikely to be the primary cause.

### **Hypothesis 5: Interpolation Artifacts (LESS LIKELY)**
**Evidence:**
- GFS data interpolated from coarse lat/lon grid (~0.5°?) to GRAF grid (4km)
- Interpolation could smooth/blur features

**Assessment:**
While interpolation could contribute to spatial spreading of moisture signals, it wouldn't explain the domain-wide bias. The smoothing would affect both wet and dry regions similarly.

## Recommended Solutions (in Priority Order)

### **Solution 1: Rebalance Training Data Sampling (HIGHEST PRIORITY)**
Modify `save_patched_GRAF_MRMS_GFS.py` to ensure adequate dry patch representation:

**Option A - Stratified Sampling:**
```python
# Target composition:
# - 30% truly dry patches (GRAF < 0.01mm AND MRMS < 0.01mm)
# - 40% light precipitation (0.01-2.5mm)
# - 20% moderate precipitation (2.5-10mm)
# - 10% heavy precipitation (>10mm)
```

**Option B - Explicit Dry Patch Inclusion:**
- Keep current wet-biased sampling for 70% of patches
- Force inclusion of 30% dry patches (random sampling from GRAF < 0.01mm regions)

**Expected Impact:** Model will learn proper GFS thresholds for dry conditions

### **Solution 2: Add Explicit "Dry" Features (MEDIUM PRIORITY)**
Augment the feature set with derived variables that have clear dry signals:

```python
# Channel 9: GRAF precip indicator (0 if GRAF=0, 1 otherwise)
# Channel 10: Low moisture indicator (1 if PWAT < threshold, 0 otherwise)
# Channel 11: Moisture/GRAF ratio (helps model learn when moisture doesn't produce rain)
```

**Expected Impact:** Provides model with explicit dry/wet classification features

### **Solution 3: Adjust Loss Function Weights (LOWER PRIORITY)**
Modify class weights to reward accurate dry forecasts:

```python
# In pytorch_train_resunet_gfs.py, line 88-89:
weights_np[0] = 2.0  # Increase from 1.0 to penalize false alarms more
# OR use asymmetry factor > 1.0 to penalize false alarms vs misses
```

**Expected Impact:** Model will be more conservative in forecasting light precipitation

### **Solution 4: Post-Processing Dry Mask (QUICK FIX, NOT RECOMMENDED)**
Apply a threshold to zero out low probabilities where GRAF shows no precipitation:

```python
# In inference: if GRAF < 0.01mm, set prob < 0.1 to 0.0
```

**Drawback:** This defeats the purpose of adding GFS features - they should help correct GRAF, not be overridden by it.

## Testing the Hypotheses

### **Immediate Diagnostic Tests:**

1. **Check if problem exists at other lead times and cases:**
   - Run inference on multiple cases (wet days, dry days, transition days)
   - Compare 6h, 12h, 18h, 24h lead times
   - Document if bias is consistent or case-dependent

2. **Examine model's learned feature importance:**
   - Use gradient-based attribution (e.g., Integrated Gradients)
   - Determine which features drive low-probability forecasts in dry regions
   - Check if GFS features dominate over GRAF signal

3. **Test with artificially modified GFS inputs:**
   - Run inference with GFS features set to zero → should match non-GFS model
   - Run inference with GFS features set to training dry patch means
   - Run inference with GFS features set to training wet patch means

4. **Analyze validation set performance:**
   - Compute Brier Score for class 0 (no precip) vs other classes
   - Check if validation loss shows bias toward false alarms
   - Compare reliability diagrams for GFS vs non-GFS models

### **Long-term Solution Validation:**

After retraining with rebalanced data:
1. Verify training data has ~30% truly dry patches
2. Check that validation loss improves for low-probability forecasts
3. Compare spatial fields visually for multiple test cases
4. Compute verification metrics (Brier Score, ROC AUC) stratified by GRAF amount
5. Ensure heavy precipitation skill is not degraded

## Data Quality Issues Noted

- **MRMS corrupt values**: Max value of 9.97e36 mm suggests bad data not filtered
- **Check quality control**: Verify that `qual <= 0.01` masking is working correctly
- **Training vs inference mismatch**: Confirm GFS interpolation is identical between training data generation and inference

## Conclusion

The most likely cause of the widespread light precipitation bias is **insufficient training data for dry conditions** combined with **GFS features lacking a clear dry signal**. The model has learned to associate moderate GFS moisture values with precipitation because it rarely encountered examples of "moderate GFS values + no precipitation."

The recommended fix is to **rebalance the training data sampling** to include 25-30% truly dry patches, ensuring the model learns the full range of atmospheric conditions where precipitation does or does not occur.

---
Generated: February 6, 2026
Analysis of: ResUnet_small_GRAF_probs_IC2025120412_lead12h_gfs.png vs. ResUnet_small_GRAF_probs_IC2025120412_lead12h.png
