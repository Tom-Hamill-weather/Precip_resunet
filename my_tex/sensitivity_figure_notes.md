# Sensitivity Figure: MLP 6-h Output vs. Hourly Expected-Precipitation Characteristics

## Purpose

This figure (`mlp_sensitivity_lead{clead}h.png`) provides a data-driven illustration
of how the six-hourly MLP probabilistic forecast depends on the mean and temporal
variability of the hourly expected precipitation amounts.  It supports the paper's
central argument that the MLP has learned the serial correlation structure of
precipitation — that consecutive wet hours produce a six-hourly accumulation
distribution with a heavier right tail than would be predicted by treating those
hours as independent.

## Summary axes

For each sample in the importance-sampled training set at lead time `clead`, two
summary statistics are computed from the six hourly input distributions:

| Symbol | Definition | Interpretation |
|--------|-----------|----------------|
| x | mean of E[Y_h] across 6 hours | average expected hourly precipitation (mm/h) |
| y | std dev of E[Y_h] across 6 hours | temporal variability in hourly amounts (mm/h) |

where the per-hour expected value is

    E[Y_h] = (1 - p0_h) × [ w_h × α1_h × θ1_h + (1-w_h) × α2_h × θ2_h ]

This quantity was chosen over the median (q0.5) of the full zero-inflated
distribution because the median equals zero whenever fraction_zero > 0.5 (most
grid points at most times), which causes degenerate clustering at the origin.
E[Y_h] is always non-negative, captures both the wet probability and the
conditional intensity, and is directly proportional to the expected 6-h sum.

Both axes are displayed on a **square-root scale** (tick marks placed at
sqrt(v) but labeled with physical v values, analogous to how a log-scale axis
labels physical values).  The sqrt scale handles y=0 naturally (no epsilon
offset required) and gives visually uniform spacing over the range 0.01–5 mm/h
that contains the bulk of the data.

## Panel (a): Binned-median MLP q0.9

Each hexagonal bin is colored by the **median 90th percentile of the 6-h MLP
output distribution** across all training samples falling in that bin.

The 6-h q0.9 is computed analytically via vectorized bisection on the
zero-inflated two-component Gamma CDF:

    F(y) = p0 + (1-p0) × [ w × gammainc(α1, y/θ1) + (1-w) × gammainc(α2, y/θ2) ]

Color scale: log-spaced (LogNorm), viridis colormap.

Contour iso-lines are drawn at round physical values (0.1, 0.25, 0.5, 1, 2, 5,
10, 20 mm) using the same binned-median statistic as the hexbin, ensuring
consistency between the contours and the background color.

### Interpretation

- Moving right (higher mean E[Y]) strongly increases the 6-h tail — a larger
  average hourly rate feeds a larger 6-h accumulation.
- Moving down (lower std E[Y], all hours similar) gives a somewhat heavier tail
  than moving up (higher std E[Y], hours variable) at the same mean.  This
  reflects the temporal consistency signature.
- The axis reaches approximately x ≈ 3 mm/h at the 99.5th percentile.  Values
  above this exist (maximum ≈ 11.6 mm/h) but are represented by only ~345
  samples (< 0.1% of the filtered set at lead 24 h), too few for reliable
  hexbin medians.

## Panel (b): Ratio MLP q0.9 / Naive-independence q0.9

Each bin is colored by the **median ratio** of the MLP's 90th percentile to the
90th percentile that would result if the six hourly distributions were treated as
statistically independent.

The **naive-independence q0.9** is computed by Monte Carlo: for each sample,
300 realizations are drawn from each of the six hourly zero-inflated Gamma
mixture distributions independently, summed across hours, and the 90th
percentile of the 300 six-hour totals is taken.

Color scale: TwoSlopeNorm centered at 1.0 (ratio = 1 is white; red = MLP
heavier than independence; blue = MLP lighter than independence), RdBu_r
colormap.

Contour iso-lines at [0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0].

### Interpretation

- **Ratio > 1 (red):** the MLP assigns a heavier right tail than naive
  independence predicts.  This is the serial-correlation signature: when the
  MLP has learned from training data that persistently wet hours tend to co-occur
  (positively correlated), the realized 6-h accumulation has higher variance than
  the sum of six independent hourly draws, inflating the tail.
- **Ratio < 1 (blue):** the MLP assigns a lighter tail than independence.  This
  can occur when hourly amounts are highly variable (high std E[Y]): a mix of
  very wet and completely dry hours produces a 6-h distribution that is more
  concentrated near the conditional mean than six independently heavy hours
  would suggest.
- **Overall statistics (lead 24 h):** median ratio 1.095; 10th percentile 0.534;
  90th percentile 1.829.  The broad interquartile range confirms the MLP is
  applying a nontrivial, position-dependent correction relative to independence.

## Meteorological framing for paper text

The x–y plane parameterizes the space of possible 6-h windows:
- **Low x, low y** (lower-left): a persistently light-precipitation or drizzle
  situation — every hour expects a small amount, all consistent with each other.
- **High x, low y** (lower-right): persistent heavy precipitation — all six
  hours agree on substantial rain, analogous to a long-lived frontal or orographic
  system.  Here the serial correlation is strongest and the MLP tail correction
  relative to independence is largest (red).
- **High x, high y** (upper-right): intense but variable precipitation — some
  hours heavy, some light or dry.  Characteristic of a convective line or
  isolated storm that passes through the 6-h window.
- **Low x, high y** (upper-left): isolated weak shower — one or two lightly wet
  hours in an otherwise dry window.

The figure thus directly illustrates the Hughes (1979) and Wilks (1990) finding
that persistent warm/cold-season events behave more like correlated draws
(ratio > 1) while scattered events behave more nearly independently (ratio ≈ 1
or < 1).

## Script and reproducibility

Script: `make_plots_mlp_sensitivity.py`
Usage:  `python make_plots_mlp_sensitivity.py [clead]`   (default: 24)
Output: `my_tex/Figs_6h/mlp_sensitivity_lead{clead}h.png`
Requires: trained MLP checkpoint at `mlp_trainings/6h_mlp_lead{clead}h.pth`
          and training data at `/data/resnet_data/prob_samples/`

Key parameters:
- QUANTILE = 0.90 (the output quantile studied)
- MC_DRAWS = 300 (Monte Carlo draws per sample for naive-independence)
- MIN_MEAN_EY = 0.01 mm/h (filter for near-dry samples)
- HEXBIN_GRID = 55 (hexbin resolution)
- CONTOUR_BINS = 45, CONTOUR_SIGMA = 0.8 (contour smoothing)

Runtime at lead 24 h (344,911 samples, CUDA): approx. 2–3 minutes, dominated
by the Monte Carlo naive-independence quantile computation.

## Key technical choices (for methods section)

### Contour consistency with hexbin
Contours are computed from `binned_statistic_2d(statistic='median')` on a
rectangular grid — the same operation as `hexbin(reduce_C_function=np.median)`.
Empty bins are filled with `NearestNDInterpolator` (not a global constant) before
Gaussian smoothing (sigma=0.8), so contours track local bin values rather than
being pulled toward the global median.  The NaN mask is restored after smoothing
so contours do not extend into data-sparse regions.

### Choice of E[Y_h] over q0.5
Using the median of the full zero-inflated distribution as the x-axis variable
would cause approximately half of all samples to fall exactly at zero (wherever
fraction_zero > 0.5), producing a degenerate column of points at the origin and
making the scatter uninformative.  E[Y_h] avoids this while remaining a
physically meaningful measure of expected precipitation rate.

### Importance sampling
Training data are drawn with weights proportional to 1 - min_h(p0_h), so that
wet samples (where at least one hour has a substantial wet probability) are
over-represented relative to their occurrence frequency.  This is necessary
because the sensitivity figure is only meaningful for samples with non-negligible
precipitation; the majority of grid-point/time combinations are essentially dry
and would otherwise dominate the scatter.
