# Step 6 - Heterogeneous effects & uplift model

## Part A - Segment analysis (full data, PySpark)

Each feature cut into up to 5 quantile buckets (fewer when a feature has few distinct values). ITT lift per bucket; Cochran's Q tests whether lifts differ across buckets.

| feature | buckets | Visit Q | Visit p (Bonferroni) | Visit lift range (pp) | Conversion Q | Conversion p (Bonferroni) | Conversion lift range (pp) |
|---|---|---|---|---|---|---|---|
| f0 | 5 | 6,123 | 0 | 5.02 | 772 | 2.4e-164 | 0.536 |
| f6 | 4 | 5,863 | 0 | 3.54 | 749 | 1.5e-160 | 0.366 |
| f2 | 4 | 3,750 | 0 | 4.96 | 870 | 5.8e-187 | 0.838 |
| f8 | 3 | 3,513 | 0 | 3.58 | 868 | 7.1e-188 | 0.477 |
| f9 | 3 | 2,848 | 0 | 3.81 | 782 | 3.6e-169 | 0.530 |
| f7 | 2 | 2,287 | 0 | 5.18 | 254 | 8.1e-56 | 0.610 |
| f1 | 2 | 879 | 7e-192 | 8.51 | 39 | 1.3e-08 | 0.788 |
| f10 | 2 | 489 | 5.3e-107 | 3.66 | 462 | 3.5e-101 | 1.511 |
| f4 | 2 | 489 | 5.3e-107 | 3.66 | 462 | 3.5e-101 | 1.511 |
| f11 | 1 | 0 | 1 | 0.00 | 0 | 1 | 0.000 |
| f3 | 1 | 0 | 1 | 0.00 | 0 | 1 | 0.000 |
| f5 | 1 | 0 | 1 | 0.00 | 0 | 1 | 0.000 |

- Features with significant heterogeneity after Bonferroni (24 tests): **9/12** for visit, **9/12** for conversion.

**f0** - lift by bucket

| bucket | range | users | visit C → T | visit lift | conv. C → T | conv. lift | BH-adj. p (visit) |
|---|---|---|---|---|---|---|---|
| Q1 | (-inf, 12.6] | 3,705,545 | 5.16% → 5.33% | +0.17 pp | 0.149% → 0.187% | +0.038 pp | 1.5e-07 |
| Q2 | (12.6, 18.1] | 1,878,727 | 9.35% → 14.43% | +5.08 pp | 0.826% → 1.371% | +0.545 pp | 0 |
| Q3 | (18.1, 22.8] | 2,801,753 | 3.46% → 4.32% | +0.86 pp | 0.159% → 0.240% | +0.081 pp | 8.2e-145 |
| Q4 | (22.8, 24.8] | 2,790,879 | 1.58% → 1.78% | +0.20 pp | 0.061% → 0.078% | +0.016 pp | 1.3e-20 |
| Q5 | (24.8, +inf] | 2,802,688 | 1.24% → 1.29% | +0.06 pp | 0.036% → 0.045% | +0.009 pp | 0.0026 |

**f6** - lift by bucket

| bucket | range | users | visit C → T | visit lift | conv. C → T | conv. lift | BH-adj. p (visit) |
|---|---|---|---|---|---|---|---|
| Q1 | (-inf, -7.82] | 2,906,669 | 6.92% → 10.63% | +3.71 pp | 0.556% → 0.940% | +0.384 pp | 0 |
| Q2 | (-7.82, -3.99] | 3,104,117 | 2.81% → 3.38% | +0.57 pp | 0.114% → 0.176% | +0.062 pp | 1.9e-90 |
| Q3 | (-3.99, -1.29] | 4,263,261 | 1.37% → 1.53% | +0.16 pp | 0.059% → 0.076% | +0.018 pp | 5e-23 |
| Q4 | (-1.29, 0.294] | 3,705,545 | 5.16% → 5.33% | +0.17 pp | 0.149% → 0.187% | +0.038 pp | 1.5e-07 |

**f2** - lift by bucket

| bucket | range | users | visit C → T | visit lift | conv. C → T | conv. lift | BH-adj. p (visit) |
|---|---|---|---|---|---|---|---|
| Q1 | (-inf, 8.21] | 7,303,429 | 0.10% → 0.15% | +0.05 pp | 0.006% → 0.009% | +0.003 pp | 1.8e-43 |
| Q2 | (8.21, 8.38] | 1,082,885 | 30.03% → 35.04% | +5.01 pp | 1.769% → 2.610% | +0.841 pp | 0 |
| Q3 | (8.38, 8.81] | 2,790,708 | 6.98% → 8.98% | +2.00 pp | 0.268% → 0.443% | +0.175 pp | 0 |
| Q4 | (8.81, +inf] | 2,802,570 | 0.88% → 1.22% | +0.33 pp | 0.037% → 0.058% | +0.021 pp | 2e-77 |

**Multiple comparisons.** 60 bucket-level tests were run. 60 have raw p < 0.05, 60 survive Benjamini-Hochberg (FDR 5%). With this many cuts some 'significant' segments are expected by chance alone (~5% of true nulls), so segment findings are treated as hypotheses and confirmed with a held-out model evaluation below.

![Segment lift](figures/06_segment_lift.png)

## Part B - T-learner uplift model

- Random sample of 2,000,620 users, split 50/50 into train (1,000,310) and held-out test (1,000,310), stratified by treatment.
- Two LightGBM classifiers per outcome (treated / control); uplift(x) = P(y=1 | x, T) − P(y=1 | x, C).
- Evaluation on the test set only. Qini coefficient = area between model and random Qini curves, normalised by total incremental outcomes (0 = random).

| Outcome | Qini coef. | Top 10% | Top 20% | **Top 30%** | Top 50% |
|---|---|---|---|---|---|
| visit | 0.281 | 54% [49%, 59%] | 71% [66%, 77%] | **80% [75%, 85%]** | 86% [81%, 90%] |
| conversion | 0.273 | 67% [57%, 77%] | 77% [69%, 86%] | **78% [71%, 87%]** | 83% [76%, 91%] |

Cells = share of all incremental outcomes captured by targeting only that top share of users (bootstrap 95% CI). Random targeting would capture exactly the share targeted (10%, 20%, ...).

**Visit uplift by decile (test set)**

| decile | predicted | observed | 95% CI |
|---|---|---|---|
| 1 | +8.730 pp | +5.442 pp | [+4.702, +6.182] |
| 2 | +1.295 pp | +1.044 pp | [+0.656, +1.433] |
| 3 | +0.326 pp | +0.391 pp | [+0.183, +0.599] |
| 4 | +0.137 pp | +0.072 pp | [-0.052, +0.197] |
| 5 | +0.067 pp | +0.073 pp | [-0.018, +0.163] |
| 6 | +0.042 pp | +0.034 pp | [-0.029, +0.096] |
| 7 | +0.028 pp | +0.011 pp | [-0.055, +0.076] |
| 8 | +0.014 pp | +0.019 pp | [-0.042, +0.079] |
| 9 | -0.050 pp | +0.094 pp | [-0.040, +0.229] |
| 10 | -2.246 pp | +1.235 pp | [+0.691, +1.778] |

**Conversion uplift by decile (test set)**

| decile | predicted | observed | 95% CI |
|---|---|---|---|
| 1 | +1.983 pp | +0.859 pp | [+0.635, +1.082] |
| 2 | +0.132 pp | +0.067 pp | [+0.004, +0.130] |
| 3 | +0.045 pp | -0.016 pp | [-0.062, +0.030] |
| 4 | +0.021 pp | +0.028 pp | [+0.010, +0.046] |
| 5 | +0.011 pp | +0.005 pp | [-0.015, +0.025] |
| 6 | +0.008 pp | -0.001 pp | [-0.021, +0.018] |
| 7 | +0.006 pp | +0.003 pp | [-0.012, +0.017] |
| 8 | +0.005 pp | +0.008 pp | [+0.002, +0.014] |
| 9 | +0.004 pp | -0.016 pp | [-0.043, +0.011] |
| 10 | -0.394 pp | +0.225 pp | [+0.133, +0.318] |

## Takeaways

- **Targeting recommendation:** serving ads only to the 30% of users with the highest predicted uplift captures **78%** of incremental conversions (95% CI 71%–87%) and **80%** of incremental visits, versus 30% under random targeting - at 30% of the media cost.
- The effect is ITT (per assigned user), which is exactly the lever a targeting policy controls: who is *eligible* for ads.
- **Use the model for ranking, not for absolute values.** Top-decile predicted visit uplift is +8.73 pp vs +5.44 pp observed. T-learner predictions are over-dispersed because the two models' errors do not cancel; the Qini curve, which only depends on the ranking, is the right evaluation.
- Caveats: features are anonymised, so segments cannot be named in business terms; the model is trained on a sample; conversion uplift is estimated from few control conversions and is the noisier of the two.

![Qini curves](figures/06_qini_curves.png)

![Uplift by decile](figures/06_uplift_by_decile.png)

_Runtime: 70s_
