# Step 2 - Experiment validity

## 1. Sample Ratio Mismatch (SRM)

| | Treatment | Control |
|---|---|---|
| Observed users | 11,882,655 | 2,096,937 |
| Designed share | 85.00% | 15.00% |
| Observed share | 85.00001% | 14.99999% |

- 95% CI of observed treatment share: [84.98129%, 85.01873%]
- Chi-square = 1.818e-06, p = 0.9989 (alarm threshold p < 0.001)
- No SRM: the observed split is consistent with the design.

## 2. Covariate balance (f0-f11)

SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2); |SMD| < 0.1 is treated as balanced.

| feature | mean T | mean C | SMD | var ratio | Welch p | balanced |
|---|---|---|---|---|---|---|
| f0 | 19.6148 | 19.6517 | -0.0069 | 0.995 | 5.26e-20 | yes |
| f1 | 10.0703 | 10.0679 | +0.0240 | 1.316 | 1.06e-248 | yes |
| f2 | 8.4463 | 8.4482 | -0.0062 | 0.989 | 9.06e-17 | yes |
| f3 | 4.1694 | 4.2328 | -0.0488 | 1.186 | 0 | yes |
| f4 | 10.3392 | 10.3365 | +0.0080 | 1.032 | 1.08e-26 | yes |
| f5 | 4.0266 | 4.0393 | -0.0306 | 1.221 | 0 | yes |
| f6 | -4.1828 | -3.9999 | -0.0404 | 1.073 | 0 | yes |
| f7 | 5.1056 | 5.0803 | +0.0213 | 1.086 | 1.41e-182 | yes |
| f8 | 3.9334 | 3.9347 | -0.0224 | 1.052 | 1.42e-200 | yes |
| f9 | 16.0526 | 15.8863 | +0.0240 | 1.076 | 4.01e-231 | yes |
| f10 | 5.3337 | 5.3319 | +0.0106 | 1.051 | 6.16e-46 | yes |
| f11 | -0.1710 | -0.1709 | -0.0052 | 1.037 | 4.12e-12 | yes |

- Largest |SMD|: **0.0488**; features balanced: 12/12
- Features with Welch p < 0.05: 12/12. With millions of users even a trivial difference becomes 'significant', which is why balance is judged on the effect size (SMD), not the p-value.

![Covariate balance](figures/02_covariate_balance.png)

_Runtime: 5.3s_
