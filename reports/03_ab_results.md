# Step 3 - Core A/B results

Treatment n = 11,882,655 · Control n = 2,096,937 · bootstrap replicates = 10,000

| Metric | Control | Treatment | Abs. lift | 95% CI (z) | Rel. lift | 95% CI (delta) | 95% CI (bootstrap) | z | p-value |
|---|---|---|---|---|---|---|---|---|---|
| visit | 3.8201% | 4.8543% | +1.034 pp | [+1.006, +1.063] pp | **+27.07%** | [+26.16%, +28.00%] | [+26.16%, +28.00%] | 65.2 | 0 |
| conversion | 0.1938% | 0.3089% | +0.115 pp | [+0.108, +0.122] pp | **+59.45%** | [+54.37%, +64.69%] | [+54.55%, +64.65%] | 28.5 | 7.31e-179 |

## Reading the results

- **Visits:** ads lift the visit rate by 1.034 pp (+27.1%). Across the 11,882,655 treated users that is roughly **122,895 incremental visits**.
- **Conversions:** 0.115 pp (+59.4%), roughly **13,687 incremental conversions**.
- **Signal vs noise:** conversion is ~20x rarer than a visit, so its relative-lift CI is 5.6x wider (10.3% vs 1.8%). Visit is the stronger, more stable signal; conversion is the business outcome but noisier.
- **Cross-check:** bootstrap and analytic CIs agree (largest endpoint gap: visit 0.00 pts, conversion 0.18 pts), so the normal approximation holds at this sample size.
- These are **intent-to-treat (ITT)** effects: they compare users *assigned* to see ads, most of whom never saw one. Step 5 scales them to users actually exposed.

![Relative lift with 95% CIs](figures/03_lift_ci.png)

_Runtime: 3.1s_
