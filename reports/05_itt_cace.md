# Step 5 - Assignment vs. exposure: ITT vs. CACE

## Compliance

- Treated users actually exposed to an ad: **3.60%** (428,212 of 11,882,655).
- Control users exposed: 0 → one-sided non-compliance, so there are no 'always-takers' and CACE = effect on the treated (ATT).

## Effects

| Metric | ITT | ITT 95% CI | CACE | CACE 95% CI (delta) | CACE 95% CI (bootstrap) | Exposed users' rate | Same users without ad (est.) | CACE relative lift |
|---|---|---|---|---|---|---|---|---|
| visit | +1.034 pp | [+1.006, +1.063] | **+28.700 pp** | [+27.911, +29.488] | [+27.894, +29.516] | 41.454% | 12.754% | **+225%** [+206%, +247%] |
| conversion | +0.115 pp | [+0.108, +0.122] | **+3.196 pp** | [+3.010, +3.383] | [+3.016, +3.388] | 5.378% | 2.182% | **+146%** [+129%, +169%] |

## Why not just compare exposed users with everyone else?

| Metric | Naive: exposed − control | Naive: exposed − unexposed treated | CACE (causal) | Naive / CACE |
|---|---|---|---|---|
| visit | +37.634 pp | +37.968 pp | +28.700 pp | 1.31x |
| conversion | +5.185 pp | +5.259 pp | +3.196 pp | 1.62x |

Exposure is **not** randomised - it is likely driven by user behaviour (e.g. how active a user is on sites where ads are served). Exposed vs. unexposed treated users on f0-f11:

| feature | SMD exposed vs unexposed |
|---|---|
| f3 | -1.469 |
| f6 | -1.389 |
| f8 | -1.019 |
| f9 | +0.882 |
| f0 | -0.852 |

11/12 features have |SMD| ≥ 0.1 (vs. 0/12 between randomised groups in Step 2). Exposed users are a different population, so the naive comparison mixes the ad effect with who gets exposed. CACE avoids this by only comparing *randomised* groups and rescaling by the compliance rate.

## Takeaways

- **ITT** answers the campaign question: turning ads on raises visits by 1.03 pp and conversions by 0.115 pp *per targeted user*.
- **CACE** answers the creative/media question: for users who actually saw an ad, visits rise by 28.7 pp (+225% vs. their own counterfactual) and conversions by 3.20 pp (+146%).
- CACE = ITT ÷ 0.036, so it is ~28x the ITT - and its CI is ~28x wider too: dividing by a small compliance rate amplifies noise.
- The naive exposed-vs-control comparison overstates the effect: 1.31x (visit) and 1.62x (conversion) the causal estimate. The gap is selection bias - who gets exposed - not ad impact.
- Assumptions: (1) random assignment (checked in Step 2); (2) **exclusion restriction** - assignment affects outcomes only through exposure (plausible, but violated if unlogged impressions exist); (3) monotonicity - trivially true since control cannot be exposed.

![ITT vs CACE](figures/05_itt_vs_cace.png)

_Bootstrap replicates: 2,000 · Runtime: 5.4s_
