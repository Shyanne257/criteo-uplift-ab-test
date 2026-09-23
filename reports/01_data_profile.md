# Step 1 - Data profile

- Rows read: **13,979,592**; rows written to Parquet: **13,979,592** (dropped for missing values: 0)
- Treatment group: 11,882,655 (85.00%); control group: 2,096,937
- Overall visit rate 4.6992%, conversion rate 0.2917%, exposure rate 3.0631%

## Data quality checks

| Check | Result |
|---|---|
| Missing values (all columns) | 0 |
| `treatment` values | [0, 1] OK |
| `conversion` values | [0, 1] OK |
| `visit` values | [0, 1] OK |
| `exposure` values | [0, 1] OK |
| Control users with exposure = 1 | 0 |
| Conversions without a visit | 0 |
| Fully duplicated rows | 1,259,545 (9.01%) |
| Duplicated feature vectors | 1,626,259 (11.63%) |

Duplicates are kept on purpose: each row is a distinct user and the features are
anonymised/discretised, so identical rows are expected and are not data errors.

## Feature summary

| feature | mean | std | min | p25 | median | p75 | max | ~unique |
|---|---|---|---|---|---|---|---|---|
| f0 | 19.620 | 5.377 | 12.616 | 12.616 | 21.923 | 24.436 | 26.745 | 2,179,829 |
| f1 | 10.070 | 0.105 | 10.060 | 10.060 | 10.060 | 10.060 | 16.344 | 62 |
| f2 | 8.447 | 0.299 | 8.214 | 8.214 | 8.214 | 8.723 | 9.052 | 2,155,702 |
| f3 | 4.179 | 1.337 | -8.398 | 4.680 | 4.680 | 4.680 | 4.680 | 547 |
| f4 | 10.339 | 0.343 | 10.281 | 10.281 | 10.281 | 10.281 | 21.124 | 252 |
| f5 | 4.029 | 0.431 | -9.012 | 4.115 | 4.115 | 4.115 | 4.115 | 129 |
| f6 | -4.155 | 4.578 | -31.430 | -6.699 | -2.411 | 0.294 | 0.294 | 1,617 |
| f7 | 5.102 | 1.205 | 4.834 | 4.834 | 4.834 | 4.834 | 11.998 | 653,377 |
| f8 | 3.934 | 0.057 | 3.635 | 3.911 | 3.972 | 3.972 | 3.972 | 3,486 |
| f9 | 16.028 | 7.019 | 13.190 | 13.190 | 13.190 | 13.190 | 75.295 | 1,581 |
| f10 | 5.333 | 0.168 | 5.300 | 5.300 | 5.300 | 5.300 | 6.474 | 507,246 |
| f11 | -0.171 | 0.023 | -1.384 | -0.169 | -0.169 | -0.169 | -0.169 | 129 |

_Runtime: 60.5s_
