## Per asset: each model against HAR (Diebold–Mariano, 5% level)

Number of the 27 assets where the model is significantly better / not distinguishable / significantly worse.
By luck alone, about 1 asset in 27 (5% split over two tails, so ~0.7 per tail) would come out significant.

| Horizon | Model | QLIKE: better / tie / worse | MSE: better / tie / worse |
|:--|:--|:-:|:-:|
| 5 days | GARCH(1,1) | 13 / 14 / 0 | 0 / 27 / 0 |
| 5 days | Linear (all features) | 16 / 10 / 1 | 0 / 27 / 0 |
| 5 days | XGBoost | 10 / 16 / 1 | 2 / 23 / 2 |
| 5 days | MLP | 6 / 20 / 1 | 0 / 25 / 2 |
| 22 days | GARCH(1,1) | 3 / 24 / 0 | 0 / 27 / 0 |
| 22 days | Linear (all features) | 1 / 25 / 1 | 0 / 27 / 0 |
| 22 days | XGBoost | 0 / 25 / 2 | 0 / 27 / 0 |
| 22 days | MLP | 0 / 25 / 2 | 0 / 27 / 0 |

## All 27 assets pooled (QLIKE averaged across assets each day)

| Horizon | Comparison | Loss ratio | DM t-stat | p-value |
|:--|:--|--:|--:|--:|
| 5 days | GARCH(1,1) vs HAR | 0.910 | -2.43 | 0.015 ✔ |
| 5 days | Linear (all features) vs HAR | 0.879 | -2.75 | 0.006 ✔ |
| 5 days | XGBoost vs HAR | 0.902 | -2.40 | 0.017 ✔ |
| 5 days | MLP vs HAR | 0.923 | -1.92 | 0.055 |
| 5 days | XGBoost vs Linear (all features) | 1.026 | +2.67 | 0.008 ✔ |
| 5 days | MLP vs Linear (all features) | 1.050 | +3.56 | 0.000 ✔ |
| 5 days | GARCH(1,1) vs Linear (all features) | 1.035 | +1.89 | 0.058 |
| 22 days | GARCH(1,1) vs HAR | 0.926 | -2.02 | 0.043 ✔ |
| 22 days | Linear (all features) vs HAR | 0.988 | -0.32 | 0.750 |
| 22 days | XGBoost vs HAR | 1.034 | +0.76 | 0.448 |
| 22 days | MLP vs HAR | 1.097 | +1.60 | 0.110 |
| 22 days | XGBoost vs Linear (all features) | 1.046 | +1.96 | 0.050 |
| 22 days | MLP vs Linear (all features) | 1.110 | +2.48 | 0.013 ✔ |
| 22 days | GARCH(1,1) vs Linear (all features) | 0.938 | -1.45 | 0.146 |

## By period: pooled QLIKE relative to HAR (forecast date inside the period)

| Horizon | Period | Days | GARCH(1,1) | Linear (all features) | XGBoost | MLP | HAR QLIKE |
|:--|:--|--:|--:|--:|--:|--:|--:|
| 5 days | COVID 2020 | 94 | 0.505 | 0.436 | 0.529 | 0.618 | 1.889 |
| 5 days | 2022 bear market | 251 | 0.867* | 0.763* | 0.774* | 0.803* | 0.413 |
| 5 days | Tariff shock 2025 | 47 | 1.012 | 0.742 | 0.827 | 0.841 | 1.163 |
| 5 days | Calm (everything else) | 2578 | 0.968* | 0.960* | 0.970 | 0.982 | 0.470 |
| 22 days | COVID 2020 | 94 | 0.835 | 0.849 | 0.953 | 1.022 | 2.466 |
| 22 days | 2022 bear market | 251 | 0.834* | 0.661* | 0.699* | 0.734* | 0.213 |
| 22 days | Tariff shock 2025 | 47 | 1.149* | 1.010 | 1.067 | 1.104 | 0.527 |
| 22 days | Calm (everything else) | 2561 | 0.956* | 1.057 | 1.084 | 1.149 | 0.277 |

\* significantly different from HAR at 5% (Diebold–Mariano on the pooled daily loss within the period).
