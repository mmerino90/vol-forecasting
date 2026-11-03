Out of sample: 2015-01-02 to 2026-10-23 (2,970 forecast days). Target: annualized realized variance over the next h days, from daily squared returns.

| Horizon | Model | MSE (×10⁴) | MSE / HAR | QLIKE | QLIKE / HAR |
|:--|:--|--:|--:|--:|--:|
| 5 days | Historical (22d) | 86.36 | 1.40 | 0.629 | 1.29 |
| 5 days | EWMA (λ=0.94) | 74.26 | 1.20 | 0.556 | 1.14 |
| 5 days | GARCH(1,1) | 63.12 | 1.02 | 0.472 | 0.97 |
| 5 days | HAR | 61.76 | 1.00 | 0.489 | 1.00 |
| 22 days | Historical (22d) | 75.47 | 1.64 | 0.652 | 1.44 |
| 22 days | EWMA (λ=0.94) | 61.39 | 1.34 | 0.593 | 1.31 |
| 22 days | GARCH(1,1) | 61.81 | 1.35 | 0.420 | 0.93 |
| 22 days | HAR | 45.91 | 1.00 | 0.451 | 1.00 |

Full-sample parameters (in-sample, for interpretation only):

HAR, 5-day target: b0 = 0.0110, b_d = 0.127, b_w = 0.431, b_m = 0.067

GARCH(1,1): ω = 0.0258, α = 0.161, β = 0.827, α + β = 0.988 (half-life of a shock: 58 days), Student-t ν = 5.4
