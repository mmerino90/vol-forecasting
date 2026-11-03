## 5-day forecasts, rebalanced daily, 10 bp per trade (2015-01-02 to 2026-10-30)

### Average over the 27 assets (each traded on its own)

| Strategy | Return / yr | Vol | Sharpe | Max drawdown | Turnover / yr | Vol of vol | Sharpe > HAR |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Buy & hold | +16.3% | 24.5% | 0.64 | -43% | 0.1x | 0.43 | 22/27 |
| Constant 10% (2010-14 vol) | +7.6% | 11.9% | 0.64 | -24% | 0.0x | 0.43 | 22/27 |
| HAR | +5.9% | 10.3% | 0.57 | -21% | 8.6x | 0.20 |  |
| GARCH(1,1) | +6.0% | 10.2% | 0.59 | -20% | 5.6x | 0.15 | 15/27 |
| Linear (all features) | +5.6% | 9.9% | 0.55 | -20% | 9.1x | 0.15 | 8/27 |
| XGBoost | +5.5% | 10.1% | 0.53 | -21% | 11.0x | 0.15 | 7/27 |
| MLP | +5.0% | 10.1% | 0.49 | -22% | 11.9x | 0.16 | 4/27 |
| Oracle (knows the future) | +10.5% | 9.9% | 1.06 | -18% | 29.8x | 0.03 | 24/27 |

### Equal-weight portfolio of the 27 assets

| Strategy | Return / yr | Vol | Sharpe | Max drawdown | Turnover / yr | Vol of vol | p-value Sharpe vs HAR |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Buy & hold | +16.3% | 15.8% | 1.03 | -30% | 0.1x | 0.57 | 0.610 |
| Constant 10% (2010-14 vol) | +7.6% | 7.5% | 1.01 | -16% | 0.0x | 0.59 | 0.778 |
| HAR | +5.9% | 6.1% | 0.97 | -9% | 8.6x | 0.30 |  |
| GARCH(1,1) | +6.0% | 5.9% | 1.01 | -9% | 5.6x | 0.25 | 0.196 |
| Linear (all features) | +5.6% | 5.6% | 0.99 | -8% | 9.1x | 0.21 | 0.797 |
| XGBoost | +5.5% | 5.8% | 0.95 | -9% | 11.0x | 0.22 | 0.680 |
| MLP | +5.0% | 5.7% | 0.87 | -10% | 11.9x | 0.23 | 0.102 |
| Oracle (knows the future) | +10.5% | 5.7% | 1.84 | -7% | 29.8x | 0.17 | 0.000 |

### Portfolio Sharpe ratio by cost (bp per unit traded)

| Cost | Buy & hold | Constant 10% (2010-14 vol) | HAR | GARCH(1,1) | Linear (all features) | XGBoost | MLP | Oracle (knows the future) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 bp | 1.03 | 1.01 | 1.11 | 1.10 | 1.15 | 1.14 | 1.08 | 2.36 |
| 5 bp | 1.03 | 1.01 | 1.04 | 1.06 | 1.07 | 1.05 | 0.98 | 2.10 |
| 10 bp | 1.03 | 1.01 | 0.97 | 1.01 | 0.99 | 0.95 | 0.87 | 1.84 |
| 20 bp | 1.03 | 1.00 | 0.83 | 0.91 | 0.82 | 0.76 | 0.67 | 1.32 |
| 40 bp | 1.03 | 1.00 | 0.55 | 0.72 | 0.50 | 0.37 | 0.25 | 0.28 |

## 22-day forecasts, rebalanced daily, 10 bp per trade (2015-01-02 to 2026-10-30)

### Average over the 27 assets (each traded on its own)

| Strategy | Return / yr | Vol | Sharpe | Max drawdown | Turnover / yr | Vol of vol | Sharpe > HAR |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Buy & hold | +16.3% | 24.5% | 0.64 | -43% | 0.1x | 0.43 | 18/27 |
| Constant 10% (2010-14 vol) | +7.6% | 11.9% | 0.64 | -24% | 0.0x | 0.43 | 18/27 |
| HAR | +6.2% | 10.1% | 0.61 | -20% | 4.7x | 0.23 |  |
| GARCH(1,1) | +6.0% | 9.9% | 0.60 | -20% | 4.0x | 0.17 | 12/27 |
| Linear (all features) | +5.6% | 9.6% | 0.57 | -19% | 5.9x | 0.17 | 7/27 |
| XGBoost | +5.4% | 9.8% | 0.54 | -20% | 7.5x | 0.17 | 3/27 |
| MLP | +5.1% | 9.7% | 0.52 | -21% | 8.7x | 0.18 | 1/27 |
| Oracle (knows the future) | +11.1% | 10.1% | 1.09 | -15% | 5.4x | 0.07 | 26/27 |

### Equal-weight portfolio of the 27 assets

| Strategy | Return / yr | Vol | Sharpe | Max drawdown | Turnover / yr | Vol of vol | p-value Sharpe vs HAR |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Buy & hold | +16.3% | 15.8% | 1.03 | -30% | 0.1x | 0.57 | 0.869 |
| Constant 10% (2010-14 vol) | +7.6% | 7.5% | 1.01 | -16% | 0.0x | 0.59 | 0.908 |
| HAR | +6.2% | 6.1% | 1.02 | -10% | 4.7x | 0.35 |  |
| GARCH(1,1) | +6.0% | 5.8% | 1.03 | -9% | 4.0x | 0.29 | 0.581 |
| Linear (all features) | +5.6% | 5.6% | 1.00 | -10% | 5.9x | 0.28 | 0.602 |
| XGBoost | +5.4% | 5.7% | 0.95 | -10% | 7.5x | 0.28 | 0.087 |
| MLP | +5.1% | 5.6% | 0.91 | -9% | 8.7x | 0.27 | 0.022 |
| Oracle (knows the future) | +11.1% | 6.0% | 1.86 | -6% | 5.4x | 0.19 | 0.000 |

### Portfolio Sharpe ratio by cost (bp per unit traded)

| Cost | Buy & hold | Constant 10% (2010-14 vol) | HAR | GARCH(1,1) | Linear (all features) | XGBoost | MLP | Oracle (knows the future) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 bp | 1.03 | 1.01 | 1.09 | 1.10 | 1.10 | 1.08 | 1.06 | 1.95 |
| 5 bp | 1.03 | 1.01 | 1.06 | 1.07 | 1.05 | 1.01 | 0.98 | 1.91 |
| 10 bp | 1.03 | 1.01 | 1.02 | 1.03 | 1.00 | 0.95 | 0.91 | 1.86 |
| 20 bp | 1.03 | 1.00 | 0.94 | 0.96 | 0.89 | 0.81 | 0.75 | 1.77 |
| 40 bp | 1.03 | 1.00 | 0.78 | 0.82 | 0.68 | 0.55 | 0.44 | 1.59 |

Vol of vol = std / mean of the 3-month rolling volatility: lower means steadier risk.
p-value: circular block bootstrap (22-day blocks) of the difference in Sharpe ratios.
