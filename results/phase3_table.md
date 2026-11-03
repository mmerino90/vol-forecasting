Out of sample 2015-01-02 to 2026-10-23, 27 assets, 80,189 asset-days at 5 days. Each ratio is a model's loss divided by HAR's on the same asset (below 1 = better than HAR), then averaged across assets.

| Horizon | Model | MSE / HAR, mean (median) | beats HAR on MSE | QLIKE / HAR, mean (median) | beats HAR on QLIKE | S&P 500 QLIKE / HAR |
|:--|:--|--:|--:|--:|--:|--:|
| 5 days | GARCH(1,1) | 0.944 (0.955) | 18/27 | 0.924 (0.952) | 26/27 | 0.965 |
| 5 days | Linear (all features) | 0.968 (0.957) | 19/27 | 0.893 (0.910) | 26/27 | 0.947 |
| 5 days | XGBoost | 1.010 (1.009) | 13/27 | 0.915 (0.930) | 26/27 | 0.942 |
| 5 days | MLP | 1.171 (1.127) | 5/27 | 0.938 (0.948) | 23/27 | 0.968 |
| 22 days | GARCH(1,1) | 1.027 (1.012) | 12/27 | 0.929 (0.919) | 24/27 | 0.930 |
| 22 days | Linear (all features) | 0.953 (0.956) | 22/27 | 0.994 (0.970) | 19/27 | 1.065 |
| 22 days | XGBoost | 1.098 (1.044) | 9/27 | 1.037 (1.002) | 12/27 | 1.147 |
| 22 days | MLP | 1.640 (1.650) | 0/27 | 1.104 (1.085) | 7/27 | 1.190 |

XGBoost feature importance (share of total gain, 5-day target):

- log_rv_m: 44.4%
- log_rv_q: 26.5%
- log_rv_w: 12.1%
- log_vix: 8.3%
- vix_change_w: 3.6%
- r_m: 1.9%
- r_w: 1.7%
- log_volume_ratio: 1.1%
- r_d: 0.3%
- log_rv_d: 0.1%
