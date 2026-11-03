# Does ML beat GARCH? Volatility forecasting with honest evaluation

A computational finance project: forecasting the volatility of the S&P 500 and 26 ETFs and large stocks with classical econometric models and machine learning, and comparing them **out of sample**, with walk-forward validation and statistical tests. Whatever the result, the method is the point: a model only counts as better if it is better on data it has never seen, by more than luck would explain.

## Highlights

- **Data:** 4,232 trading days of the S&P 500 (2010 to Oct 2026) plus the VIX. Daily returns have excess kurtosis **13.5** (normal: 0), and the VIX sits above subsequent realized vol on **86%** of days.
- **Volatility clustering:** returns themselves are nearly unpredictable, but the autocorrelation of |returns| is **0.31** at lag 1 and still **0.15** a month later, which is what makes volatility forecastable.
- **Baselines (2015–2026, 2,970 out-of-sample days):** HAR has the lowest MSE at both horizons. GARCH(1,1) has the lowest QLIKE (**3%** below HAR at 5 days, **7%** at 22 days). The two loss functions disagree, so "best model" depends on how you score it.
- **Machine learning (27 assets, 80,000 out-of-sample asset-days):** XGBoost and a small neural network do **not** beat a plain linear regression on the same features. At 5 days, QLIKE relative to HAR is **0.893** for the linear model, 0.915 for XGBoost and 0.938 for the MLP. At 22 days XGBoost loses to HAR on 15 of 27 assets and the MLP on 20. The gains over HAR come from better inputs (logs, the VIX, quarterly vol), not from non-linear models.
- **No look-ahead, tested:** a unit test scrambles all returns after a cutoff date and checks that no forecast made before it changes. Each test fails when the bug is planted on purpose.

## Phases

| Phase | Content | Status |
|---|---|---|
| 1 | Data: S&P 500 + VIX, realized volatility, stylized facts | ✅ |
| 2 | Baselines: historical vol, EWMA, GARCH(1,1), HAR | ✅ |
| 3 | ML: linear regression, XGBoost and a small neural network on vol, returns, volume and VIX features | ✅ |
| 4 | Evaluation: Diebold–Mariano tests, crisis periods | ⏳ |
| 5 | Application: volatility targeting with transaction costs (Sharpe ratio, drawdown) | ⏳ |

## Results

### Phase 1: data and stylized facts

![realized vol](results/phase1_realized_vol.png)

Realized volatility over the past $h$ days, annualized, is

$$
\mathrm{RV}_t^{(h)} = \frac{252}{h}\sum_{i=0}^{h-1} r_{t-i}^2 ,
$$

where $r_t$ is the daily log return. The plot shows the three facts every model in this project tries to exploit:

- volatility comes in **bursts** (2011, 2015, 2018, COVID 2020, 2022, April 2025), with a 22-day vol median of 12.5% and a peak of 93.5%;
- big moves are followed by big moves: |returns| stay autocorrelated for about 80 trading days;
- the VIX (the market's 30-day *implied* vol) is above subsequent realized vol on average by **3.7 vol points**: investors pay a premium for protection.

Full statistics in [results/phase1_table.md](results/phase1_table.md).

### Phase 2: baselines

![baselines](results/phase2_baselines.png)

Four forecasts of the realized variance over the next 5 and 22 days:

- **Historical:** the last 22 days' realized variance.
- **EWMA** (RiskMetrics, λ = 0.94): an exponentially weighted average of squared returns.
- **GARCH(1,1)** with Student-t errors: $\sigma^2_t = \omega + \alpha r^2_{t-1} + \beta \sigma^2_{t-1}$.
- **HAR:** a linear regression on the realized variance of the last day, week and month.

GARCH and HAR are re-estimated every 22 trading days on all data available at that date (expanding window).

| Horizon | Model | MSE / HAR | QLIKE / HAR |
|:--|:--|--:|--:|
| 5 days | Historical (22d) | 1.40 | 1.29 |
| 5 days | EWMA | 1.20 | 1.14 |
| 5 days | GARCH(1,1) | 1.02 | **0.97** |
| 5 days | HAR | **1.00** | 1.00 |
| 22 days | Historical (22d) | 1.64 | 1.44 |
| 22 days | EWMA | 1.34 | 1.31 |
| 22 days | GARCH(1,1) | 1.35 | **0.93** |
| 22 days | HAR | **1.00** | 1.00 |

- Historical vol and EWMA are clearly worse: they react late and never pull volatility back to its long-run level.
- HAR versus GARCH depends on the loss. HAR is fitted by least squares, so it is tuned for MSE, which is dominated by a few crisis days. GARCH is fitted by maximum likelihood, which is close to QLIKE, a loss that punishes *under*-predicting volatility.
- Fitted GARCH: α + β = 0.988, so a shock to variance has a half-life of about 58 days.

Whether these gaps are larger than noise is phase 4's job (Diebold–Mariano tests). Absolute losses and fitted parameters in [results/phase2_table.md](results/phase2_table.md).

### Phase 3: machine learning

![ml](results/phase3_ml.png)

The universe grows to 27 assets: the S&P 500, 11 ETFs (Nasdaq, small caps, international, emerging markets, Treasuries, gold, sectors) and 15 large stocks. Ten features, all known at the close of day $t$:

- past realized variance over 1, 5, 22 and 66 days (in logs);
- past returns over 1, 5 and 22 days (falling prices tend to raise volatility: the *leverage effect*);
- volume relative to its past month;
- the VIX level and its change over a week.

Three models learn $\log$ of the future realized variance:
- a **linear regression** on all features, as a control;
- **XGBoost** (300 trees of depth 3);
- an **MLP** (hidden layers of 32 and 16 neurons).

Each is one model pooled across all assets, re-trained once a year on every row whose target was already known. Hyperparameters were fixed in advance, never tuned on the test years. Predicting $\log$ variance gives the median, not the mean, so predictions are scaled up by Duan's smearing factor. HAR and GARCH are still fitted per asset, as in phase 2.

| Horizon | Model | QLIKE / HAR, mean (median) | beats HAR (QLIKE) | MSE / HAR, mean (median) | beats HAR (MSE) |
|:--|:--|--:|--:|--:|--:|
| 5 days | GARCH(1,1) | 0.924 (0.952) | 26/27 | **0.944** (0.955) | 18/27 |
| 5 days | Linear (all features) | **0.893** (0.910) | 26/27 | 0.968 (0.957) | 19/27 |
| 5 days | XGBoost | 0.915 (0.930) | 26/27 | 1.010 (1.009) | 13/27 |
| 5 days | MLP | 0.938 (0.948) | 23/27 | 1.171 (1.127) | 5/27 |
| 22 days | GARCH(1,1) | **0.929** (0.919) | 24/27 | 1.027 (1.012) | 12/27 |
| 22 days | Linear (all features) | 0.994 (0.970) | 19/27 | **0.953** (0.956) | 22/27 |
| 22 days | XGBoost | 1.037 (1.002) | 12/27 | 1.098 (1.044) | 9/27 |
| 22 days | MLP | 1.104 (1.085) | 7/27 | 1.640 (1.650) | 0/27 |

- **ML doesn't beat the linear model.** At 5 days XGBoost comes close to it on QLIKE but never passes it, and the MLP is behind on every measure. At 22 days both are worse than plain HAR. With about 250 new rows per asset per year and a very noisy target, there is little non-linear structure left to find, and flexible models mostly fit noise.
- **The inputs matter more than the model.** The linear regression beats HAR on 26 of 27 assets (5-day QLIKE) because it works in logs and sees the VIX and longer-term vol. XGBoost confirms this: 83% of its gain comes from past volatility, 12% from the VIX, and only about 5% from returns and volume.
- **GARCH is still the best single model on QLIKE at 22 days**, ahead of every ML model.
- **A fragile baseline nearly fooled me.** A linear HAR can predict a variance below zero right after a one-day crash, and QLIKE then explodes. That made every model look much better than HAR on XOM, CVX and XLE. HAR now falls back to the past month's variance when its forecast drops below anything seen in training (an "insanity filter").

Which of these gaps are real, and how the models do in the 2020 and 2022 crises, is phase 4. Full table and feature importance in [results/phase3_table.md](results/phase3_table.md).

## Usage

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m experiments.phase1_data        # downloads data/ on the first run (Yahoo Finance), then works offline
python -m experiments.phase2_baselines
python -m experiments.phase3_ml          # ~15 minutes (GARCH on 27 assets); caches forecasts in data/
```

## Layout

```
volfc/          Python package: data and realized vol, baselines, features, walk-forward ML, loss functions
experiments/    one script per phase; each regenerates its plot and table in results/
tests/          pytest suite
data/           cached Yahoo Finance downloads and forecasts (git-ignored; prices up to 2026-10-30)
```
