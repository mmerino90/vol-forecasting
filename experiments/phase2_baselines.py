"""Phase 2: baseline forecasts for the S&P 500 (historical vol, EWMA, GARCH(1,1), HAR).

Walk-forward: the first five years are training only; from 2015 on, HAR and GARCH are re-estimated
every 22 trading days using only data available at that date.

Produces:
  results/phase2_baselines.png  (forecasts vs realized vol in 2020, losses relative to HAR)
  results/phase2_table.md       (MSE and QLIKE out of sample, fitted parameters)
Run from the repo root:  python -m experiments.phase2_baselines
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from volfc import (ewma_forecast, future_realized_var, garch_forecast, har_features, har_forecast,
                   hist_forecast, load_prices, log_returns, mse, qlike)

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
OOS_START = "2015-01-01"
HORIZONS = (5, 22)

r = log_returns(load_prices("^GSPC")["Close"])

forecasts, targets = {}, {}
for h in HORIZONS:
    y = future_realized_var(r, h)
    fc = pd.DataFrame({
        "Historical (22d)": hist_forecast(r),
        "EWMA (λ=0.94)": ewma_forecast(r),
        "GARCH(1,1)": garch_forecast(r, h, OOS_START),
        "HAR": har_forecast(r, h, OOS_START),
    })
    # Same out-of-sample days for every model: from OOS_START until the last fully observed target
    fc = fc[fc.index >= OOS_START].join(y.rename("y")).dropna()
    targets[h], forecasts[h] = fc.pop("y"), fc

# Loss table: absolute values and ratios to HAR (below 1 = better than HAR)
rows = ["| Horizon | Model | MSE (×10⁴) | MSE / HAR | QLIKE | QLIKE / HAR |",
        "|:--|:--|--:|--:|--:|--:|"]
ratios = {}
for h in HORIZONS:
    y, fc = targets[h], forecasts[h]
    m = {k: mse(y, f) for k, f in fc.items()}
    q = {k: qlike(y, f) for k, f in fc.items()}
    ratios[h] = {k: (m[k] / m["HAR"], q[k] / q["HAR"]) for k in fc}
    for k in fc:
        rows.append(f"| {h} days | {k} | {1e4 * m[k]:.2f} | {ratios[h][k][0]:.2f} | {q[k]:.3f} | {ratios[h][k][1]:.2f} |")
n_days = len(targets[5])
period = f"{targets[5].index[0]:%Y-%m-%d} to {targets[5].index[-1]:%Y-%m-%d}"

# Parameters fitted on the full sample (in-sample, only to read what the models learned)
X = har_features(r).join(future_realized_var(r, 5).rename("y")).dropna()
A = np.column_stack([np.ones(len(X)), X[["rv_d", "rv_w", "rv_m"]]])
b = np.linalg.lstsq(A, X.y, rcond=None)[0]
g = arch_model(100 * r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off").params
params = (f"HAR, 5-day target: b0 = {b[0]:.4f}, b_d = {b[1]:.3f}, b_w = {b[2]:.3f}, b_m = {b[3]:.3f}\n\n"
          f"GARCH(1,1): ω = {g['omega']:.4f}, α = {g['alpha[1]']:.3f}, β = {g['beta[1]']:.3f}, "
          f"α + β = {g['alpha[1]'] + g['beta[1]']:.3f} (half-life of a shock: "
          f"{np.log(0.5) / np.log(g['alpha[1]'] + g['beta[1]']):.0f} days), Student-t ν = {g['nu']:.1f}")
table = (f"Out of sample: {period} ({n_days:,} forecast days). Target: annualized realized variance "
         f"over the next h days, from daily squared returns.\n\n" + "\n".join(rows) +
         "\n\nFull-sample parameters (in-sample, for interpretation only):\n\n" + params)
(OUT / "phase2_table.md").write_text(table + "\n", encoding="utf-8")

# Plot: 2020 zoom of the 5-day forecasts, then loss ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
zoom = slice("2020-01-15", "2020-07-31")
ax1.plot(100 * np.sqrt(targets[5][zoom]), color="k", lw=1.5, label="realized, next 5 days")
for k, f in forecasts[5].items():
    ax1.plot(100 * np.sqrt(f[zoom]), lw=1, label=k)
ax1.set(ylabel="annualized vol (%)", title="5-day-ahead forecasts during COVID 2020")
ax1.legend(fontsize=8)

models = list(forecasts[5].columns)
x = np.arange(len(models))
for j, (h, metric) in enumerate([(5, 0), (5, 1), (22, 0), (22, 1)]):
    vals = [ratios[h][k][metric] for k in models]
    ax2.bar(x + (j - 1.5) * 0.2, vals, width=0.2, label=f"{h}d {'MSE' if metric == 0 else 'QLIKE'}")
ax2.axhline(1, color="k", lw=0.8)
ax2.set_xticks(x, models, rotation=15, fontsize=8)
ax2.set(ylabel="loss / HAR loss", title="Out-of-sample loss relative to HAR (lower is better)")
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase2_baselines.png", dpi=150)
print(table)
