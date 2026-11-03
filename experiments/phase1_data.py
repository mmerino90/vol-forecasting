"""Phase 1: S&P 500 data, realized volatility and the stylized facts that motivate the models.

Produces:
  results/phase1_realized_vol.png  (price, realized vol vs VIX, autocorrelation of returns vs |returns|)
  results/phase1_table.md          (summary statistics)
Run from the repo root:  python -m experiments.phase1_data
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

from volfc import load_prices, log_returns, realized_var

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
CRISES = {"COVID 2020": ("2020-02-15", "2020-06-30"), "2022 bear market": ("2022-01-01", "2022-12-31")}

spx = load_prices("^GSPC")
vix = load_prices("^VIX")["Close"]
r = log_returns(spx["Close"])
vol5 = 100 * np.sqrt(realized_var(r, 5))  # annualized, in %
vol22 = 100 * np.sqrt(realized_var(r, 22))
vix = vix.reindex(r.index)

# Stylized facts: returns are (almost) unpredictable, their size is not
LAGS = 100
acf_r = acf(r, nlags=LAGS)[1:]
acf_abs = acf(r.abs(), nlags=LAGS)[1:]
band = 1.96 / np.sqrt(len(r))  # 95% band for "no autocorrelation"

fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(3, 1, 1)
ax1.semilogy(spx["Close"], color="k", lw=0.8)
ax1.set(title="S&P 500 (log scale)", ylabel="index level")

ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
ax2.plot(vol5, lw=0.5, alpha=0.6, label="realized vol, 5 days")
ax2.plot(vol22, lw=1.0, label="realized vol, 22 days")
ax2.plot(vix, lw=0.8, color="k", alpha=0.7, label="VIX (implied, 30 days ahead)")
ax2.set(ylabel="annualized vol (%)", title="Volatility comes in bursts", ylim=(0, 100))
for name, (a, b) in CRISES.items():
    for ax in (ax1, ax2):
        ax.axvspan(a, b, color="red", alpha=0.12)
    ax2.text(np.datetime64(a), 92, name, fontsize=8, color="darkred")
ax2.legend(loc="upper left", fontsize=8)

ax3 = fig.add_subplot(3, 1, 3)
lags = np.arange(1, LAGS + 1)
ax3.bar(lags, acf_r, width=0.8, alpha=0.6, label="returns $r_t$")
ax3.plot(lags, acf_abs, "o-", ms=2, color="C3", label="absolute returns $|r_t|$")
ax3.axhspan(-band, band, color="gray", alpha=0.3, label="95% band if no autocorrelation")
ax3.set(xlabel="lag (trading days)", ylabel="autocorrelation",
        title="Returns barely predict returns; big moves predict big moves (volatility clustering)")
ax3.legend(fontsize=8)
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase1_realized_vol.png", dpi=150)

# Summary table
both = vol22.to_frame("rv22").join(vix.rename("vix")).dropna()
stats = {
    "Sample": f"{r.index[0]:%Y-%m-%d} to {r.index[-1]:%Y-%m-%d} ({len(r):,} trading days)",
    "Annualized vol of daily returns": f"{100 * r.std() * np.sqrt(252):.1f}%",
    "Excess kurtosis of daily returns (normal = 0)": f"{r.kurt():.1f}",
    "Worst day": f"{100 * r.min():.1f}% ({r.idxmin():%Y-%m-%d})",
    "22-day realized vol: median / max": f"{vol22.median():.1f}% / {vol22.max():.1f}% ({vol22.idxmax():%Y-%m-%d})",
    "Mean VIX minus mean 22-day realized vol": f"{both.vix.mean() - both.rv22.mean():+.1f} vol points",
    "Days with VIX above 22-day realized vol": f"{(both.vix > both.rv22).mean():.0%}",
    "Autocorrelation of r_t, lag 1 / lag 22": f"{acf_r[0]:+.3f} / {acf_r[21]:+.3f}",
    "Autocorrelation of abs(r_t), lag 1 / lag 22 / lag 100": f"{acf_abs[0]:+.3f} / {acf_abs[21]:+.3f} / {acf_abs[99]:+.3f}",
}
table = "| Statistic | Value |\n|:--|--:|\n" + "\n".join(f"| {k} | {v} |" for k, v in stats.items())
(OUT / "phase1_table.md").write_text(table + "\n", encoding="utf-8")
print(table)
