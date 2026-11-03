"""Phase 4: are the differences between models real, and who wins in a crisis?

Reuses the out-of-sample forecasts cached by phase 3 (run that first).
- Diebold–Mariano tests per asset, each model against HAR, on QLIKE and MSE.
- A pooled test across all 27 assets: the daily loss difference averaged over assets.
- Results split by period: COVID 2020, the 2022 bear market, the April 2025 tariff shock, and calm days.

Produces:
  results/phase4_evaluation.png  (DM statistics per asset, QLIKE by period)
  results/phase4_table.md
Run from the repo root:  python -m experiments.phase4_evaluation
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volfc import dm_test, qlike_loss, squared_error

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
HORIZONS = (5, 22)
MODELS = ["GARCH(1,1)", "Linear (all features)", "XGBoost", "MLP"]
PAIRS = [(m, "HAR") for m in MODELS] + [("XGBoost", "Linear (all features)"), ("MLP", "Linear (all features)"),
                                        ("GARCH(1,1)", "Linear (all features)")]
PERIODS = {"COVID 2020": ("2020-02-15", "2020-06-30"), "2022 bear market": ("2022-01-01", "2022-12-31"),
           "Tariff shock 2025": ("2025-03-25", "2025-05-31")}
ALPHA = 0.05


def losses(fc, loss):
    return pd.DataFrame({m: loss(fc.y, fc[m]) for m in MODELS + ["HAR"]}, index=fc.index)


def period_of(dates):
    label = pd.Series("Calm (everything else)", index=dates)
    for name, (a, b) in PERIODS.items():
        label[(dates >= a) & (dates <= b)] = name
    return label.to_numpy()


forecasts = {h: pd.read_pickle(Path("data") / f"forecasts_h{h}.pkl") for h in HORIZONS}
lines = []

# 1) Per asset: is each model significantly better or worse than HAR?
lines += ["## Per asset: each model against HAR (Diebold–Mariano, 5% level)", "",
          "Number of the 27 assets where the model is significantly better / not distinguishable / significantly worse.",
          "By luck alone, about 1 asset in 27 (5% split over two tails, so ~0.7 per tail) would come out significant.", "",
          "| Horizon | Model | QLIKE: better / tie / worse | MSE: better / tie / worse |", "|:--|:--|:-:|:-:|"]
tstats = {}
for h, fc in forecasts.items():
    counts = {}
    for lname, loss in (("QLIKE", qlike_loss), ("MSE", squared_error)):
        L = losses(fc, loss)
        t = pd.DataFrame({m: L.groupby(level="ticker").apply(lambda d: dm_test(d[m], d.HAR, 2 * h)[0])
                          for m in MODELS})
        tstats[h, lname] = t
        crit = 1.96  # two-sided 5%
        counts[lname] = {m: f"{(t[m] < -crit).sum()} / {(t[m].abs() <= crit).sum()} / {(t[m] > crit).sum()}"
                         for m in MODELS}
    lines += [f"| {h} days | {m} | {counts['QLIKE'][m]} | {counts['MSE'][m]} |" for m in MODELS]

# 2) Pooled over assets: average the daily QLIKE across the 27 assets, then test.
#    QLIKE only: it is scale-free, so a calm utility and a volatile chip stock count the same.
#    (Pooled MSE would be dominated by NVDA-like assets.)
lines += ["", "## All 27 assets pooled (QLIKE averaged across assets each day)", "",
          "| Horizon | Comparison | Loss ratio | DM t-stat | p-value |", "|:--|:--|--:|--:|--:|"]
pooled = {}
for h, fc in forecasts.items():
    daily = losses(fc, qlike_loss).groupby(level="date").mean()
    pooled[h] = daily
    for a, b in PAIRS:
        t, p = dm_test(daily[a], daily[b], 2 * h)
        star = " ✔" if p < ALPHA else ""
        lines.append(f"| {h} days | {a} vs {b} | {daily[a].mean() / daily[b].mean():.3f} | {t:+.2f} | {p:.3f}{star} |")

# 3) By period: does the ranking survive a crisis?
lines += ["", "## By period: pooled QLIKE relative to HAR (forecast date inside the period)", "",
          "| Horizon | Period | Days | " + " | ".join(MODELS) + " | HAR QLIKE |",
          "|:--|:--|--:|" + "--:|" * len(MODELS) + "--:|"]
by_period = {}
for h, daily in pooled.items():
    g = daily.groupby(period_of(daily.index))
    means, n = g.mean(), g.size()
    by_period[h] = means[MODELS].div(means.HAR, axis=0)
    for per in list(PERIODS) + ["Calm (everything else)"]:
        cells = []
        for m in MODELS:
            d = daily[period_of(daily.index) == per]
            p = dm_test(d[m], d.HAR, 2 * h)[1]
            cells.append(f"{by_period[h].loc[per, m]:.3f}{'*' if p < ALPHA else ''}")
        lines.append(f"| {h} days | {per} | {n[per]} | {' | '.join(cells)} | {means.loc[per, 'HAR']:.3f} |")
lines += ["", "\\* significantly different from HAR at 5% (Diebold–Mariano on the pooled daily loss within the period)."]

table = "\n".join(lines)
(OUT / "phase4_table.md").write_text(table + "\n", encoding="utf-8")

# Plot: DM t-statistics per asset (QLIKE) for both horizons, then QLIKE ratio by period
fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1, 1, 1.3]})
for ax, h in zip(axes[:2], HORIZONS):
    t = tstats[h, "QLIKE"].sort_values("Linear (all features)")
    im = ax.imshow(t.clip(-5, 5), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    for (i, j), v in np.ndenumerate(t.to_numpy()):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                fontweight="bold" if abs(v) > 1.96 else "normal")
    ax.set_xticks(range(len(MODELS)), [m.replace(" (all features)", "") for m in MODELS], fontsize=8)
    ax.set_yticks(range(len(t)), t.index, fontsize=7)
    ax.set_title(f"{h} days: DM t-stat vs HAR (QLIKE)\nblue = better than HAR, bold = significant", fontsize=10)
fig.colorbar(im, ax=axes[1], shrink=0.6)

ax = axes[2]
r = by_period[5].loc[list(PERIODS) + ["Calm (everything else)"]]
x = np.arange(len(r))
for j, m in enumerate(MODELS):
    ax.bar(x + (j - 1.5) * 0.2, r[m], width=0.2, label=m)
ax.axhline(1, color="k", lw=0.8)
ax.set_xticks(x, r.index, fontsize=8)
ax.set(ylabel="pooled QLIKE / HAR QLIKE", title="5 days: who wins in a crisis? (below 1 beats HAR)",
       ylim=(0.3, 1.15))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase4_evaluation.png", dpi=150)
print(table)
