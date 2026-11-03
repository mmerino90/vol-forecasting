"""Phase 3: do XGBoost and a neural network beat HAR and GARCH on 27 assets?

ML models are trained on all 27 assets pooled (one model, ~80,000 rows by the end) and re-trained
once a year; HAR and GARCH are fitted per asset and re-estimated monthly, as in phase 2.
Out of sample from 2015. All forecasts are cached in data/forecasts_h{h}.pkl for phase 4.

Produces:
  results/phase3_ml.png     (per-asset loss relative to HAR, XGBoost feature importance)
  results/phase3_table.md   (average loss ratios and how many assets each model wins)
Run from the repo root:  python -m experiments.phase3_ml
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from volfc import (FEATURES, MODELS, UNIVERSE, build_panel, garch_forecast, har_forecast,
                   load_prices, log_returns, mse, qlike, walk_forward)

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
OOS_START = "2015-01-01"
HORIZONS = (5, 22)


def baselines(ticker, h):
    warnings.simplefilter("ignore")  # arch warns when an optimizer step is imprecise
    r = log_returns(load_prices(ticker)["Close"])
    return ticker, pd.DataFrame({"HAR": har_forecast(r, h, OOS_START),
                                 "GARCH(1,1)": garch_forecast(r, h, OOS_START)})


def all_forecasts(h):
    """DataFrame indexed by (date, ticker): target y plus one column per model."""
    path = Path("data") / f"forecasts_h{h}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    panel = build_panel(h)
    fc = panel[["y"]].copy()
    for name, make in MODELS.items():
        print(f"h={h}: {name}...", flush=True)
        fc[name] = walk_forward(panel, make, OOS_START)
    print(f"h={h}: HAR and GARCH on {len(UNIVERSE)} assets...", flush=True)
    base = Parallel(n_jobs=-1)(delayed(baselines)(t, h) for t in UNIVERSE)
    base = pd.concat(dict(base), names=["ticker", "date"]).swaplevel()
    fc = fc.join(base).dropna()  # same rows for every model: out of sample, target known
    fc.to_pickle(path)
    return fc


def per_asset_ratios(fc, loss):
    models = [c for c in fc.columns if c != "y"]
    by = fc.groupby(level="ticker")
    raw = pd.DataFrame({m: by.apply(lambda d: loss(d.y, d[m])) for m in models})
    return raw.div(raw["HAR"], axis=0)


forecasts = {h: all_forecasts(h) for h in HORIZONS}
order = ["GARCH(1,1)", "Linear (all features)", "XGBoost", "MLP"]

rows = ["| Horizon | Model | MSE / HAR, mean (median) | beats HAR on MSE | QLIKE / HAR, mean (median) | beats HAR on QLIKE | S&P 500 QLIKE / HAR |",
        "|:--|:--|--:|--:|--:|--:|--:|"]
ratios = {}
for h, fc in forecasts.items():
    ratios[h] = {"MSE": per_asset_ratios(fc, mse), "QLIKE": per_asset_ratios(fc, qlike)}
    n = len(ratios[h]["MSE"])
    for m in order:
        cells = []
        for loss in ("MSE", "QLIKE"):
            s = ratios[h][loss][m]
            cells += [f"{s.mean():.3f} ({s.median():.3f})", f"{(s < 1).sum()}/{n}"]
        rows.append(f"| {h} days | {m} | {' | '.join(cells)} | {ratios[h]['QLIKE'].loc['^GSPC', m]:.3f} |")
dates = forecasts[5].index.get_level_values("date")
table = (f"Out of sample {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, {len(UNIVERSE)} assets, "
         f"{len(forecasts[5]):,} asset-days at 5 days. Each ratio is a model's loss divided by HAR's "
         f"on the same asset (below 1 = better than HAR), then averaged across assets.\n\n" + "\n".join(rows))

# XGBoost feature importance (one fit on all 5-day data, in-sample, only to see what it uses)
panel = build_panel(5).dropna(subset=["y"])
xgb = MODELS["XGBoost"]().fit(panel[FEATURES], np.log(panel.y))
importance = pd.Series(xgb.get_booster().get_score(importance_type="total_gain")).reindex(FEATURES).fillna(0)
importance = (importance / importance.sum()).sort_values()
table += "\n\nXGBoost feature importance (share of total gain, 5-day target):\n\n" + \
    "\n".join(f"- {k}: {v:.1%}" for k, v in importance[::-1].items())
(OUT / "phase3_table.md").write_text(table + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
for ax, h in zip(axes, HORIZONS):
    data = [ratios[h]["QLIKE"][m] for m in order]
    ax.boxplot(data, showmeans=True)
    for j, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(d)), d, s=8, alpha=0.5)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(1, len(order) + 1), [m.replace(" (all features)", "\n(all features)") for m in order], fontsize=8)
    ax.set(ylabel="QLIKE / HAR QLIKE", title=f"{h}-day horizon: one dot per asset (below 1 beats HAR)")
    ax.grid(alpha=0.3)
axes[2].barh(importance.index, importance.values)
axes[2].set(xlabel="share of total gain", title="What XGBoost uses (5-day target)")
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase3_ml.png", dpi=150)
print(table)
