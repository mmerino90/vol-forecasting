"""Phase 5: does a better forecast make money? Volatility targeting with transaction costs.

Each day, at the close, every model sets the position in each of the 27 assets to
w = 10% / forecast vol (at most 2x), and pays 10 bp on what it trades. Same out-of-sample
forecasts as phases 3-4 (run phase 3 first). Compared with buy-and-hold, with a constant
position sized to 10% vol on 2010-14 data (same average risk, no timing), and with an
"oracle" that knows the future realized variance (impossible in practice: an upper bound).

Produces:
  results/phase5_strategy.png  (S&P 500 wealth and rolling vol, portfolio Sharpe vs cost)
  results/phase5_table.md
Run from the repo root:  python -m experiments.phase5_strategy
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volfc import (TRADING_DAYS, load_prices, max_drawdown, sharpe, sharpe_diff_pvalue,
                   strategy_returns, vol_target_weights)

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("results")
HORIZONS = (5, 22)
TARGET, CAP, COST = 0.10, 2.0, 0.0010
COSTS_BP = (0, 5, 10, 20, 40)
MODELS = ["HAR", "GARCH(1,1)", "Linear (all features)", "XGBoost", "MLP"]
STRATEGIES = ["Buy & hold", "Constant 10% (2010-14 vol)"] + MODELS + ["Oracle (knows the future)"]


def weights_for(fc, name, R):
    if name == "Buy & hold":
        return pd.Series(1.0, index=fc.index)
    if name.startswith("Constant"):  # sized once, before the test period starts
        return pd.Series(min(TARGET / (R[:"2014"].std() * np.sqrt(TRADING_DAYS)), CAP), index=fc.index)
    var = fc.y if name.startswith("Oracle") else fc[name]
    return vol_target_weights(var, TARGET, CAP)


def stats(p, turnover):
    roll = p.rolling(63).std().dropna() * np.sqrt(TRADING_DAYS)
    return {"Return": p.mean() * TRADING_DAYS, "Vol": p.std() * np.sqrt(TRADING_DAYS), "Sharpe": sharpe(p),
            "Max DD": max_drawdown(p), "Turnover": turnover.mean() * TRADING_DAYS,
            "Vol of vol": roll.std() / roll.mean()}  # how stable the 3-month risk is, scale-free


def run(fc, cost):
    """Daily returns and turnover per (strategy, ticker)."""
    rets, turns = {}, {}
    for ticker, f in fc.groupby(level="ticker"):
        f = f.droplevel("ticker")
        R = load_prices(ticker)["Close"].pct_change()
        for s in STRATEGIES:
            rets[s, ticker], turns[s, ticker] = strategy_returns(R, weights_for(f, s, R), cost)
    return pd.DataFrame(rets).dropna(), pd.DataFrame(turns).dropna()


def fmt(row):
    return (f"{row.Return:+.1%} | {row.Vol:.1%} | {row.Sharpe:.2f} | {row['Max DD']:.0%} | "
            f"{row.Turnover:.1f}x | {row['Vol of vol']:.2f}")


lines, saved = [], {}
header = "| Strategy | Return / yr | Vol | Sharpe | Max drawdown | Turnover / yr | Vol of vol |"
for h in HORIZONS:
    fc = pd.read_pickle(Path("data") / f"forecasts_h{h}.pkl")
    rets, turns = run(fc, COST)
    port = rets.T.groupby(level=0).mean().T  # equal weight across the 27 assets, per strategy
    port_turn = turns.T.groupby(level=0).mean().T
    per_asset = pd.DataFrame({k: stats(rets[k], turns[k]) for k in rets}).T
    per_asset.index.names = ["strategy", "ticker"]
    avg = per_asset.groupby(level="strategy").mean()
    saved[h] = rets, port
    sweep = {}
    for bp in COSTS_BP:
        r, _ = run(fc, bp / 1e4) if bp != COST * 1e4 else (rets, None)
        sweep[bp] = r.T.groupby(level=0).mean().T.apply(sharpe)
    saved[h, "sweep"] = pd.DataFrame(sweep).T

    sr = per_asset.Sharpe.unstack("strategy")
    lines += [f"## {h}-day forecasts, rebalanced daily, {COST * 1e4:.0f} bp per trade "
              f"({port.index[0]:%Y-%m-%d} to {port.index[-1]:%Y-%m-%d})", "",
              "### Average over the 27 assets (each traded on its own)", "",
              header + " Sharpe > HAR |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for s in STRATEGIES:
        beats = "" if s == "HAR" else f"{(sr[s] > sr['HAR']).sum()}/27"
        lines.append(f"| {s} | {fmt(avg.loc[s])} | {beats} |")
    lines += ["", "### Equal-weight portfolio of the 27 assets", "",
              header + " p-value Sharpe vs HAR |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for s in STRATEGIES:
        p = "" if s == "HAR" else f"{sharpe_diff_pvalue(port[s], port['HAR']):.3f}"
        lines.append(f"| {s} | {fmt(pd.Series(stats(port[s], port_turn[s])))} | {p} |")
    lines += ["", "### Portfolio Sharpe ratio by cost (bp per unit traded)", "",
              "| Cost | " + " | ".join(STRATEGIES) + " |", "|--:|" + "--:|" * len(STRATEGIES)]
    lines += [f"| {bp} bp | " + " | ".join(f"{v:.2f}" for v in row[STRATEGIES]) + " |"
              for bp, row in saved[h, "sweep"].iterrows()]
    lines.append("")
lines += ["Vol of vol = std / mean of the 3-month rolling volatility: lower means steadier risk.",
          "p-value: circular block bootstrap (22-day blocks) of the difference in Sharpe ratios."]

table = "\n".join(lines)
(OUT / "phase5_table.md").write_text(table + "\n", encoding="utf-8")

# Plot: S&P 500 wealth and rolling vol (5-day forecasts), then portfolio Sharpe vs cost
rets, _ = saved[5]
show = ["Buy & hold", "Constant 10% (2010-14 vol)", "HAR", "GARCH(1,1)", "Linear (all features)", "XGBoost"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for s in show:
    p = rets[s, "^GSPC"]
    axes[0].plot((1 + p).cumprod(), lw=1, label=s)
    axes[1].plot(p.rolling(63).std() * np.sqrt(TRADING_DAYS), lw=1, label=s)
axes[0].set(yscale="log", title="S&P 500: growth of $1 (5-day forecasts, 10 bp costs)")
axes[1].axhline(TARGET, color="k", ls="--", lw=0.8)
axes[1].set(title="S&P 500: 3-month rolling volatility (target 10%)", ylim=(0, 0.6))
for s in STRATEGIES:
    axes[2].plot(COSTS_BP, saved[5, "sweep"][s], marker="o", label=s)
axes[2].set(xlabel="cost per unit traded (bp)", ylabel="Sharpe ratio",
            title="27-asset portfolio (5-day forecasts): Sharpe vs costs")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "phase5_strategy.png", dpi=150)
print(table)
