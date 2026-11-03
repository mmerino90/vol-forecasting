"""Volatility targeting: scale the position so its risk stays near a constant target.

w_t = target vol / forecast vol, decided at the close of t, held over day t+1.
"""
import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def vol_target_weights(var_forecast, target=0.10, cap=2.0):
    """Position size from an annualized variance forecast; at most `cap` times leverage."""
    return (target / np.sqrt(var_forecast)).clip(upper=cap)


def strategy_returns(returns, weights, cost=0.0010):
    """Daily returns of the strategy and its daily turnover |w_t - w_{t-1}|.

    returns: simple daily returns, R_t = P_t / P_{t-1} - 1. weights: indexed by the day they are
    decided. Day t earns w_{t-1} * R_t and pays `cost` (fraction of the amount traded) on the trade
    made at the close of t. Starts from cash on the first weight's date. Cash earns nothing.
    """
    returns = returns.loc[weights.index[0]:]
    w = weights.reindex(returns.index).ffill()  # no new forecast that day: keep the position
    turnover = w.diff().abs().fillna(w.abs())
    return w.shift(1).fillna(0) * returns - cost * turnover, turnover


def sharpe(p):
    """Annualized Sharpe ratio (cash earns nothing here, so no risk-free rate to subtract)."""
    return p.mean() / p.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(p):
    """Largest fall from a previous peak of the wealth curve, as a negative fraction."""
    wealth = (1 + p).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def sharpe_diff_pvalue(a, b, block=22, reps=2000, seed=0):
    """Two-sided p-value for Sharpe(a) == Sharpe(b), circular block bootstrap of the paired days.

    Days are resampled in blocks of `block` days (keeping volatility clusters together), and the
    same days are drawn for a and b, so their strong correlation is kept.
    """
    x = np.column_stack([a, b])
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, -(-n // block)))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    s = x[idx]  # reps x n x 2
    sr = s.mean(1) / s.std(1, ddof=1)
    diff = sr[:, 0] - sr[:, 1]
    obs = sharpe(pd.Series(a)) - sharpe(pd.Series(b))
    return np.mean(np.abs(diff * np.sqrt(TRADING_DAYS) - obs) >= abs(obs))
