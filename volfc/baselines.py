"""Baseline volatility forecasts: historical vol, EWMA, HAR and GARCH(1,1).

Every function returns a Series indexed by the forecast date t: the forecast of the annualized
realized variance over t+1..t+h, using only returns up to and including day t.
HAR and GARCH are re-estimated every `refit_every` days on an expanding window (walk-forward).
"""
import numpy as np
import pandas as pd
from arch import arch_model

from .data import TRADING_DAYS, future_realized_var, realized_var


def hist_forecast(r, window=22):
    """Tomorrow looks like the last `window` days: a random walk on variance."""
    return realized_var(r, window)


def ewma_forecast(r, lam=0.94):
    """RiskMetrics: sigma^2_{t+1} = lam * sigma^2_t + (1 - lam) * r_t^2 (flat across horizons)."""
    return TRADING_DAYS * (r**2).ewm(alpha=1 - lam, adjust=False).mean()


def har_features(r):
    """HAR regressors (Corsi 2009): realized variance over the last day, week and month."""
    return pd.DataFrame({"rv_d": realized_var(r, 1), "rv_w": realized_var(r, 5),
                         "rv_m": realized_var(r, 22)}).dropna()


def _refit_points(index, oos_start, refit_every):
    first = index.searchsorted(pd.Timestamp(oos_start))
    return range(first, len(index), refit_every)


def har_forecast(r, h, oos_start, refit_every=22):
    """HAR: RV_{t+1..t+h} = b0 + b_d RV_d + b_w RV_w + b_m RV_m, fitted by OLS."""
    X = har_features(r)
    y = future_realized_var(r, h).reindex(X.index)
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    out = pd.Series(np.nan, index=X.index)
    for i in _refit_points(X.index, oos_start, refit_every):
        # At the close of day i, a target y_j covers days j+1..j+h, so it is only known if j + h <= i.
        # Training on rows up to i would silently use the future: the classic look-ahead bug.
        train = slice(0, i - h + 1)
        beta = np.linalg.lstsq(A[train], y.iloc[train].to_numpy(), rcond=None)[0]
        block = slice(i, i + refit_every)
        out.iloc[block] = A[block] @ beta
    # ponytail: linear model can (rarely) go negative; floor it rather than switch to log-HAR
    return out.dropna().clip(lower=1e-6)


def garch_forecast(r, h, oos_start, refit_every=22):
    """GARCH(1,1) with Student-t errors: sigma^2_t = omega + alpha r_{t-1}^2 + beta sigma^2_{t-1}.

    The parameters are estimated on data up to the refit date; between refits they stay fixed while
    sigma^2_t keeps updating with each new return, exactly as it would in real time.
    """
    am = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")  # % scale helps the optimizer
    out = []
    for i in _refit_points(r.index, oos_start, refit_every):
        res = am.fit(last_obs=i + 1, disp="off")  # last_obs is exclusive: uses r_0..r_i
        block = r.index[i:i + refit_every]
        var = res.forecast(start=block[0], horizon=h, reindex=False).variance.loc[:block[-1]]
        out.append(var.mean(axis=1) * TRADING_DAYS / 100**2)  # average daily variance -> annualized
    return pd.concat(out)
