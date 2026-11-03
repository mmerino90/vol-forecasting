"""Daily prices from Yahoo Finance, cached as CSV, and realized volatility."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2010-01-01", "2026-10-31"  # fixed END so every run sees the same data
CACHE = Path("data")
TRADING_DAYS = 252


def load_prices(ticker, start=START, end=END):
    """OHLCV for one ticker (prices adjusted for splits and dividends).

    Downloaded once and cached in data/<ticker>.csv, so later runs are offline and identical.
    """
    path = CACHE / f"{ticker.replace('^', '')}.csv"
    if not path.exists():
        import yfinance as yf
        with warnings.catch_warnings():  # yfinance trips pandas deprecation warnings
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False,
                             multi_level_index=False)
        if df.empty:
            raise RuntimeError(f"no data downloaded for {ticker}")
        CACHE.mkdir(exist_ok=True)
        df.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def log_returns(close):
    """r_t = log(P_t / P_{t-1})."""
    return np.log(close).diff().dropna()


def realized_var(r, window):
    """Annualized realized variance over the *past* `window` days, ending at t (inclusive).

    RV_t = 252/window * sum_{i=0}^{window-1} r_{t-i}^2. The mean return is ignored:
    over a few days it is tiny compared with the squared moves.
    """
    return TRADING_DAYS * (r**2).rolling(window).mean()


def future_realized_var(r, h):
    """Forecast target: annualized realized variance over the *next* h days, t+1..t+h.

    Indexed at t, the day the forecast is made. The last h rows are NaN (the future isn't known yet).
    """
    return realized_var(r, h).shift(-h)
