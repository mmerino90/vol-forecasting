"""Feature panel for the ML models: one row per (date, ticker), everything known at the close of t."""
import numpy as np
import pandas as pd

from .data import future_realized_var, load_prices, log_returns, realized_var

# S&P 500 plus 26 liquid ETFs and large stocks, all trading since 2010.
# ponytail: today's survivors (survivorship bias); fine for forecasting vol, not for picking stocks.
UNIVERSE = ["^GSPC", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLF", "XLK", "XLV", "XLU",
            "AAPL", "MSFT", "AMZN", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "PG", "KO", "WMT",
            "NVDA", "INTC", "CSCO"]
FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q", "r_d", "r_w", "r_m",
            "log_volume_ratio", "log_vix", "vix_change_w"]
EPS = 1e-5  # keeps log() finite on days with a zero return


def build_features(prices, vix, h):
    """Features, target y (future realized variance over t+1..t+h) and y_end (date of day t+h)."""
    r = log_returns(prices["Close"])
    vix = vix.reindex(r.index).ffill()
    volume = prices["Volume"].replace(0, np.nan)  # a few days report zero volume: a data error
    f = pd.DataFrame({
        # past volatility at four speeds: day, week, month, quarter (HAR's idea, in logs)
        "log_rv_d": np.log(realized_var(r, 1) + EPS),
        "log_rv_w": np.log(realized_var(r, 5) + EPS),
        "log_rv_m": np.log(realized_var(r, 22) + EPS),
        "log_rv_q": np.log(realized_var(r, 66) + EPS),
        # past returns: falling prices tend to raise volatility (the leverage effect)
        "r_d": r, "r_w": r.rolling(5).sum(), "r_m": r.rolling(22).sum(),
        # unusual trading activity, relative to the past month
        "log_volume_ratio": np.log(volume / volume.rolling(22, min_periods=15).mean()).reindex(r.index),
        # the market's own 30-day forecast, and how fast it is moving
        "log_vix": np.log(vix), "vix_change_w": np.log(vix / vix.shift(5)),
    })
    f["y"] = future_realized_var(r, h)
    f["y_end"] = pd.Series(r.index, index=r.index).shift(-h)  # the target is only known after this date
    return f.dropna(subset=FEATURES)


def build_panel(h, tickers=UNIVERSE):
    vix = load_prices("^VIX")["Close"]
    parts = {t: build_features(load_prices(t), vix, h) for t in tickers}
    return pd.concat(parts, names=["ticker", "date"]).swaplevel().sort_index()
