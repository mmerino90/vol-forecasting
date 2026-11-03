import numpy as np
import pandas as pd

from volfc import max_drawdown, sharpe, sharpe_diff_pvalue, strategy_returns, vol_target_weights

DAYS = pd.bdate_range("2020-01-01", periods=5)


def test_weight_decided_today_earns_tomorrows_return():
    R = pd.Series([0.01, 0.02, -0.03, 0.04, 0.05], index=DAYS)
    w = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=DAYS)  # buy at the close of day 3
    p, _ = strategy_returns(R, w, cost=0.0)
    np.testing.assert_allclose(p, [0, 0, 0, 0.04, 0.05])  # day 3's -3% is not earned


def test_costs_are_charged_on_turnover():
    R = pd.Series(0.0, index=DAYS)
    w = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0], index=DAYS)
    p, turnover = strategy_returns(R, w, cost=0.001)
    np.testing.assert_allclose(turnover, [1, 0, 1, 1, 0])  # first buy, sell, buy back
    np.testing.assert_allclose(p, -0.001 * turnover)


def test_right_forecast_hits_the_target_vol():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2000-01-01", periods=20_000)
    R = pd.Series(rng.standard_normal(len(days)) * 0.30 / np.sqrt(252), index=days)  # 30% vol
    w = vol_target_weights(pd.Series(0.30**2, index=days), target=0.10)
    p, _ = strategy_returns(R, w, cost=0.0)
    assert abs(p.std() * np.sqrt(252) - 0.10) < 0.002
    assert vol_target_weights(pd.Series([0.0001]), cap=2.0).iloc[0] == 2.0  # 1% vol: capped


def test_max_drawdown():
    p = pd.Series([0.10, -0.50, 0.20])  # wealth 1.1, 0.55, 0.66
    assert np.isclose(max_drawdown(p), -0.50)


def test_sharpe_diff_pvalue():
    rng = np.random.default_rng(1)
    common = rng.standard_normal(3000) * 0.01
    a = common + rng.standard_normal(3000) * 0.002
    assert sharpe_diff_pvalue(a, a.copy()) == 1.0  # identical strategies
    assert sharpe_diff_pvalue(a + 0.001, a) < 0.01  # same risk, 25% more return a year
    assert sharpe(pd.Series(a + 0.001)) > sharpe(pd.Series(a))
