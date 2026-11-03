import numpy as np
import pandas as pd
import pytest

from volfc import (ewma_forecast, future_realized_var, garch_forecast, har_forecast, qlike,
                   realized_var)


def fake_returns(n=1500, seed=0):
    """GARCH-like returns with volatility clustering, on business days."""
    rng = np.random.default_rng(seed)
    r, s2 = np.empty(n), 1e-4
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 2e-6 + 0.1 * r[t] ** 2 + 0.88 * s2
    return pd.Series(r, index=pd.bdate_range("2010-01-01", periods=n))


def test_realized_var_formula():
    r = pd.Series([0.01, -0.02, 0.03])
    assert realized_var(r, 3).iloc[-1] == pytest.approx(252 * (1 + 4 + 9) * 1e-4 / 3)


def test_target_covers_the_next_h_days():
    r = fake_returns(50)
    y = future_realized_var(r, 5)
    assert y.iloc[10] == pytest.approx(252 * (r.iloc[11:16] ** 2).mean())
    assert y.iloc[-5:].isna().all()  # the future isn't known yet


def test_ewma_recursion():
    r = fake_returns(100)
    f = ewma_forecast(r, lam=0.9) / 252
    assert f.iloc[50] == pytest.approx(0.9 * f.iloc[49] + 0.1 * r.iloc[50] ** 2)


def test_qlike_zero_when_perfect_and_punishes_underprediction_more():
    assert qlike([0.04], [0.04]) == pytest.approx(0)
    assert qlike([0.04], [0.02]) > qlike([0.04], [0.08])  # half the variance vs double it


@pytest.mark.parametrize("model", [har_forecast, garch_forecast])
def test_no_look_ahead(model):
    # Scrambling every return after a cutoff must not change any forecast made up to the cutoff
    r = fake_returns()
    cutoff = r.index[1200]
    r2 = r.copy()
    r2[r2.index > cutoff] = fake_returns(seed=1)[r2.index > cutoff] * 5
    f1 = model(r, 5, oos_start=r.index[1000]).loc[:cutoff]
    f2 = model(r2, 5, oos_start=r.index[1000]).loc[:cutoff]
    assert len(f1) > 150
    np.testing.assert_allclose(f1, f2, rtol=1e-6)
