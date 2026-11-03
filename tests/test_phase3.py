import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from volfc import FEATURES, build_features, walk_forward

N = 1000
DATES = pd.bdate_range("2010-01-01", periods=N)


def fake_prices(seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(N)))
    return pd.DataFrame({"Close": close, "Volume": rng.uniform(1e6, 2e6, N)}, index=DATES)


def fake_vix(seed=0):
    return pd.Series(20 * np.exp(0.1 * np.random.default_rng(seed).standard_normal(N)), index=DATES)


def test_features_only_use_the_past():
    cutoff = DATES[600]
    p1, p2 = fake_prices(), fake_prices()
    v1, v2 = fake_vix(), fake_vix()
    p2.loc[p2.index > cutoff] = fake_prices(seed=1).loc[p2.index > cutoff]
    v2.loc[v2.index > cutoff] = fake_vix(seed=1).loc[v2.index > cutoff]
    f1 = build_features(p1, v1, 5).loc[:cutoff, FEATURES]
    f2 = build_features(p2, v2, 5).loc[:cutoff, FEATURES]
    pd.testing.assert_frame_equal(f1, f2)


def test_target_end_date_is_h_days_ahead():
    f = build_features(fake_prices(), fake_vix(), 5)
    i = f.index.get_loc(DATES[500])
    assert f.y_end.iloc[i] == f.index[i + 5]
    assert f.y_end.iloc[-5:].isna().all()


def fake_panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([DATES, ["A", "B"]], names=["date", "ticker"])
    panel = pd.DataFrame(rng.standard_normal((len(idx), len(FEATURES))), index=idx, columns=FEATURES)
    panel["y"] = np.exp(0.5 * panel[FEATURES[0]] + rng.standard_normal(len(idx)))
    panel["y_end"] = np.repeat(pd.Series(DATES).shift(-5).to_numpy(), 2)
    return panel


def test_walk_forward_ignores_targets_not_yet_known():
    p1, p2 = fake_panel(), fake_panel()
    start = DATES[500]
    p2.loc[p2.y_end > start, "y"] *= 100  # change every target that ends after the first refit
    f1 = walk_forward(p1, LinearRegression, oos_start=start, refit_every=100)
    f2 = walk_forward(p2, LinearRegression, oos_start=start, refit_every=100)
    first_block = f1.index.get_level_values("date") < DATES[600]
    np.testing.assert_allclose(f1[first_block], f2[first_block])


def test_smearing_targets_the_mean_not_the_median():
    # log y ~ N(mu, 1): the median is e^mu but the mean is e^(mu + 1/2), 65% higher
    p = fake_panel()
    p["y"] = np.exp(np.random.default_rng(1).standard_normal(len(p)))
    f = walk_forward(p, LinearRegression, oos_start=DATES[500], refit_every=100)
    assert f.mean() == pytest.approx(np.exp(0.5), rel=0.05)
