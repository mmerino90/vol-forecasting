"""Loss functions for variance forecasts and the Diebold–Mariano test.

y = realized variance (noisy proxy), f = forecast.
"""
import numpy as np
import statsmodels.api as sm


def squared_error(y, f):
    return (np.asarray(y) - np.asarray(f)) ** 2


def qlike_loss(y, f):
    """y/f - log(y/f) - 1 (Patton 2011): zero for a perfect forecast.

    Penalizes under-predicting volatility more than over-predicting it, and, like squared error,
    ranks forecasts correctly even though y is only a noisy estimate of the true variance.
    """
    ratio = np.asarray(y) / np.asarray(f)
    return ratio - np.log(ratio) - 1


def mse(y, f):
    """Mean squared error. Dominated by a handful of crisis days."""
    return np.mean(squared_error(y, f))


def qlike(y, f):
    return np.mean(qlike_loss(y, f))


def dm_test(loss_a, loss_b, lags):
    """Diebold–Mariano test of equal accuracy. Returns (t statistic, two-sided p-value).

    d_t = loss_a - loss_b. Under the null, E[d_t] = 0. Negative t means model a is better.
    d_t is autocorrelated (h-day targets overlap, and volatility clusters), so the standard error
    of its mean uses the Newey–West estimator with `lags` lags instead of the iid formula.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res.tvalues[0], res.pvalues[0]
