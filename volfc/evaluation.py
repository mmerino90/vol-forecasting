"""Loss functions for variance forecasts. y = realized variance (noisy proxy), f = forecast."""
import numpy as np


def mse(y, f):
    """Mean squared error. Dominated by a handful of crisis days."""
    return np.mean((np.asarray(y) - np.asarray(f)) ** 2)


def qlike(y, f):
    """QLIKE (Patton 2011): mean of y/f - log(y/f) - 1. Zero for a perfect forecast.

    Penalizes under-predicting volatility more than over-predicting it, and, like MSE, ranks
    forecasts correctly even though y is only a noisy estimate of the true variance.
    """
    ratio = np.asarray(y) / np.asarray(f)
    return np.mean(ratio - np.log(ratio) - 1)
