"""ML forecasts on the pooled panel of assets, with walk-forward re-training."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import FEATURES

# Hyperparameters fixed in advance, never tuned on the test years.
MODELS = {
    "Linear (all features)": lambda: LinearRegression(),
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=4, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=200, random_state=0)),
}


def walk_forward(panel, make_model, oos_start, refit_every=252):
    """Re-train once a year on every row whose target was fully known; forecast the next year.

    The model learns log(variance), which is far less skewed than variance itself. Then
    exp(prediction) is the median, not the mean, so it is scaled up by the average exp(residual)
    on the training set (Duan's smearing estimate).
    """
    dates = panel.index.get_level_values("date")
    test_dates = dates.unique().sort_values()
    test_dates = test_dates[test_dates >= pd.Timestamp(oos_start)]
    refits = list(test_dates[::refit_every]) + [pd.Timestamp.max]
    out = []
    for start, stop in zip(refits[:-1], refits[1:]):
        train = panel[(panel.y_end <= start)]  # NaT (target still in the future) compares False
        test = panel[(dates >= start) & (dates < stop)]
        model = make_model().fit(train[FEATURES], np.log(train.y))
        # ponytail: residuals are in-sample, so a flexible model's smearing factor is slightly low
        smear = np.mean(np.exp(np.log(train.y) - model.predict(train[FEATURES])))
        out.append(pd.Series(smear * np.exp(model.predict(test[FEATURES])), index=test.index))
    return pd.concat(out)
