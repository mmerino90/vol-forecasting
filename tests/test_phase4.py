import numpy as np

from volfc import dm_test, qlike_loss


def test_dm_detects_a_real_difference():
    rng = np.random.default_rng(0)
    a, b = rng.exponential(1.0, 2000), rng.exponential(1.0, 2000)
    t, p = dm_test(a, b + 0.3, lags=5)  # b is worse by 0.3 per day
    assert t < 0 and p < 0.001


def test_dm_rejects_about_5_percent_when_models_are_equal_even_with_overlap():
    # Overlapping 5-day targets make d_t a moving average of 5 shocks: strongly autocorrelated.
    # Newey-West keeps the false-alarm rate near 5%; the naive iid formula (0 lags) does not.
    # With 3,000 days (our sample) it measures ~7%; with only 500 days it would be ~11%.
    rng = np.random.default_rng(1)
    reject_hac, reject_naive = 0, 0
    sims = 300
    for _ in range(sims):
        d = np.convolve(rng.standard_normal(3004), np.ones(5), mode="valid")  # 3,000 days, mean 0
        reject_hac += dm_test(d, np.zeros_like(d), lags=10)[1] < 0.05
        reject_naive += dm_test(d, np.zeros_like(d), lags=0)[1] < 0.05
    assert reject_hac / sims < 0.10
    assert reject_naive / sims > 0.30


def test_qlike_loss_elementwise():
    np.testing.assert_allclose(qlike_loss([1.0, 2.0], [1.0, 1.0]), [0.0, 2.0 - np.log(2.0) - 1.0])
