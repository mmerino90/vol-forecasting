# What I learned forecasting volatility

I built this project to prepare for *Numerical Methods for Finance*, *Financial Engineering* and *Applied Computational Finance*, after a [Monte Carlo option pricer](https://github.com/mmerino90/mc-option-pricing) where volatility was simply an input, σ = 20%. Here it is the thing to predict. These are the ideas that stuck, each tied to the experiment that taught it to me.

*(Phases 1–2 so far; this file grows with the project.)*

---

## 1. You can't predict returns, but you can predict their size

The autocorrelation of S&P 500 returns is close to zero, while that of *absolute* returns is **0.31** at lag 1 and stays positive for about 80 trading days. Calm periods are followed by calm, storms by storms. This is **volatility clustering**, and every model in this project is a different way to exploit it.

## 2. Volatility is never observed

There is no "true" volatility in the data, only returns. I measure it after the fact with realized variance, $\mathrm{RV} = \frac{252}{h}\sum r^2$, which is itself noisy: over 5 days it is an average of only 5 squared returns. So every forecast is scored against a noisy target, and that changes which loss functions are fair (see 4).

## 3. Look-ahead bias hides in one index

The HAR target on day $t$ is the variance of days $t+1,\dots,t+h$. When I refit the model on day $t$, the last $h$ targets aren't known yet. Training on them would quietly use the future. The fix is one line (`slice(0, i - h + 1)` instead of `slice(0, i + 1)`). I wrote a test that scrambles all returns after a cutoff and checks that no earlier forecast moves, and confirmed the test fails when I plant the bug.

## 4. "Best model" depends on the loss

HAR beats GARCH on MSE; GARCH beats HAR on QLIKE. MSE is dominated by a handful of crisis days, where errors are huge in absolute terms. QLIKE, $\frac{y}{f} - \log\frac{y}{f} - 1$, looks at *ratios* and punishes under-predicting volatility more than over-predicting it. Each model wins on the loss closest to how it was fitted: HAR by least squares, GARCH by maximum likelihood. Both losses still rank forecasts correctly with a noisy target, which is why they are the standard pair.

## 5. Mean reversion matters more than reacting fast

Historical vol and EWMA only look backwards: after a spike they stay high until the spike drops out of the window. GARCH and HAR also pull the forecast back toward a long-run level (fitted α + β = 0.988, a shock half-life of about 58 days), and that alone cuts MSE by 17–39%.
