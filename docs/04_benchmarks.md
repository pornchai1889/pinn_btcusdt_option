# Benchmarking & Empirical Validation

To certify the **PINN-BTC** solver for production-grade financial engineering, we adhere to a rigorous dual-phase validation protocol. This chapter details the quantitative performance of the model against both theoretical physics benchmarks (In-Silico) and live cryptocurrency market data (Empirical).

---

## 1. In-Silico Validation: Black-Scholes Consistency

The primary benchmark evaluates the model's ability to reconstruct the exact solution of the Black-Scholes PDE across the continuous 5-dimensional domain. This verifies that the neural network has correctly learned the underlying physics without overfitting to specific trajectories.

### 1.1 Pricing Surface Reconstruction
We evaluate the model's global approximation capability by comparing the predicted pricing surface $\hat{V}(S, \tau)$ against the analytical solution $V_{BS}(S, \tau)$.

* **Metric:** Mean Squared Error (MSE) and Relative Error across the domain grid.
* **Result:** The PINN generates a smooth, highly accurate pricing surface that aligns almost perfectly with the theoretical manifold, even in extrapolation regions (Deep OTM/ITM).

![3D Surface Comparison](../models/call/train_2025-12-21_10-22-39_call/3d_surface_comparison.png)
*Figure 1: 3D Surface Reconstruction (PINN Prediction vs. Analytical Solution). The visualization confirms that the model captures the convex curvature of the option price with high fidelity.*

### 1.2 The "Kink" Resolution (Singularity Handling)
A critical stress test for any option pricing solver is the non-differentiable payoff at maturity ($\tau=0$). Standard deep learning models often produce "smoothed" approximations near the strike price ($S=K$), leading to arbitrage opportunities.

Thanks to our **Weighted Kink Loss** strategy, the PINN-BTC solver exhibits sharp convergence at the singularity:

![Payoff at Maturity](../models/call/train_2025-12-21_10-22-39_call/payoff_at_maturity_kink.png)
*Figure 2: Payoff function at maturity ($\tau=0$). The model strictly adheres to the $C^0$ continuity while preserving the sharp hinge at the strike price ($K$), validating the efficacy of the Hard Attention mechanism.*

### 1.3 Statistical Alignment
Cross-sectional regression analysis between the predicted values and the ground truth demonstrates a near-perfect linear correlation ($R^2 > 0.99$), indicating unbiased estimation across diverse volatility regimes ($\sigma \in [0.1, 2.0]$).

![Scatter Comparison](../models/call/train_2025-12-21_10-22-39_call/scatter_comparison.png)
*Figure 3: Scatter plot of Model Predictions vs. Analytical Solutions showing a tight 1:1 alignment.*

---

## 2. Empirical Validation: Real-World Binance Markets

Beyond theoretical correctness, we validate the model's generalization to noisy, real-world data using **Bitcoin (BTC/USDT)** options traded on Binance.

### 2.1 Methodology
* **Data Source:** Binance Options API (Klines/Candlestick data).
* **Volatility Estimation:** We utilize a trailing **Historical Volatility** window (e.g., 7-day standard deviation of log-returns) to estimate the unobservable $\sigma$ parameter for the PINN input.
* **Benchmark:** The PINN's output is compared against the actual market close prices of specific option contracts.

### 2.2 Case Study: Volatile Market Regimes
We tested the model on multiple expiration cycles (Monthly/Quarterly) during high-volatility periods. The results show that the Physics-Informed model tracks market dynamics robustly, often filtering out microstructure noise.

**Case A: Quarterly Call Option (Strike 95k)**
![Real Market Validation 1](../models/call/train_2025-12-21_10-22-39_call/result_BTC-251226-95000-C_Quarterly_2h_sigma7day_134200.jpg)
*Figure 4: Validation against BTC-251226-95000-C. The PINN prediction (Blue) tightly follows the Market Price (Green) and the Analytical Benchmark (Red), confirming its applicability for real-time market marking.*

**Case B: Monthly Call Option (Strike 100k - Deep OTM)**
![Real Market Validation 2](../models/call/train_2025-12-21_10-22-39_call/result_BTC-251128-100000-C_Monthly_1h_sigma7day_134139.jpg)
*Figure 5: Validation against BTC-251128-100000-C. Even for Out-of-the-Money options where liquidity is lower, the model provides stable pricing consistent with implied volatility trends.*

---

## 3. Performance Metrics Definition

To ensure reproducibility, we define the key metrics used in our evaluation pipelines:

1.  **Physics Loss ($\mathcal{L}_{PDE}$):** Measures the violation of the Black-Scholes equation.
    $$\mathcal{L}_{PDE} = \frac{1}{N} \sum ||\mathcal{N}[\hat{V}]||^2$$
2.  **Kink Error (MAE at Strike):** Specifically measures precision at the point of maximum convexity.
    $$\text{KinkMAE} = \frac{1}{N} \sum_{S=K} |\hat{V} - \text{Payoff}|$$
3.  **Market RMSE:** Root Mean Square Error between predicted prices and traded market prices.
    $$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (\hat{V}_t - V_{market, t})^2}$$

These benchmarks collectively confirm that **PINN-BTC** is not just a theoretical artifact but a robust numerical solver capable of operating in production financial environments.