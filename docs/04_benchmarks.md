# Benchmarking & Empirical Validation

To certify the **PINN-BTC** solver for production-grade financial engineering, we adhere to a rigorous dual-phase validation protocol. This chapter details the quantitative performance of the model against both theoretical physics benchmarks (In-Silico) and live cryptocurrency market data (Empirical).

---

## 1. In-Silico Validation: Black-Scholes Consistency

The primary benchmark evaluates the model's ability to reconstruct the exact solution of the Black-Scholes PDE across the continuous 5-dimensional domain. This verifies that the neural network has correctly learned the underlying physics without overfitting to specific trajectories.

### 1.1 Pricing Surface Reconstruction
We evaluate the model's global approximation capability by comparing the predicted pricing surface $\hat{V}(S, \tau)$ against the analytical solution $V_{BS}(S, \tau)$.

* **Metric:** Mean Squared Error (MSE) and Relative Error across the domain grid.
* **Result:** The PINN generates a smooth, highly accurate pricing surface that aligns almost perfectly with the theoretical manifold, even in extrapolation regions (Deep OTM/ITM).

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/3d_surface_comparison.png" alt="3D Surface Comparison" width=65%" />
</p>

*Figure 1*: 3D Surface Reconstruction (PINN Prediction vs. Analytical Solution). The visualization confirms that the model captures the convex curvature of the option price with high fidelity.

### 1.2 The "Kink" Resolution (Singularity Handling)
A critical stress test for any option pricing solver is the non-differentiable payoff at maturity ($\tau=0$). Standard deep learning models often produce "smoothed" approximations near the strike price ($S=K$), leading to arbitrage opportunities.

Thanks to our **Weighted Kink Loss** strategy, the PINN-BTC solver exhibits sharp convergence at the singularity:

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/payoff_at_maturity_kink.png" alt="Payoff at Maturity" width=65%" />
</p>

*Figure 2*: Payoff function at maturity ($\tau=0$). The model strictly adheres to the $C^0$ continuity while preserving the sharp hinge at the strike price ($K$), validating the efficacy of the Hard Attention mechanism.

### 1.3 Statistical Alignment
Cross-sectional regression analysis between the predicted values and the ground truth demonstrates a near-perfect linear correlation (Corr > 0.99), indicating unbiased estimation across diverse volatility regimes ($\sigma \in [0.1, 2.0]$).

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/scatter_comparison.png" alt="Scatter Comparison" width=65%" />
</p>

*Figure 3*: Scatter plot of Model Predictions vs. Analytical Solutions showing a tight 1:1 alignment.

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

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/result_BTC-251226-95000-C_Quarterly_2h_sigma7day_134200.jpg" alt="Real Market Validation 1" width=65%" />
</p>

*Figure 4*: Validation against BTC-251226-95000-C. The PINN prediction (Blue) tightly follows the Market Price (Green) and the Analytical Benchmark (Red), confirming its applicability for real-time market marking.

**Case B: Monthly Call Option (Strike 100k - Deep OTM)**

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/result_BTC-251128-100000-C_Monthly_1h_sigma7day_134139.jpg" alt="Real Market Validation 2" width=65%" />
</p>

*Figure 5*: Validation against BTC-251128-100000-C. Even for Out-of-the-Money options where liquidity is lower, the model provides stable pricing consistent with implied volatility trends.

---

## 3. Performance Metrics Definition

To ensure a rigorous evaluation of the **PINN-BTC** solver, covering both mathematical generalization across the high-dimensional domain and empirical applicability in live financial markets, we classify our performance metrics into two distinct categories based on the evaluation context:

### 3.1. Generalization Metrics (Dimensionless Ratio Evaluation)

During the training and in-training validation phases, the model is evaluated against a synthetic dataset sampled from a 5-dimensional hypercube using the **Mixed Distribution Strategy**. Given the vast dynamic range of Strike Prices ($K \in [10k, 500k]$ USDT), direct error measurement in currency units introduces significant scale bias.

To mitigate this, we evaluate performance using the **Dimensionless Option Price Ratio** ($R = V/K$). This approach assesses the neural network's ability to generalize the pricing law independent of asset magnitude.

Let $R_{true}$ denote the ground truth (Analytical Solution) and $R_{pred}$ denote the model prediction. We define the following metrics:

1.  **Symmetric Mean Absolute Percentage Error (SMAPE)**
    The primary metric for assessing relative accuracy across varying price scales.

```math
    \text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|\mathcal{R}_{pred}^{(i)} - \mathcal{R}_{true}^{(i)}|}{(|\mathcal{R}_{true}^{(i)}| + |\mathcal{R}_{pred}^{(i)}|)/2 + \epsilon}
```

2.  **Root Mean Squared Error (RMSE)**
    Measures the standard deviation of prediction errors, heavily penalizing large deviations.

```math
    \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\mathcal{R}_{pred}^{(i)} - \mathcal{R}_{true}^{(i)})^2}
```

3.  **Mean Absolute Error (MAE)**
    Represents the average magnitude of errors, providing a linear score of accuracy.

```math
    \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\mathcal{R}_{pred}^{(i)} - \mathcal{R}_{true}^{(i)}|
```

4.  **Kink Mean Absolute Error (KinkMAE)**
    A specialized metric designed to evaluate **Hard Attention** performance at the singularity point where $S=K$ and $\tau=0$ (At-The-Money at Expiration). Theoretical arbitrage-free conditions dictate a value of exactly 0.

```math
    \text{KinkMAE} = \frac{1}{N_{kink}} \sum_{j \in \Omega_{kink}} |\mathcal{R}_{pred}^{(j)} - 0|
```

1.  **Pearson Correlation Coefficient**
    Evaluates the linear correlation between predicted and theoretical values to confirm trend alignment.

```math
    \text{Corr} = \frac{\sum (\mathcal{R}_{pred} - \bar{\mathcal{R}}_{pred})(\mathcal{R}_{true} - \bar{\mathcal{R}}_{true})}{\sqrt{\sum (\mathcal{R}_{pred} - \bar{\mathcal{R}}_{pred})^2} \sqrt{\sum (\mathcal{R}_{true} - \bar{\mathcal{R}}_{true})^2}}
```

6.  **Mean Bias**
    Indicates systematic error, revealing whether the model tends to **overestimate** (positive bias) or **underestimate** (negative bias) the option premiums.

```math
    \text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (\mathcal{R}_{pred}^{(i)} - \mathcal{R}_{true}^{(i)})
```

7.  **Max Error**
    The worst-case prediction error observed within the batch, used for risk boundary assessment.

```math
    \text{MaxError} = \max_{i} |\mathcal{R}_{pred}^{(i)} - \mathcal{R}_{true}^{(i)}|
```

---

### 3.2. Empirical Metrics (Real Market Valuation)

For real-world inference and historical backtesting against market data (e.g., Binance Option Kline), the input parameters ($S, K, r, \sigma, \tau$) are fixed to realized market conditions extracted from historical timestamps.

To avoid ambiguity with the time-to-maturity variable ($\tau$), we denote $N$ as the total number of observed data points (historical candles) and use the index $i$ to represent the $i$-th observation in the time series.

Let $V_{mkt}^{(i)}$ be the observed market price and $V_{model}^{(i)}$ be the PINN-predicted price for the $i$-th sample.

#### 1. Root Mean Squared Error (RMSE)
Measures the standard deviation of the residuals in USDT across the evaluation period.

* **Model vs. Market:**

```math
  \text{RMSE}_{mkt} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (V_{model}^{(i)} - V_{mkt}^{(i)})^2}
```

* **Model vs. Analytical:**

```math
  \text{RMSE}_{BS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (V_{model}^{(i)} - V_{BS}^{(i)})^2}
```

#### 2. Pearson Correlation Coefficient (Corr)
Evaluates the linear relationship and trend alignment between the predicted prices and the baselines.

* **Model vs. Market:**
  Indicates how well the model tracks market dynamics.

```math
  \text{Corr}_{mkt} = \frac{\sum_{i=1}^{N} (V_{model}^{(i)} - \bar{V}_{model})(V_{mkt}^{(i)} - \bar{V}_{mkt})}{\sqrt{\sum_{i=1}^{N} (V_{model}^{(i)} - \bar{V}_{model})^2} \sqrt{\sum_{i=1}^{N} (V_{mkt}^{(i)} - \bar{V}_{mkt})^2}}
```

* **Model vs. Analytical:**
  Indicates the structural fidelity of the model to the governing physics.

```math
  \text{Corr}_{BS} = \frac{\sum_{i=1}^{N} (V_{model}^{(i)} - \bar{V}_{model})(V_{BS}^{(i)} - \bar{V}_{BS})}{\sqrt{\sum_{i=1}^{N} (V_{model}^{(i)} - \bar{V}_{model})^2} \sqrt{\sum_{i=1}^{N} (V_{BS}^{(i)} - \bar{V}_{BS})^2}}
```

#### 2. Pearson Correlation Coefficient (Corr)
We denote this metric as **Corr** (distinct from $r$, the risk-free rate) to evaluate the linear relationship and trend alignment.

* **Model vs. Market:**
    Indicates how well the model tracks market dynamics.

```math
    \text{Corr}_{mkt} = \frac{\sum (V_{model} - \bar{V}_{model})(V_{mkt} - \bar{V}_{mkt})}{\sqrt{\sum (V_{model} - \bar{V}_{model})^2} \sqrt{\sum (V_{mkt} - \bar{V}_{mkt})^2}}
```

* **Model vs. Analytical:**
    Indicates the structural fidelity of the model to the governing physics.

```math
    \text{Corr}_{BS} = \frac{\sum (V_{model} - \bar{V}_{model})(V_{BS} - \bar{V}_{BS})}{\sqrt{\sum (V_{model} - \bar{V}_{model})^2} \sqrt{\sum (V_{BS} - \bar{V}_{BS})^2}}
```
---

### 3.3. Physics-Informed Loss Components

Beyond evaluation metrics, we monitor the component-wise breakdown of the loss function during training to ensure physical consistency with the Black-Scholes-Merton framework:

* **PDE Loss ($\mathcal{L}_{PDE}$):** Measures the residual of the partial differential equation (Physics Violation).
* **IVP Loss ($\mathcal{L}_{IVP}$):** Enforces the payoff condition at maturity ($\tau=0$).
* **BVP Loss ($\mathcal{L}_{BVP}$):** Enforces asymptotic boundary conditions at $S \to 0$ and $S \to S_{max}$.
* **Kink Loss ($\mathcal{L}_{Kink}$):** A weighted loss component focusing on the high-convexity region at the strike price.