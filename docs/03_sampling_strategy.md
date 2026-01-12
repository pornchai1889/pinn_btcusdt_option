# Massive-Scale Data Synthesis: The Mixed-Distribution Strategy

## 1. Paradigm Shift: From Finite Datasets to Infinite Generators

Traditional financial machine learning relies on historical datasets (e.g., past 10 years of option prices), which suffer from **sparsity** (limited data points for deep OTM/ITM options) and **regime bias** (overfitting to specific historical market conditions).

The **PINN-BTC** framework eliminates these limitations by abandoning static datasets entirely. Instead, we employ a **Dynamic Stochastic Generator** that synthesizes physics-compliant training batches on-the-fly. This approach ensures that the model never sees the exact same scenario twice, effectively training on an infinite stream of market states.

---

## 2. The "7.5 Billion Points" Scale (Code-Verified)

A distinctive feature of this solver is the sheer magnitude of the training domain coverage. By leveraging the governing PDE constraints rather than labeled data, we can sample the high-dimensional input space $(S, K, \tau, r, \sigma)$ with unprecedented density.

Based on the training configuration (`train_config.yaml`) and the multi-objective loss structure, the data generation per epoch is decomposed as follows:

* **Initial Condition (IVP):** 10,000 samples (Payoff at $\tau=0$)
* **Lower Boundary (BC_Min):** 10,000 samples (Spot $\to 0$)
* **Upper Boundary (BC_Max):** 10,000 samples (Spot $\to S_{max}$)
* **PDE Interior Domain:** 40,000 samples (Physics consistency check)
    * *Multiplier: 4.0x of base sample size*
* **Strike Singularity (Kink):** 5,000 samples (Hard attention at $S=K$)
    * *Multiplier: 0.5x of base sample size*

**Total Unique Scenarios per Epoch:**

```math
10k + 10k + 10k + 40k + 5k = \mathbf{75,000 \text{ points}}
```

**Total Training Exposure (100,000 Epochs):**

```math
75,000 \times 100,000 = \mathbf{7,500,000,000 \text{ (7.5 Billion Points)}}
```

This scale ensures robust generalization across all theoretical market regimes, far exceeding the capacity of any historical database.

---

## 3. Mixed-Distribution Sampling Logic

To prevent the "curse of dimensionality" and focus learning on critical regions, we employ a **Mixed-Distribution Strategy**. The visualization below demonstrates how the generator concentrates data points in high-importance areas (Near-Maturity and At-the-Money).

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/data_sampling_distribution.png" alt="Data Sampling Distribution" width=65%" />
</p>

*Figure 1*: Distribution of training points across Time to Maturity (X-axis) and Spot Price (Y-axis). Notice the density increase as $\tau \to 0$ (left side) and around the Strike Price, effectively capturing the regions of highest non-linearity.

### 3.1 Time to Maturity: Power-Law Sampling
Option pricing dynamics are most non-linear and volatile as time to maturity approaches zero ($\tau \to 0$). Uniform sampling would waste computational resources on long-dated options where the price surface is flat.

We utilize a **Power-Law Distribution** with power $\alpha = 2.0$ to concentrate sampling density near expiration:

```math
\tau \sim \text{PowerLaw}(\alpha=2.0, \text{range}=[0, T_{max}])
```

This strategy forces the model to "pay more attention" to the high-gamma regions where hedging errors are typically highest.

### 3.2 Spot Price: Log-Normal Moneyness
Since asset prices cannot be negative and typically follow geometric Brownian motion, we sample the **Moneyness ratio** ($M = S/K$) using a Log-Normal distribution centered at $M=1.0$ (At-the-Money):

```math
\ln(S/K) \sim \mathcal{N}(0, \sigma_{sample}^2)
```

<p align="center">
  <img src="../models/call/train_2025-12-21_10-22-39_call/moneyness_density_mixed.png" alt="Moneyness Density" width=65%" />
</p>

*Figure 2*: Probability Density Function of the sampled Moneyness. The distribution is centered at 1.0 (ATM) to maximize precision for active trading zones, while the heavy tails ensure the model learns to price deep ITM/OTM options (Black Swan events).

This ensures that the model is trained extensively on At-the-Money (ATM) and Near-the-Money options, which constitute the majority of trading volume.

### 3.3 Volatility & Interest Rates: Uniform Regimes
To ensure the model is "Universal," we sample volatility ($\sigma$) and risk-free rates ($r$) uniformly across a wide theoretical spectrum:
* $\sigma \sim \mathcal{U}[0.1, 2.0]$ (Covering both stable and black-swan crypto regimes).
* $r \sim \mathcal{U}[0.0, 0.15]$ (Covering various DeFi staking yield environments).

---

## 4. Dynamic Domain Randomization

The `DataGenerator` module refreshes the random seed and resamples the entire batch at the start of every epoch. This technique, known as **Domain Randomization**, prevents the neural network from memorizing specific data points.

Consequently, the Validation Loss serves as a true measure of the model's understanding of the Black-Scholes physics, rather than a measure of its ability to recall training examples.