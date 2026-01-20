# Mathematical Formulation & Physics-Informed Constraints

## 1. Governing Dynamics: The Black-Scholes PDE

The core physics governing the **PINN-BTC** solver is derived from the Black-Scholes-Merton framework. To facilitate real-time pricing and generalize across different expiration dates, we formulate the problem in terms of **Time to Maturity** ($\tau$) rather than calendar time ($t$).

Let $V(S, \tau)$ denote the price of a European option, where:
* $S \in [0, \infty)$: Spot price of the underlying asset (Bitcoin).
* $\tau = T - t \in [0, T]$: Time remaining until expiration.
* $K$: Strike price.
* $r$: Risk-free interest rate.
* $\sigma$: Volatility of the underlying asset.

The no-arbitrage price $V$ must satisfy the linear parabolic Partial Differential Equation (PDE):

```math
\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV
```

We define the differential operator $\mathcal{N}[V]$ representing the PDE residual, which the Neural Network minimizes:

```math
\mathcal{F}(V) := \frac{\partial V}{\partial \tau} - \left( \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV \right) = 0
```

---

## 2. Non-Dimensionalization & Network Input Scaling

Training a neural network directly on raw financial data (where $S \sim 10^5$ and $r \sim 10^{-2}$) leads to severe gradient instability and slow convergence. To mitigate this, we employ a strict **Non-dimensionalization Strategy** aligning with the scaling laws of fluid dynamics and quantitative finance.

### 2.1 Input Normalization
All inputs to the PINN $\hat{V}(S, K, \tau, r, \sigma; \theta)$ are normalized to the domain $\mathcal{D}_{norm} \approx [0, 1]$ before passing through the hidden layers:

```math
\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}, \quad \text{where } x \in \{S, K, \tau, r, \sigma\}
```

### 2.2 Output Scaling (Homogeneity)
Exploiting the linear homogeneity of the Black-Scholes equation with respect to the strike price $K$, the network is trained to predict the **Normalized Option Price** (Moneyness-adjusted value) rather than the absolute price:

```math
\hat{V}_{net} \approx \frac{V}{K}
```

The physical price is reconstructed during inference and loss computation as:

```math
V_{pred} = \hat{V}_{net} \cdot K
```

This technique decouples the magnitude of the asset price from the learning process, allowing the model to generalize seamlessly to any price range (e.g., from 100$ to 1,000,000$).

---

## 3. Initial and Boundary Value Problems (IBVP)

The solution to the PDE is unique only when constrained by specific Initial Conditions (IC) and Boundary Conditions (BC). We rigorously enforce these constraints via the loss function components $L_{IVP}$ and $L_{BVP}$.

### 3.1 Initial Condition (Payoff at $\tau=0$)
At the moment of expiration ($\tau=0$), the option value is deterministic. This introduces a $C^0$ continuity but $C^1$ discontinuity (a "Kink") at $S=K$.

**For Call Options:**
```math
V(S, 0) = \max(S - K, 0)
```

**For Put Options:**
```math
V(S, 0) = \max(K - S, 0)
```

> **Note:** The non-differentiability at $S=K$ is the primary motivation for our **Weighted Kink Loss ($\lambda_{Kink}$)** strategy, which applies hard attention to this singular point.

### 3.2 Boundary Conditions
We apply Dirichlet boundary conditions based on the asymptotic behavior of the asset.

#### Lower Boundary ($S \to 0$)
As the asset price approaches zero:
* **Call:** The option becomes worthless.

```math
\lim_{S \to 0} V_{call}(S, \tau) = 0
```

* **Put:** The option value approaches the present value of the strike.

```math
\lim_{S \to 0} V_{put}(S, \tau) = K e^{-r\tau}
```

#### Upper Boundary ($S \to S_{max}$)
As the asset price increases significantly ($S \gg K$):
* **Call:** The option behaves like a forward contract.

```math
\lim_{S \to \infty} V_{call}(S, \tau) \approx S - K e^{-r\tau}
```

* **Put:** The option becomes worthless (Deep Out-of-the-Money).

```math
\lim_{S \to \infty} V_{put}(S, \tau) = 0
```

---

## 4. Analytical Benchmarks (Exact Solution)

For validation purposes, we compare the PINN's approximation against the closed-form Black-Scholes analytical solution. The exact prices for European Call ($C$) and Put ($P$) options are formulated as follows:

**Call Option Price:**
```math
C(S, \tau) = S\Phi(d_1) - Ke^{-r\tau}\Phi(d_2)
```

**Put Option Price:**
```math
P(S, \tau) = Ke^{-r\tau}\Phi(-d_2) - S\Phi(-d_1)
```

Where $\Phi(\cdot)$ denotes the standard normal cumulative distribution function (CDF), and the auxiliary variables $d_1$ and $d_2$ are defined as:

```math
d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}
```

```math
d_2 = d_1 - \sigma\sqrt{\tau}
```

---

## 5. Greek Letters (Automatic Differentiation)

A significant advantage of the PINN framework is the ability to compute financial Greeks analytically via Automatic Differentiation (AD), without finite difference approximations.

* **Delta ($\Delta$):** $\frac{\partial V}{\partial S}$
* **Gamma ($\Gamma$):** $\frac{\partial^2 V}{\partial S^2}$
* **Theta ($\Theta$):** $-\frac{\partial V}{\partial \tau}$
* **Vega ($\nu$):** $\frac{\partial V}{\partial \sigma}$ (Sensitivity to Volatility)
* **Rho ($\rho$):** $\frac{\partial V}{\partial r}$ (Sensitivity to Interest Rate)

These derivatives are essentially "free by-products" of the training graph, enabling real-time risk management.