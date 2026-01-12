# Methodology: Neural Architecture & Physics-Informed Optimization

## 1. Deep Neural Network Architecture

The approximation of the option pricing function $V(S, K, \tau, r, \sigma)$ is parameterized by a fully connected Deep Neural Network (DNN), denoted as $\hat{V}(x; \theta)$. The architecture is meticulously designed to balance expressivity with the smoothness required for higher-order derivative computations.

### 1.1 Network Topology
We employ a **Multi-Layer Perceptron (MLP)** architecture with the following specifications:
* **Input Layer:** 5 Neurons, corresponding to the state vector $\mathbf{x} = [S, K, \tau, r, \sigma]^T$.
* **Hidden Layers:** 4 layers, each containing 256 neurons. This depth allows the model to capture the complex, non-linear pricing surfaces characteristic of volatile markets.
* **Output Layer:** 1 Neuron, representing the normalized option price.

![PINN Architecture Diagram](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-023-49977-3/MediaObjects/41598_2023_49977_Fig1_HTML.png?as=webp)
*Figure 1*: Schematic representation of the Physics-Informed Neural Network (PINN) architecture. The workflow demonstrates how domain variables (a) are processed through the neural network (b) to produce design variables (c), which are then optimized via a composite loss function (d) incorporating physics constraints. Adapted from [Nature Scientific Reports](https://www.nature.com/articles/s41598-023-49977-3).

In the context of our **PINN-BTC** solver, this architecture is implemented as follows: The input layer **(a)** accepts the 5-dimensional market state vector $(S, K, \tau, r, \sigma)$. These inputs propagate through the fully connected perceptrons **(b)** to predict the normalized option price **(c)**. Crucially, the optimization **(d)** is not driven by labeled labels, but by the "Physics Loss" (Black-Scholes Residual) and "Constraint Loss" (Boundary & Kink conditions), enabling the model to learn financial laws purely through mathematical constraints.

### 1.2 Activation Functions
The choice of activation functions is critical for Physics-Informed learning, as the network must be twice differentiable ($C^2$ continuous) to satisfy the Black-Scholes PDE.

1.  **Hidden Layers (Tanh):** We utilize the Hyperbolic Tangent (`Tanh`) activation. Unlike `ReLU` (which has zero second derivatives), `Tanh` provides smooth, non-vanishing gradients for $\frac{\partial^2 V}{\partial S^2}$, essential for calculating the Gamma ($\Gamma$) Greek and the diffusion term of the PDE.
2.  **Output Layer (Softplus):** To enforce the financial constraint that option prices cannot be negative ($V \ge 0$), the final output is passed through a `Softplus` activation:
    $$\text{Softplus}(x) = \ln(1 + e^x)$$
    This serves as a smooth approximation of the `ReLU` function, ensuring differentiability while preserving the non-negativity constraint.

---

## 2. Composite Physics-Informed Loss Function

The network parameters $\theta$ are optimized by minimizing a composite loss function $\mathcal{L}_{total}$, which transforms the learning problem into a multi-objective optimization task comprising physical laws, boundary constraints, and structural priors.

```math
\mathcal{L}_{total} = \lambda_{PDE}\mathcal{L}_{PDE} + \lambda_{IVP}\mathcal{L}_{IVP} + \lambda_{BVP}\mathcal{L}_{BVP} + \lambda_{Kink}\mathcal{L}_{Kink}
```

Where $\lambda$ denotes the scalar weighting coefficient for each term.

### 2.1 The Physics Residual ($\mathcal{L}_{PDE}$)
This term enforces the governing Black-Scholes dynamics across the interior of the domain. It is computed via **Automatic Differentiation (AD)**, ensuring exact gradient calculation without discretization errors.

```math
\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left\| \frac{\partial \hat{V}}{\partial \tau} - \left( \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 \hat{V}}{\partial S^2} + rS \frac{\partial \hat{V}}{\partial S} - r\hat{V} \right) \right\|^2
```

### 2.2 Boundary & Initial Value Constraints
To ensure a unique and valid solution, we enforce Dirichlet boundary conditions:
* **$\mathcal{L}_{IVP}$ (Payoff):** Enforces the solution at maturity ($\tau=0$) to match the contractual payoff $\max(S-K, 0)$ or $\max(K-S, 0)$.
* **$\mathcal{L}_{BVP}$ (Asymptotic):** Enforces the correct behavior as $S \to 0$ and $S \to S_{max}$.

---

## 3. The "Weighted Kink" Strategy (Hard Attention Mechanism)

A fundamental challenge in pricing options with neural networks is the **Gradient Discontinuity** at the strike price ($S=K$) at maturity ($\tau=0$). Standard loss functions often fail to capture this sharp "kink," leading to smoothed-out approximations and significant pricing errors near At-the-Money (ATM) regions.

To resolve this, we introduce a specialized **Weighted Kink Loss** that acts as a "Hard Attention" mechanism.

### 3.1 Formulation
We explicitly sample a dense batch of points $N_{Kink}$ exactly at the singularity coordinates $(S=K, \tau=0)$ and enforce the payoff condition with a disproportionately high penalty weight.

```math
\mathcal{L}_{Kink} = \frac{1}{N_{Kink}} \sum_{j=1}^{N_{Kink}} \left( \hat{V}(K_j, 0) - \text{Payoff}(K_j) \right)^2
```

### 3.2 Strategic Weighting
Based on empirical convergence analysis, we assign the highest priority to this term:
* **$\lambda_{Kink} = 100.0$**
* $\lambda_{IVP, BVP} = 20.0$
* $\lambda_{PDE} = 1.0$

This configuration forces the optimizer to prioritize resolving the singularity first, ensuring that the critical "hinge" shape of the option payoff is preserved with high fidelity before minimizing the residual error in the smooth regions.

---

## 4. Optimization Protocol

The model is trained using the **Adam** optimizer, a stochastic gradient descent method with adaptive moment estimation, chosen for its robustness in navigating the complex loss landscapes of PINNs.

* **Learning Rate:** $1 \times 10^{-4}$
* **Training Schedule:** 100,000 Epochs
* **Batch Strategy:** Dynamic resampling (See *Sampling Strategy*) prevents overfitting by presenting the model with "fresh" data points at every iteration.

This methodology results in a solver that is not only theoretically consistent with financial physics but also numerically robust in regimes where traditional numerical methods struggle.