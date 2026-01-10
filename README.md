# PINN-BTC: Physics-Informed Neural Networks for Real-time Bitcoin Option Pricing

[![CI Pipeline](https://github.com/pornchai1889/pinn_btcusdt_option/actions/workflows/ci.yml/badge.svg)](https://github.com/pornchai1889/pinn_btcusdt_option/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/pornchai1889/pinn_btcusdt_option/actions/workflows/cd.yml/badge.svg)](https://github.com/pornchai1889/pinn_btcusdt_option/actions/workflows/cd.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dockerized](https://img.shields.io/badge/docker-containerized-2496ED.svg?logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


> **A high-fidelity deep learning framework for pricing European Options via the Black-Scholes PDE. Featuring a novel mixed-distribution sampling strategy, Kink-Weighted Loss mechanism, and real-time validation against Binance market data.**

## Abstract

This study presents a novel **Physics-Informed Neural Network (PINN)** framework for real-time pricing of European options in the cryptocurrency market. While traditional deep learning approaches often suffer from overfitting due to reliance on limited historical data, and numerical methods (e.g., Finite Difference) entail high computational costs, our approach leverages the governing **Black-Scholes Partial Differential Equation (PDE)** directly as a regularization mechanism. This ensures theoretical consistency while maximizing generalization capabilities.

A key contribution of this work is the introduction of a **Mixed-Distribution Sampling Strategy**, which synthesizes Gaussian and Power-law time-distributed samples to train the model on millions of stochastic market scenarios, eliminating the dependency on labeled datasets. Furthermore, we address the gradient discontinuity problem at the strike price through a specialized **Weighted Kink Loss**, enabling the model to capture high-curvature pricing dynamics with superior precision compared to standard loss functions.

Empirical validation was conducted using real-world **Bitcoin (BTC/USDT)** options data retrieved via the Binance API. The results demonstrate that the proposed PINN framework not only aligns with analytical benchmarks but also exhibits robust performance in volatile market conditions. The final model is containerized and deployed via FastAPI, achieving millisecond-latency inference suitable for high-frequency trading and production-grade financial engineering tasks.

## 1. Introduction

Financial derivatives pricing, particularly for cryptocurrency options, presents a unique challenge due to the extreme volatility and non-stationary nature of digital asset markets. Traditional pricing models, such as the **Black-Scholes-Merton (BSM)** framework, rely on rigid assumptions like constant volatility and geometric Brownian motion, which often fail to capture the "fat-tailed" distributions and volatility smiles observed in real-world crypto trading. Conversely, numerical methods like Finite Difference (FDM) or Monte Carlo simulations, while accurate, are computationally expensive and ill-suited for high-frequency trading (HFT) environments that demand millisecond-latency inference.

To bridge this gap, this project introduces a **Physics-Informed Neural Network (PINN)** solver that hybridizes the data-driven power of deep learning with the governing physical laws of financial mathematics. By explicitly embedding the Black-Scholes PDE into the neural network's loss function (as originally proposed by Raissi et al.), we constrain the search space of the model, ensuring that predictions remain theoretically consistent even in regions with sparse training data.

Our approach specifically addresses the limitations of standard deep learning in finance by:
1.  **Eliminating the need for massive labeled datasets:** Using a self-supervised physics loss.
2.  **Resolving the "Strike Price Singularity":** Through a novel **Kink-Weighted Loss** that captures the sharp payoff transitions.
3.  **Enabling Real-time Greeks:** Providing instantaneous sensitivities (Delta, Gamma, Vega) via automatic differentiation, crucial for dynamic hedging strategies.

## 2. Methodology

### 2.1 Governing Dynamics: Black-Scholes PDE
We formulate the option pricing problem effectively as a solution to the **Black-Scholes Partial Differential Equation (PDE)**. To enhance numerical stability and align with the model's inference logic (where options are priced based on remaining duration), we perform a variable transformation from calendar time $t$ to **Time to Maturity** $\tau = T - t$.

The governing dynamics for the option price $V(S, \tau)$ are defined by the linear parabolic PDE:

```math
\mathcal{F}(V) := \frac{\partial V}{\partial \tau} - \left( \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV \right) = 0
```

Subject to the domain constraints: $\tau \in [0, T]$ and $S \in [0, S_{max}]$.

### 2.2 Initial and Boundary Value Problems (IBVP)
The PDE is solved subject to specific boundary and initial conditions corresponding to European Options. We rigorously define these conditions for both Call and Put options as follows:

#### **Initial Condition (Payoff at $\tau=0$)**
At maturity (time to maturity $\tau=0$), the option price must converge to the intrinsic payoff. This creates a non-differentiable point ("Kink") at the strike price $K$:

```math
V(S, 0) =
\begin{cases} 
\max(S - K, 0) & \text{for Call Option} \\
\max(K - S, 0) & \text{for Put Option}
\end{cases}
```

#### **Boundary Conditions (Spatial Extremes)**
We enforce Dirichlet boundary conditions derived from the asymptotic behavior of the asset price $S$:

* **Lower Boundary ($S \to 0$):**
```math
V(S, \tau) =
\begin{cases} 
0 & \text{for Call Option} \\
K e^{-r\tau} & \text{for Put Option}
\end{cases}
```

* **Upper Boundary ($S \to S_{max}$):**
```math
V(S, \tau) =
\begin{cases} 
S - K e^{-r\tau} & \text{for Call Option} \\
0 & \text{for Put Option}
\end{cases}
```

### 2.3 Analytical Benchmark (Exact Solution)
To validate the PINN's accuracy, we compare predictions against the closed-form **Black-Scholes Analytical Solution** ($V_{exact}$), defined as:

```math
\begin{aligned}
V_{call} &= S \cdot \Phi(d_1) - K e^{-r\tau} \cdot \Phi(d_2) \\
V_{put} &= K e^{-r\tau} \cdot \Phi(-d_2) - S \cdot \Phi(-d_1)
\end{aligned}
```

Where $\Phi(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution, and:

```math
d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}
```

### 2.4 PINN Architecture & Composite Loss Landscape
We approximate the solution $V(S, \tau)$ using a Deep Neural Network $\hat{V}(S, \tau, K, r, \sigma; \theta)$. The network parameters $\theta$ are optimized by minimizing a composite loss function $\mathcal{L}_{total}$ that strictly enforces physical laws and boundary constraints.

A key innovation of this framework is the introduction of a **Weighted Kink Loss** ($\mathcal{L}_{Kink}$), which applies "Hard Attention" to the singularity at the strike price ($S=K, \tau=0$) to resolve gradient vanishing issues common in standard PINNs.

```math
\mathcal{L}_{total} = \lambda_{PDE}\mathcal{L}_{PDE} + \lambda_{IVP}\mathcal{L}_{IVP} + \lambda_{BVP}\mathcal{L}_{BVP} + \lambda_{Kink}\mathcal{L}_{Kink}
```

The individual loss components are defined as Mean Squared Errors (MSE):

1.  **Physics Loss (PDE Residual):** Enforces the Black-Scholes equation on collocation points ($N_{PDE}$).
```math
\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left( \frac{\partial \hat{V}}{\partial \tau} - \left[ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 \hat{V}}{\partial S^2} + rS \frac{\partial \hat{V}}{\partial S} - r\hat{V} \right] \right)^2
```

2.  **Initial Value Loss (Payoff):** Enforces the payoff structure at maturity ($N_{IVP}$).
```math
\mathcal{L}_{IVP} = \frac{1}{N_{IVP}} \sum_{i=1}^{N_{IVP}} \left( \hat{V}(S_i, 0) - \text{Payoff}(S_i) \right)^2
```

3.  **Boundary Value Loss:** Enforces asymptotic behavior at $S_{min}$ and $S_{max}$ ($N_{BVP}$).
```math
\mathcal{L}_{BVP} = \frac{1}{N_{BVP}} \left[ \sum \left( \hat{V}(S_{min}, \tau) - V_{LB} \right)^2 + \sum \left( \hat{V}(S_{max}, \tau) - V_{UB} \right)^2 \right]
```

4.  **Kink Loss (Hard Attention):** Specifically targets the critical point $S=K$ at $\tau=0$ to sharpen the hinge.
```math
\mathcal{L}_{Kink} = \frac{1}{N_{Kink}} \sum_{j=1}^{N_{Kink}} \left( \hat{V}(K_j, 0) \right)^2
```
Note: For both Call and Put options, the payoff at $S=K$ is exactly 0.

## 3. Performance & Validation

To demonstrate the robustness of the PINN-BTC framework, we conducted extensive evaluations across three dimensions: convergence stability, analytical accuracy against the Black-Scholes benchmark, and empirical generalization to real-world cryptocurrency market data.

### 3.1 Training Convergence & Loss Landscape
The model was trained for 100,000 epochs using the composite loss function described in the methodology. We monitor the evolution of individual loss components—PDE residual, Boundary conditions (IVP/BVP), and the **Weighted Kink Loss**—to ensure balanced optimization.

* **Convergence Behavior:** The training history reveals that the specialized **Kink Loss** effectively accelerates learning at the critical strike price region ($S=K$), preventing the "smoothing" artifacts typically observed in standard PINNs near non-differentiable points.
* **Loss Visualization:** (See `runs/` directory for generated plots)
    * *Log-scale Loss History:* Demonstrates the steady decay of physics-informed residuals ($\mathcal{L}_{PDE}$) alongside data-driven constraints.

### 3.2 Analytical Benchmarking (In-Silico Validation)
We benchmarked the trained model against the exact Black-Scholes analytical solution across the entire domain $\tau \in [0, T]$ and $S \in [S_{min}, S_{max}]$.

#### **Pricing Accuracy & Error Heatmaps**
Visual inspections via `plot_solution_snapshot` confirm high-fidelity reconstruction of the option pricing surface:
* **Heatmaps:** The absolute error distribution shows negligible deviation ($< 10^{-4}$) in the ITM/OTM regions, with minor localized errors strictly confined to the ATM boundary, validating the efficacy of the hard-attention mechanism.
* **2D Slices:**
    * *Price vs. Spot ($S$):* Perfectly matches the analytical curve at varying times ($t=0, t=T/2$), capturing the convex payoff structure.
    * *Price vs. Time ($\tau$):* Accurately tracks time-decay (Theta) characteristics.

### 3.3 Real-World Market Validation (Binance Data)
Beyond theoretical benchmarks, we validated the model's performance on live **BTC/USDT** options data fetched directly from the **Binance API**.

* **Methodology:** Real-time market inputs (Spot, Strike, Time to Maturity) were fed into the model. Volatility ($\sigma$) was estimated using a trailing historical volatility window to align the physical model with market realities.
* **Performance Metrics:**
    * **Corr:** $> 0.98$ (Indicating strong correlation with market pricing).
    * **MAPE (Mean Absolute Percentage Error):** Demonstrates the model's practical viability for pricing ATM and liquid ITM options.
* **Visual Validation:** The `market_vs_model` plots illustrate that the PINN predictions tightly track the actual market bid-ask midpoints, validating the **Mixed-Distribution Sampling** strategy's ability to generalize to unseen, noisy real-world data.

## 4. Technical Implementation

To transition from theoretical modeling to a production-ready artifact, the framework is engineered with a modular architecture focusing on scalability, security, and inference latency. The implementation pipeline consists of three core stages:

### 4.1 End-to-End Workflow
The project follows a rigorous data-to-deployment pipeline:
1.  **Data Synthesis:** The `DataGenerator` module dynamically creates mixed-distribution training batches (Gaussian + Power-law) on-the-fly, eliminating storage bottlenecks.
2.  **Model Training:** Executed via the `Trainer` engine with configurable loss weights and automatic checkpointing to TensorBoard for real-time monitoring.
3.  **Validation:** Automated evaluation against both Analytical Solutions (Black-Scholes) and Real-Market Data (Binance) ensures theoretical and practical integrity.

### 4.2 High-Performance Inference Engine (API)
The trained models are served via a **FastAPI** microservice, designed for high-throughput financial applications:
* **Asynchronous Architecture:** Utilizing Python's `async/await` paradigm to handle concurrent pricing requests without blocking the computation thread.
* **Input Validation:** Strict type enforcement using **Pydantic** schemas ensures that market parameters (Spot, Strike, Time) fall within the valid training domain before inference.
* **Dual-Model Serving:** The engine simultaneously loads both Call and Put option models, routing requests dynamically based on the instrument type.

### 4.3 Containerization & Deployment
The application is fully containerized using **Docker**, adhering to DevSecOps best practices:
* **Multi-Stage Build:** Separates the build environment from the runtime environment to minimize image size.
* **CPU Optimization:** Explicitly targets PyTorch CPU-only binaries to reduce the container footprint (~60% size reduction) and cost, making it suitable for serverless deployment (e.g., AWS Lambda, Google Cloud Run).
* **Security Hardening:** The container runs as a non-root user (`appuser`) to mitigate privilege escalation risks.

### 4.4 CI/CD Automation
Continuous Integration and Deployment are managed via **GitHub Actions**:
* **Automated Testing:** Unit tests for physics logic and gradient computations are triggered on every push.
* **Linting & Formatting:** Enforces code quality standards using `ruff` or `flake8`.

## 5. Installation & Usage

To facilitate reproducibility and seamless integration into quantitative finance pipelines, we provide two deployment methods: a **Dockerized environment** (recommended for production/inference) and a **Local Python environment** (recommended for research/training).

### 5.1 Prerequisites
* **Docker Engine:** Version 20.10+ (for containerized deployment).
* **Python:** Version 3.10+ (for local development).
* **Git:** To clone the repository.

### 5.2 Docker Deployment
We utilize a multi-stage Docker build optimized for CPU inference, significantly reducing the image size and ensuring consistency across different operating systems.

**1. Build the Image:**
```bash
docker build -t pinn-btc-option .
```
**2. Run the Container:**
Expose the FastAPI inference engine on port 8000:
```bash
docker run -d -p 8000:8000 --name pinn-container pinn-btc-option
```

**3. Test the API:** Once running, the interactive documentation (Swagger UI) is available at http://localhost:8000/docs. You can also send a pricing request via curl:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/price' \
  -H 'Content-Type: application/json' \
  -d '{
  "spot_price": 45000.0,
  "strike_price": 46000.0,
  "time_to_maturity": 0.5,
  "risk_free_rate": 0.02,
  "volatility": 0.6,
  "option_type": "call"
}'
```

### 5.3 Local Development Setup
For researchers intending to modify the network architecture or loss functions:

**1. Clone and Setup:**
```bash
git clone [https://github.com/pornchai1889/pinn_btcusdt_option.git](https://github.com/pornchai1889/pinn_btcusdt_option.git)
cd pinn_btcusdt_option

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**2. Install Dependencies:** We enforce strict versioning to prevent dependency conflicts.
```bash
pip install -r requirements.txt
```

### 5.4 Operational Commands
**Training the Model**
to train the PINN from scratch using the mixed-distribution sampling strategy:
```bash
python train.py --config configs/train_config.yaml
```

* Outputs: Checkpoints are saved to models/ and logs to runs/ (viewable via TensorBoard).

**Fine-Tuning (Transfer Learning)**
to adapt a pre-trained model to a specific market regime using recent data:
```bash
python finetune.py --base_model models/call/latest.pth --config configs/finetune_config.yaml
```

## 6. Project Structure

The repository is organized into modular components to ensure separation of concerns between the physical modeling, deep learning architecture, and deployment services. This structure facilitates reproducibility and scalability.

```text
pinn_btcusdt_option/
├── .github/workflows/      # CI/CD pipelines for automated testing and deployment
├── configs/                # YAML configuration files for training, fine-tuning, and evaluation
├── models/                 # Directory for storing serialized model artifacts (Call/Put)
├── scripts/                # Standalone scripts for market data acquisition and validation
├── src/                    # Core source code directory
│   ├── api/                # FastAPI inference engine, routes, and Pydantic schemas
│   ├── core/               # Training logic, loss aggregation, and optimization loops
│   ├── data/               # Data synthesis (mixed-sampling) and Binance API connectors
│   ├── models/             # Deep Neural Network architecture definitions
│   ├── physics/            # Black-Scholes PDE formulations and boundary conditions
│   └── utils/              # Helper modules for visualization, metrics, and logging
├── tests/                  # Unit tests ensuring physics consistency and code reliability
├── Dockerfile              # Multi-stage build configuration for containerized deployment
├── train.py                # Main entry point for training the PINN model
├── finetune.py             # Entry point for transfer learning on specific market regimes
└── requirements.txt        # Python dependency specifications
```

## 7. References & Citations

This work builds upon foundational research in scientific machine learning and quantitative finance. We explicitly acknowledge and cite the following pioneering works:

### 7.1 Key Literature
* **[1] Physics-Informed Neural Networks (Original Framework):**
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
* **[2] PINNs for Option Pricing:**
    Dhiman, A., & Hu, Y. (2021). *Physics Informed Neural Network for Option Pricing*. Georgia Institute of Technology, CS7643 Deep Learning Project.
* **[3] Black-Scholes Model:**
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.

### 7.2 Citing This Project
If you use this code or methodology in your research, please consider citing it as follows:

```bibtex
@misc{pinn_btc_option_2026,
  author = {Pornchai Khamdaeng},
  title = {PINN-BTC: A Physics-Informed Neural Network for Real-time Bitcoin Option Pricing},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/pornchai1889/pinn_btcusdt_option](https://github.com/pornchai1889/pinn_btcusdt_option)}},
  note = {Evaluating Black-Scholes PDE constraints with Mixed-Distribution Sampling and Kink-Weighted Loss}
}
