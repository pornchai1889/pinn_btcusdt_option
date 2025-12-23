# src/api/services.py
import os
import time
import yaml
import torch
import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

# Project Modules
from src.models.pinn_net import UniversalPINN
from src.data.normalizer import MarketNormalizer
from src.api.schemas import (
    OptionPricingRequest,
    PricingResponse,
    GreeksResponse,
    OptionType,
)

# Setup Logger
logger = logging.getLogger("PINN_Inference_Service")
logger.setLevel(logging.INFO)


class ModelBundle:
    """
    Container class to hold all artifacts required for a specific option type (Call/Put).
    Includes the Neural Network, Normalizer, and Configuration.
    """

    def __init__(self, run_dir: str, device: torch.device):
        self.device = device
        self.run_dir = Path(run_dir)

        # 1. Load Configuration
        self.config = self._load_config()

        # 2. Initialize Normalizer (Crucial for correct input scaling)
        self.normalizer = MarketNormalizer(self.config)

        # 3. Load Model Architecture & Weights
        self.model = self._load_model()

        # Cache Normalization Ranges for efficient Chain Rule calculations (Greeks)
        self._cache_ranges()

    def _load_config(self) -> Dict[str, Any]:
        """Loads config.yaml from the run directory."""
        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            # Fallback to config.json if yaml is missing
            config_path = self.run_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found in {self.run_dir}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self) -> UniversalPINN:
        """Initializes the PINN and loads pre-trained weights."""
        model = UniversalPINN(self.config).to(self.device)

        # Determine weight path (Prefer .pth in root, then checkpoints)
        weights_path = self.run_dir / "model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        logger.info(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode

        # Freeze parameters to prevent accidental gradients accumulation
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _cache_ranges(self) -> None:
        """
        Pre-calculates scaling factors for Autograd Chain Rule.
        Optimization: Avoiding repeated calculations during inference.
        """
        # Extract ranges from config (assumes [min, max])
        m = self.config["market"]
        
        # [Fixed] Explicitly store ranges as attributes so InferenceEngine can access them
        self.t_range = m["t_range"]
        self.S_range = m["S_range"]
        self.K_range = m["K_range"]
        self.sigma_range = m["sigma_range"]
        self.r_range = m["r_range"]
        
        self.t_min, self.t_max = self.t_range
        self.S_min, self.S_max = self.S_range
        self.K_min, self.K_max = self.K_range
        self.sig_min, self.sig_max = self.sigma_range
        self.r_min, self.r_max = self.r_range
        
        # Calculate denominators for chain rule derivatives
        # d(Norm)/d(Real) = 1 / (Max - Min)
        self.dt_scale = 1.0 / (self.t_max - self.t_min)
        self.dS_scale = 1.0 / (self.S_max - self.S_min)
        self.dsig_scale = 1.0 / (self.sig_max - self.sig_min)
        self.dr_scale = 1.0 / (self.r_max - self.r_min)


class InferenceEngine:
    """
    Core Business Logic Layer.
    Handles the lifecycle of PINN models and performs high-performance inference using Autograd.
    """

    def __init__(
        self, call_model_dir: str, put_model_dir: str, device_str: str = "cpu"
    ):
        """
        Initializes the engine with dual models (Call & Put).

        Args:
            call_model_dir: Path to the training run directory for Call Option.
            put_model_dir: Path to the training run directory for Put Option.
            device_str: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Inference Engine on Device: {self.device}")

        # Initialize Model Bundles
        logger.info("Loading CALL Model Bundle...")
        self.call_bundle = ModelBundle(call_model_dir, self.device)

        logger.info("Loading PUT Model Bundle...")
        self.put_bundle = ModelBundle(put_model_dir, self.device)

        logger.info("Inference Engine Ready.")

    def predict(self, request: OptionPricingRequest) -> PricingResponse:
        """
        Main entry point for pricing requests.
        Orchestrates input normalization, model inference, and Greek calculation.
        """
        start_time = time.perf_counter()

        # 1. Select appropriate model bundle
        bundle = (
            self.call_bundle
            if request.option_type == OptionType.CALL
            else self.put_bundle
        )

        # 2. Prepare Tensors with Gradient Tracking (Essential for Greeks)
        # Inputs: t, S, sigma, r, K
        inputs_norm, K_norm = self._prepare_inputs(request, bundle)

        # 3. Perform Inference & Autograd
        price, greeks = self._compute_price_and_greeks(inputs_norm, request, bundle)

        # 4. Construct Response
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return PricingResponse(
            price=price,
            greeks=greeks,
            model_version=bundle.run_dir.name,
            inference_time_ms=round(latency_ms, 4),
            device=str(self.device),
        )

    def _prepare_inputs(
        self, req: OptionPricingRequest, bundle: ModelBundle
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes inputs and creates PyTorch tensors requiring gradients.
        """
        # Normalize inputs using the bundle's normalizer
        # Note: We create numpy arrays first
        t_n = bundle.normalizer._normalize(req.time_to_maturity, bundle.t_range)
        S_n = bundle.normalizer._normalize(req.spot_price, bundle.S_range)
        sig_n = bundle.normalizer._normalize(req.volatility, bundle.sigma_range)
        r_n = bundle.normalizer._normalize(req.risk_free_rate, bundle.r_range)
        K_n = bundle.normalizer._normalize(req.strike_price, bundle.K_range)

        # Stack into tensor: [batch_size=1, n_inputs=5]
        # Order must match training: t, S, sigma, r, K
        x_np = np.array([[t_n, S_n, sig_n, r_n, K_n]], dtype=np.float32)

        # Create Tensor and enable grad
        x_tensor = torch.tensor(x_np, device=self.device, requires_grad=True)

        # Also return Normalized K tensor for denormalization later
        # (Since Model Output is V/K, we need K_norm to reconstruct V, but actually we need Real K)
        # We pass K_norm just in case, but usually we multiply by Real K.
        # However, for Autograd w.r.t inputs, x_tensor contains K_norm at index 4.

        return x_tensor, torch.tensor([K_n], device=self.device)

    def _compute_price_and_greeks(
        self, x_tensor: torch.Tensor, req: OptionPricingRequest, bundle: ModelBundle
    ) -> Tuple[float, GreeksResponse]:
        """
        Executes the PINN forward pass and computes 1st & 2nd order derivatives.
        Uses exact analytical gradients via PyTorch Autograd.
        """
        # --- Forward Pass ---
        # Model outputs Normalized Price Ratio (V / K)
        v_ratio_pred = bundle.model(x_tensor)

        # Real Price V = Ratio * K
        # We perform this multiplication symbolically to let gradients flow back to K if needed
        # But 'req.strike_price' is a float constants here for the multiplication scope.
        # To get proper derivatives w.r.t S, t, etc., we treat V_real as a function of x_tensor.
        # But wait, K is also an input in x_tensor[:, 4].
        # For Greeks definition, K is constant when finding Delta (dV/dS).

        # Calculate Real Price for Output
        price_pred = v_ratio_pred.item() * req.strike_price

        # --- Autograd for Greeks ---
        # We need gradients of V_ratio w.r.t Normalized Inputs [t_n, S_n, sig_n, r_n, K_n]
        grads = torch.autograd.grad(
            outputs=v_ratio_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(v_ratio_pred),
            create_graph=True,  # Required for higher-order derivatives (Gamma)
        )[0]

        # Extract Normalized Gradients
        # x_tensor columns: [0]:t, [1]:S, [2]:sigma, [3]:r, [4]:K
        dv_dt_n = grads[0, 0]
        dv_ds_n = grads[0, 1]
        dv_dsig_n = grads[0, 2]
        dv_dr_n = grads[0, 3]

        # --- Second Order Derivative (Gamma) ---
        # d^2V / dS^2 -> We need grad of (dv_ds_n) w.r.t S_n
        grad_gamma = torch.autograd.grad(
            outputs=dv_ds_n,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(dv_ds_n),
            create_graph=False,  # No need for 3rd order
        )[0]
        d2v_ds2_n = grad_gamma[0, 1]

        # --- Chain Rule: Convert Normalized Gradients to Physical Greeks ---
        # V = V_ratio * K_real
        # Let y = V_ratio
        # Delta = dV/dS = K * (dy/dS_n) * (dS_n/dS)

        K = req.strike_price

        # 1. Delta (dV/dS)
        delta = K * dv_ds_n.item() * bundle.dS_scale

        # 2. Gamma (d^2V/dS^2)
        # Gamma = d/dS (Delta) = K * d/dS_n (dy/dS_n * dS_scale) * dS_n/dS
        #       = K * (d^2y/dS_n^2) * (dS_scale)^2
        gamma = K * d2v_ds2_n.item() * (bundle.dS_scale**2)

        # 3. Theta (dV/dt)
        # Typically Theta is time decay (-dV/dt) or sensitivity to time passage.
        # PINN t input is usually "Time to Maturity" (tau).
        # If t = tau: dV/dt_calendar = - dV/dtau
        # We calculate sensitivity to input t (tau).
        theta_wrt_tau = K * dv_dt_n.item() * bundle.dt_scale
        # Financial Theta is usually negative per day or year. We return dV/d(TimeMaturity).
        # Usually Theta = dV/dt_calendar = - dV/d_tau.
        # We will return -1 * theta_wrt_tau to align with standard definition (Time Decay).
        theta = -1.0 * theta_wrt_tau

        # 4. Vega (dV/dSigma)
        vega = K * dv_dsig_n.item() * bundle.dsig_scale

        # 5. Rho (dV/dr)
        rho = K * dv_dr_n.item() * bundle.dr_scale

        return max(0.0, price_pred), GreeksResponse(
            delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho
        )


# ==============================================================================
# Factory / Singleton Instantiation (Optional)
# ==============================================================================
# This allows 'server.py' to import 'inference_engine' directly.
# Paths should ideally come from environment variables or a global config.
# For now, we leave it as None to be initialized by server.py startup event.
inference_engine: Optional[InferenceEngine] = None
