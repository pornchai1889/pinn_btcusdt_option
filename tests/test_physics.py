# tests/test_physics.py
import sys
import os
import pytest
import numpy as np
import torch
from scipy.stats import norm
from typing import Dict, Any, Union

# --- Environment Setup ---
# Add the project root to sys.path to ensure imports work correctly
# independent of where pytest is executed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.call_option import CallOption
from src.physics.put_option import PutOption


# =============================================================================
# Independent Benchmark Function (The "Golden Source")
# =============================================================================
def black_scholes_reference(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """
    Independent implementation of Black-Scholes formula using scipy.
    Used as the 'Ground Truth' to verify the project's physics engine.
    """
    # Prevent division by zero logic similar to the main code, but handled via numpy
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# =============================================================================
# Fixtures (Setup Configuration)
# =============================================================================
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Provides a standard mock configuration dictionary for initializing physics engines.
    """
    return {
        "market": {
            "t_range": [0.0, 1.0],
            "S_range": [0.0, 200.0],
            "K_range": [50.0, 150.0],
            "sigma_range": [0.1, 0.5],
            "r_range": [0.0, 0.1],
        },
        "training": {},
        "sampling": {},
    }


@pytest.fixture
def call_engine(mock_config: Dict[str, Any]) -> CallOption:
    return CallOption(mock_config)


@pytest.fixture
def put_engine(mock_config: Dict[str, Any]) -> PutOption:
    return PutOption(mock_config)


# =============================================================================
# Test Suite: Call Option Physics
# =============================================================================
class TestCallOptionPhysics:

    def test_payoff_logic(self, call_engine: CallOption) -> None:
        """
        Verifies the Call Option Payoff: max(S - K, 0).
        Checks both In-The-Money (ITM) and Out-Of-The-Money (OTM) scenarios.
        """
        K = 100.0
        # [Fixed]: Wrap K in numpy array to satisfy Mypy strict typing.
        # Since 's_itm' is a numpy array, 'K' must also be an array/tensor, not float.
        K_arr = np.array([K])

        # Case 1: ITM (S > K) -> Payoff = 20
        s_itm = np.array([120.0])
        val_itm = call_engine.payoff(s_itm, K_arr)
        np.testing.assert_allclose(val_itm, 20.0, err_msg="Call ITM Payoff incorrect")

        # Case 2: OTM (S < K) -> Payoff = 0
        s_otm = np.array([80.0])
        val_otm = call_engine.payoff(s_otm, K_arr)
        np.testing.assert_allclose(val_otm, 0.0, err_msg="Call OTM Payoff must be 0")

    def test_boundary_lower(self, call_engine: CallOption) -> None:
        """
        Verifies Lower Boundary Condition (S -> Minimum).
        For Call Option: V should be 0.
        """
        t = torch.tensor([0.5, 0.2])
        r = torch.tensor([0.05, 0.05])
        K = torch.tensor([100.0, 100.0])

        # Function expects t, r, K (S is implicitly 0 for lower bound function)
        val = call_engine.boundary_condition_lower(t, r, K)

        assert torch.all(val == 0), "Lower boundary for Call Option must be exactly 0"

    def test_analytical_accuracy(self, call_engine: CallOption) -> None:
        """
        Critical Test: Compare Physics Engine output against Independent Scipy Benchmark.
        This validates the mathematical integrity of the solution.
        """
        # Generate random test vectors
        np.random.seed(42)
        n_samples = 100
        t = np.random.uniform(0.1, 1.0, n_samples)
        S = np.random.uniform(50, 150, n_samples)
        K = np.random.uniform(80, 120, n_samples)
        r = np.random.uniform(0.01, 0.1, n_samples)
        sigma = np.random.uniform(0.1, 0.5, n_samples)

        # 1. Compute using Project Code
        project_prices = call_engine.analytical_solution(t, S, K, r, sigma)

        # 2. Compute using Independent Benchmark
        ref_prices = black_scholes_reference(S, K, t, r, sigma, option_type="call")

        # 3. Assert Closeness (Tolerance: 1e-5)
        np.testing.assert_allclose(
            project_prices,
            ref_prices,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Call Option Analytical Solution diverges from Scipy Benchmark!",
        )


# =============================================================================
# Test Suite: Put Option Physics
# =============================================================================
class TestPutOptionPhysics:

    def test_payoff_logic(self, put_engine: PutOption) -> None:
        """
        Verifies the Put Option Payoff: max(K - S, 0).
        """
        K = 100.0
        # [Fixed]: Wrap K in numpy array to satisfy Mypy strict typing.
        K_arr = np.array([K])

        # Case 1: ITM (S < K) -> Payoff = 20
        s_itm = np.array([80.0])
        val_itm = put_engine.payoff(s_itm, K_arr)
        np.testing.assert_allclose(val_itm, 20.0, err_msg="Put ITM Payoff incorrect")

        # Case 2: OTM (S > K) -> Payoff = 0
        s_otm = np.array([120.0])
        val_otm = put_engine.payoff(s_otm, K_arr)
        np.testing.assert_allclose(val_otm, 0.0, err_msg="Put OTM Payoff must be 0")

    def test_boundary_lower(self, put_engine: PutOption) -> None:
        """
        Verifies Lower Boundary Condition (S -> 0).
        For Put Option: V -> K * exp(-rt) (Present Value of Strike).
        """
        t_val = 1.0
        r_val = 0.05
        K_val = 100.0

        # Setup Tensors
        t = torch.tensor([t_val])
        r = torch.tensor([r_val])
        K = torch.tensor([K_val])

        # Compute Project Output
        val_project = put_engine.boundary_condition_lower(t, r, K)

        # Compute Expected Value manually
        val_expected = K_val * np.exp(-r_val * t_val)

        # [Fixed] Mypy Error: Argument 1 to "isclose" has incompatible type.
        # Solution: Assert type explicitly. This narrows the type for Mypy
        # and also verifies runtime correctness (Tensor inputs -> Tensor output).
        assert isinstance(
            val_project, torch.Tensor
        ), "Output must be a Tensor when inputs are Tensors"

        # Check closeness
        assert torch.isclose(
            val_project, torch.tensor(val_expected, dtype=torch.float32), atol=1e-5
        ), "Lower boundary for Put Option should approach PV(K)"

    def test_analytical_accuracy(self, put_engine: PutOption) -> None:
        """
        Critical Test: Compare Put Option implementation against Scipy Benchmark.
        """
        np.random.seed(123)  # Different seed
        n_samples = 100
        t = np.random.uniform(0.1, 1.0, n_samples)
        S = np.random.uniform(50, 150, n_samples)
        K = np.random.uniform(80, 120, n_samples)
        r = np.random.uniform(0.01, 0.1, n_samples)
        sigma = np.random.uniform(0.1, 0.5, n_samples)

        project_prices = put_engine.analytical_solution(t, S, K, r, sigma)
        ref_prices = black_scholes_reference(S, K, t, r, sigma, option_type="put")

        np.testing.assert_allclose(
            project_prices,
            ref_prices,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Put Option Analytical Solution diverges from Scipy Benchmark!",
        )


# =============================================================================
# Test Suite: Type Compatibility (Robustness)
# =============================================================================
def test_tensor_numpy_compatibility(call_engine: CallOption) -> None:
    """
    Ensures that payoff functions handle both PyTorch Tensors and NumPy arrays
    without crashing. Essential for hybrid data pipelines.
    """
    K_val = 100.0
    S_val = 120.0

    # Numpy Input
    res_np = call_engine.payoff(np.array([S_val]), np.array([K_val]))
    assert isinstance(
        res_np, np.ndarray
    ), "Failed to return numpy array for numpy input"

    # Tensor Input
    res_torch = call_engine.payoff(torch.tensor([S_val]), torch.tensor([K_val]))
    assert torch.is_tensor(res_torch), "Failed to return tensor for tensor input"

    # Value Check
    assert np.isclose(
        res_np.item(), res_torch.item()
    ), "Inconsistent results between Numpy and Tensor"
