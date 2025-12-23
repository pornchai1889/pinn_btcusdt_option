# tests/test_data_pipeline.py
import sys
import os
import pytest
import numpy as np
import torch
from typing import Dict, Any, Tuple

# --- Environment Setup ---
# Add the project root to sys.path to ensure imports work correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.normalizer import MarketNormalizer
from src.data.generator import DataGenerator
from src.physics.call_option import (
    CallOption,
)  # Used as a concrete implementation for testing


# =============================================================================
# Fixtures (Configuration & Instance Setup)
# =============================================================================
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Creates a comprehensive mock configuration covering Market and Sampling parameters.
    Ranges are chosen to be simple integers to make manual verification easier.
    """
    return {
        "market": {
            "t_range": [0.0, 1.0],  # Normalized 0->0, 1->1
            "S_range": [0.0, 200.0],  # Spot Price Range
            "K_range": [50.0, 150.0],  # Strike Price Range
            "sigma_range": [0.1, 0.5],  # Volatility Range
            "r_range": [0.0, 0.1],  # Risk-free Rate Range
            "K_step": 10.0,  # Discrete step for K sampling
        },
        "sampling": {
            "moneyness_range": [0.5, 1.5],
            "time_sampling_power": 2.0,
            "adaptive_std_factor": 1.0,
        },
    }


@pytest.fixture
def normalizer(mock_config: Dict[str, Any]) -> MarketNormalizer:
    return MarketNormalizer(mock_config)


@pytest.fixture
def physics_engine(mock_config: Dict[str, Any]) -> CallOption:
    """
    We need a Physics Engine to initialize DataGenerator.
    Using CallOption as the standard test subject.
    """
    return CallOption(mock_config)


@pytest.fixture
def data_generator(
    mock_config: Dict[str, Any],
    normalizer: MarketNormalizer,
    physics_engine: CallOption,
) -> DataGenerator:
    return DataGenerator(mock_config, normalizer, physics_engine)


# =============================================================================
# Test Suite 1: Market Normalizer (Data Integrity)
# =============================================================================
class TestMarketNormalizer:

    def test_normalization_range_integrity(self, normalizer: MarketNormalizer) -> None:
        """
        Verifies that input values within valid ranges are mapped exactly to [0, 1].
        """
        # Create a batch of min and max values for all parameters
        # Order: t, S, sigma, r, K

        # Case 1: Minimum values -> Should normalize to 0.0
        min_vals = np.array([[0.0, 0.0, 0.1, 0.0, 50.0]])  # t, S, sigma, r, K
        norm_min = normalizer.normalize_batch(
            min_vals[:, 0:1],
            min_vals[:, 1:2],
            min_vals[:, 2:3],
            min_vals[:, 3:4],
            min_vals[:, 4:5],
        )
        np.testing.assert_allclose(
            norm_min, 0.0, atol=1e-7, err_msg="Min values must normalize to 0.0"
        )

        # Case 2: Maximum values -> Should normalize to 1.0
        max_vals = np.array([[1.0, 200.0, 0.5, 0.1, 150.0]])  # t, S, sigma, r, K
        norm_max = normalizer.normalize_batch(
            max_vals[:, 0:1],
            max_vals[:, 1:2],
            max_vals[:, 2:3],
            max_vals[:, 3:4],
            max_vals[:, 4:5],
        )
        np.testing.assert_allclose(
            norm_max, 1.0, atol=1e-7, err_msg="Max values must normalize to 1.0"
        )

    def test_reversibility_round_trip(self, normalizer: MarketNormalizer) -> None:
        """
        CRITICAL: Verifies that Normalize -> Denormalize returns the EXACT original values.
        x_orig == Denorm(Norm(x_orig))
        """
        # Random physical values within valid ranges
        np.random.seed(42)
        n = 100
        t = np.random.uniform(0.0, 1.0, (n, 1))
        S = np.random.uniform(0.0, 200.0, (n, 1))
        sigma = np.random.uniform(0.1, 0.5, (n, 1))
        r = np.random.uniform(0.0, 0.1, (n, 1))
        K = np.random.uniform(50.0, 150.0, (n, 1))

        # 1. Normalize
        x_norm = normalizer.normalize_batch(t, S, sigma, r, K)

        # 2. Denormalize
        t_rec, S_rec, sig_rec, r_rec, K_rec = normalizer.denormalize_batch(x_norm)

        # 3. Check Consistency
        # We assume clean floating point math, but allow tiny epsilon (1e-7)
        np.testing.assert_allclose(
            t, t_rec, atol=1e-7, err_msg="Time (t) failed round-trip"
        )
        np.testing.assert_allclose(
            S, S_rec, atol=1e-7, err_msg="Spot (S) failed round-trip"
        )
        np.testing.assert_allclose(
            sigma, sig_rec, atol=1e-7, err_msg="Sigma failed round-trip"
        )
        np.testing.assert_allclose(
            r, r_rec, atol=1e-7, err_msg="Risk-free rate (r) failed round-trip"
        )
        np.testing.assert_allclose(
            K, K_rec, atol=1e-7, err_msg="Strike (K) failed round-trip"
        )

    def test_price_normalization_logic(self, normalizer: MarketNormalizer) -> None:
        """
        Verifies Option Price normalization logic: V_norm = V / K.
        This is crucial for financial stability (moneyness scaling).
        """
        K = np.array([100.0, 200.0])
        price = np.array([10.0, 50.0])  # e.g. Call Price

        # Expected: 10/100=0.1, 50/200=0.25
        expected_norm = price / (K + 1e-8)

        # Test Normalize
        norm_res = normalizer.normalize_price(price, K)
        np.testing.assert_allclose(
            norm_res, expected_norm, err_msg="Price normalization calculation wrong"
        )

        # Test Denormalize (Reverse)
        rec_price = normalizer.denormalize_price(norm_res, K)
        np.testing.assert_allclose(
            rec_price, price, err_msg="Price denormalization failed round-trip"
        )


# =============================================================================
# Test Suite 2: Data Generator (Pipeline Logic)
# =============================================================================
class TestDataGenerator:

    def test_pde_batch_shape(self, data_generator: DataGenerator) -> None:
        """
        Ensures that the PDE batch generator returns the correct tensor shape (N, 5).
        Columns: [t, S, sigma, r, K]
        """
        n_samples = 50
        batch = data_generator.get_pde_batch(n_samples)

        assert isinstance(batch, np.ndarray), "PDE batch must be a numpy array"
        assert batch.shape == (
            n_samples,
            5,
        ), f"Expected shape ({n_samples}, 5), got {batch.shape}"

        # Check range roughly [0, 1]
        assert (
            batch.min() >= -0.1 and batch.max() <= 1.1
        ), "Normalized data drifted too far from [0,1]"

    def test_ivp_batch_logic(self, data_generator: DataGenerator) -> None:
        """
        Verifies Initial Value Problem (IVP) generation.
        Condition: t must be EXACTLY 0 (Normalized 0.0).
        """
        n_samples = 50
        X_ivp, y_ivp = data_generator.get_ivp_batch(n_samples)

        # Check X shape
        assert X_ivp.shape == (n_samples, 5)

        # Check t column (index 0) is all 0
        t_col = X_ivp[:, 0]
        np.testing.assert_allclose(
            t_col, 0.0, atol=1e-9, err_msg="IVP batch must have t=0"
        )

        # Check y shape (Option Price)
        assert y_ivp.shape == (n_samples, 1)

    def test_kink_batch_hard_attention(
        self, data_generator: DataGenerator, normalizer: MarketNormalizer
    ) -> None:
        """
        [Research-Grade Test]
        Verifies the 'Kink' batch logic which is critical for the PINN's Hard Attention.
        Condition: t=0 AND S=K (Spot Price equals Strike Price).
        """
        n_samples = 100
        kink_batch = data_generator.get_kink_batch(n_samples)

        # 1. Check t=0 (Column 0)
        t_col = kink_batch[:, 0:1]
        np.testing.assert_allclose(
            t_col, 0.0, atol=1e-9, err_msg="Kink batch must be at maturity (t=0)"
        )

        # 2. Check S=K (Column 1 vs Column 4)
        # Note: We must DENORMALIZE to compare, because S and K have different normalization ranges!
        # S_norm uses S_range [0, 200], K_norm uses K_range [50, 150]

        S_norm = kink_batch[:, 1:2]
        K_norm = kink_batch[:, 4:5]

        # Retrieve ranges manually for manual denorm
        S_min, S_max = normalizer.S_range
        K_min, K_max = normalizer.K_range

        S_real = S_norm * (S_max - S_min) + S_min
        K_real = K_norm * (K_max - K_min) + K_min

        # The logic is: get_kink_batch forces S_real = K_real
        np.testing.assert_allclose(
            S_real, K_real, atol=1e-5, err_msg="Kink batch must enforce S = K"
        )

    def test_boundary_batch_shapes(self, data_generator: DataGenerator) -> None:
        """
        Verifies that BVP (Boundary Value Problem) returns 4 datasets:
        (Lower_X, Lower_Y, Upper_X, Upper_Y)
        """
        n_samples = 20
        bvp_data = data_generator.get_bvp_batch(n_samples)

        # Must return tuple of 4 arrays
        assert len(bvp_data) == 4, "BVP batch must return 4 elements"

        xl, yl, xu, yu = bvp_data

        # Check shapes
        assert xl.shape == (n_samples, 5)
        assert xu.shape == (n_samples, 5)
        assert yl.shape == (n_samples, 1)  # Boundary Value is scalar price
        assert yu.shape == (n_samples, 1)

    def test_discrete_k_sampling(
        self, data_generator: DataGenerator, normalizer: MarketNormalizer
    ) -> None:
        """
        Verifies if Strike Prices (K) align with the specified grid step.
        Config 'K_step' = 10.0
        """
        n_samples = 100
        batch = data_generator.get_pde_batch(n_samples)
        K_norm = batch[:, 4:5]

        # Denormalize K
        K_min, K_max = normalizer.K_range
        K_real = K_norm * (K_max - K_min) + K_min

        # Check divisibility by K_step (10.0)
        # We use modulo. Errors due to float precision should be tiny.
        step = 10.0
        remainder = np.mod(K_real, step)

        # Handle floating point modulo quirks (either close to 0 or close to step)
        min_remainder = np.minimum(remainder, step - remainder)

        np.testing.assert_allclose(
            min_remainder,
            0.0,
            atol=1e-5,
            err_msg="Sampled K is not aligned with K_step grid",
        )
