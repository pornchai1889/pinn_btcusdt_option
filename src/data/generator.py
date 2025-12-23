# src/data/generator.py
import numpy as np
from .normalizer import MarketNormalizer
from typing import Any, Dict, Tuple


class DataGenerator:
    """
    Responsible for generating training and validation data batches.
    Implements advanced sampling strategies:
    1. Mixed Moneyness Distribution (Gaussian + Uniform Outliers)
    2. Power Law Time Sampling (Focus on t=0)
    3. Discrete Strike Price (K) Sampling

    Updated to support both Call and Put options via Physics Engine delegation.
    """

    def __init__(
        self, config: Dict[str, Any], normalizer: MarketNormalizer, physics_engine: Any
    ):
        self.config = config
        self.norm = normalizer
        self.physics = (
            physics_engine  # Injected Physics Engine (CallOption or PutOption)
        )

        # Unpack Market Parameters
        market = config["market"]
        self.t_min, self.t_max = market["t_range"]
        self.S_min, self.S_max = market["S_range"]
        self.K_min, self.K_max = market["K_range"]
        self.sig_min, self.sig_max = market["sigma_range"]
        self.r_min, self.r_max = market["r_range"]
        self.K_step = market.get("K_step", None)

        # Unpack Sampling Strategy Parameters
        sampling = config["sampling"]
        self.m_min, self.m_max = sampling["moneyness_range"]
        self.time_power = sampling.get("time_sampling_power", 2.0)

        # Calculate Real Sigma for Moneyness Sampling (Adaptive STD Logic)
        adaptive_factor = sampling.get("adaptive_std_factor", 1.0)
        range_width = self.m_max - self.m_min
        self.sampling_std = (range_width / 6.0) * adaptive_factor

    def _sample_moneyness_mixed(self, n: int) -> np.ndarray:
        """
        Internal Method: Mixed Distribution Strategy.
        1. Sample from Gaussian (centered at 1.0).
        2. Identify outliers outside [m_min, m_max].
        3. Re-sample outliers using Uniform Distribution to fill gaps.
        """
        # 1. Generate Raw Gaussian
        data = np.random.normal(1.0, self.sampling_std, (n, 1))

        # 2. Identify Outliers
        flat_data = data.flatten()
        outliers_mask = (flat_data < self.m_min) | (flat_data > self.m_max)
        n_out = np.sum(outliers_mask)

        # 3. Resample Outliers (Uniformly)
        if n_out > 0:
            flat_data[outliers_mask] = np.random.uniform(self.m_min, self.m_max, n_out)

        return flat_data.reshape(n, 1)

    def _get_discrete_K(self, n: int) -> np.ndarray:
        """
        Internal Method: Samples Strike Prices (K).
        Supports both Continuous and Discrete (Grid-based) sampling.
        """
        if self.K_step is None or self.K_step <= 0:
            return np.random.uniform(self.K_min, self.K_max, (n, 1))

        # Align min/max to grid
        aligned_min = np.ceil(self.K_min / self.K_step) * self.K_step
        aligned_max = np.floor(self.K_max / self.K_step) * self.K_step

        if aligned_max < aligned_min:
            # Fallback if range is too small for step
            return np.random.uniform(self.K_min, self.K_max, (n, 1))

        n_steps = int((aligned_max - aligned_min) / self.K_step)
        random_steps = np.random.randint(0, n_steps + 1, (n, 1))

        return aligned_min + random_steps * self.K_step

    def _sample_time_power_law(self, n: int) -> np.ndarray:
        """
        Internal Method: Power Law Sampling for Time.
        Focuses more points near t=0 (Maturity) where curvature is high.
        formula: t = t_min + (t_max - t_min) * u^p
        """
        u = np.random.uniform(0, 1, (n, 1))
        return self.t_min + (self.t_max - self.t_min) * (u**self.time_power)

    def get_pde_batch(self, n: int) -> np.ndarray:
        """
        Generates Collocation Points for PDE Residual Loss (Interior Points).
        Returns: Normalized Inputs (np.array)
        """
        # 1. Sample Inputs
        K = self._get_discrete_K(n)
        moneyness = self._sample_moneyness_mixed(n)
        S = np.clip(
            K * moneyness, self.S_min, self.S_max
        )  # Derive S from K & Moneyness
        t = self._sample_time_power_law(n)
        sigma = np.random.uniform(self.sig_min, self.sig_max, (n, 1))
        r = np.random.uniform(self.r_min, self.r_max, (n, 1))

        # 2. Normalize and Return
        return self.norm.normalize_batch(t, S, sigma, r, K)

    def get_ivp_batch(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates Initial Value Problem Data (t=0).
        Delegates payoff calculation to the Physics Engine (Call or Put).
        Returns: (X_norm, y_norm)
        """
        # t is strictly 0
        t = np.zeros((n, 1))

        K = self._get_discrete_K(n)
        moneyness = self._sample_moneyness_mixed(n)
        S = np.clip(K * moneyness, self.S_min, self.S_max)
        sigma = np.random.uniform(self.sig_min, self.sig_max, (n, 1))
        r = np.random.uniform(self.r_min, self.r_max, (n, 1))

        # Normalize Inputs
        X_norm = self.norm.normalize_batch(t, S, sigma, r, K)

        # Calculate Targets (Payoff) dynamically via Physics Engine
        payoff = self.physics.payoff(S, K)
        y_norm = self.norm.normalize_price(payoff, K)

        # Explicitly cast to numpy array to resolve Mypy ambiguity (Union[Tensor, ndarray])
        return X_norm, np.asarray(y_norm)

    def get_bvp_batch(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates Boundary Value Problem Data (Lower & Upper Bounds of S).
        Delegates boundary logic to the Physics Engine (Call or Put).
        Returns: (X_lower, y_lower, X_upper, y_upper)
        """
        # Shared randoms
        t = self._sample_time_power_law(n)
        sigma = np.random.uniform(self.sig_min, self.sig_max, (n, 1))
        r = np.random.uniform(self.r_min, self.r_max, (n, 1))
        K = self._get_discrete_K(n)

        # --- Lower Boundary (S -> S_min, usually approx 0) ---
        S_lower = np.clip(K * self.m_min, self.S_min, self.S_max)
        X_lower_norm = self.norm.normalize_batch(t, S_lower, sigma, r, K)

        # Calculate Lower Boundary Value dynamically
        y_lower_val = self.physics.boundary_condition_lower(t, r, K)
        y_lower_norm = self.norm.normalize_price(y_lower_val, K)

        # --- Upper Boundary (S -> S_max) ---
        S_upper = np.clip(K * self.m_max, self.S_min, self.S_max)
        X_upper_norm = self.norm.normalize_batch(t, S_upper, sigma, r, K)

        # Calculate Upper Boundary Value dynamically
        y_upper_val = self.physics.boundary_condition_upper(t, S_upper, K, r)
        y_upper_norm = self.norm.normalize_price(y_upper_val, K)

        # Explicitly cast all outputs to numpy arrays to ensure return type consistency
        return (
            X_lower_norm,
            np.asarray(y_lower_norm),
            X_upper_norm,
            np.asarray(y_upper_norm),
        )

    def get_kink_batch(self, n_samples: int) -> np.ndarray:
        """
        Generates a specific batch where S = K and t = 0 (The Kink).
        Used for Hard Attention mechanism in training.
        """
        # 1. Reuse IVP logic to get valid distributions for sigma, r, K
        # We don't care about the S and t returned here, just the params
        raw_ivp = self.get_ivp_batch(n_samples)[
            0
        ]  # Return only [t, S, sig, r, K] values
        # At t = 0, S = K: Call Option: V = max(S - K, 0) = max(0, 0) = 0
        # At t = 0, S = K: Put Option: V = max(K - S, 0) = max(0, 0) = 0

        # 2. Extract ranges from self.norm (NormalizationHandler)
        # Note: Assuming self.norm stores the config ranges
        S_min, S_max = self.norm.S_range
        K_min, K_max = self.norm.K_range

        # 3. Process to force S = K
        # Extract Normalized K (index 4)
        K_norm = raw_ivp[:, 4]

        # Decode K_norm -> K_real
        K_real = K_norm * (K_max - K_min) + K_min

        # Set S_real = K_real
        S_real_target = K_real

        # Encode S_real -> S_norm
        S_norm_target = (S_real_target - S_min) / (S_max - S_min)

        # 4. Construct Kink Batch
        kink_batch = raw_ivp.copy()
        kink_batch[:, 0] = 0.0  # Force t=0 (Maturity)
        kink_batch[:, 1] = S_norm_target  # Force S=K (At The Money)

        return kink_batch

    def get_validation_batch(self, n: int) -> np.ndarray:
        """
        Generates a generic batch for validation/testing.
        """
        return self.get_pde_batch(n)
