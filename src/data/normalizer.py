# src/data/normalizer.py
import numpy as np
import torch
from typing import Union, Dict, Any

class MarketNormalizer:
    """
    Handles all normalization and denormalization logic for the PINN model.
    Scaling strategy: Min-Max Normalization to range [0, 1].
    """
    def __init__(self, config : Dict[str, Any]):
        self.config = config
        market_params = config['market']
        
        # Cache boundaries for performance
        self.t_range = market_params['t_range']
        self.S_range = market_params['S_range']
        self.K_range = market_params['K_range']
        self.sigma_range = market_params['sigma_range']
        self.r_range = market_params['r_range']
        
    def _normalize(self, val: Union[np.ndarray, torch.Tensor, float], v_range: list[float]) -> Union[np.ndarray, torch.Tensor, float]:
        """Helper: Linear scaling to [0, 1]"""
        return (val - v_range[0]) / (v_range[1] - v_range[0])

    def _denormalize(self, val: Union[np.ndarray, torch.Tensor, float], v_range: list[float]) -> Union[np.ndarray, torch.Tensor, float]:
        """Helper: Inverse scaling from [0, 1] to original domain"""
        return val * (v_range[1] - v_range[0]) + v_range[0]

    def normalize_batch(self, t: np.ndarray, S: np.ndarray, sigma: np.ndarray, r: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Normalizes a batch of physical inputs.
        Returns: Numpy array of shape (N, 5) -> [t, S, sigma, r, K]
        """
        t_norm = self._normalize(t, self.t_range)
        S_norm = self._normalize(S, self.S_range)
        sig_norm = self._normalize(sigma, self.sigma_range)
        r_norm = self._normalize(r, self.r_range)
        K_norm = self._normalize(K, self.K_range)
        
        # Stack into a single matrix for model input
        return np.concatenate([t_norm, S_norm, sig_norm, r_norm, K_norm], axis=1)

    def denormalize_batch(self, x_norm: Union[np.ndarray, torch.Tensor]) -> tuple[Union[np.ndarray, torch.Tensor], ...]:
        """
        Denormalizes a batch of model inputs back to physical values.
        Supports both Numpy arrays and PyTorch Tensors.
        """
        is_tensor = torch.is_tensor(x_norm)
        
        # Helper wrapper to handle both types
        def denorm_fn(data, rng):
            if is_tensor:
                return data * (rng[1] - rng[0]) + rng[0]
            return self._denormalize(data, rng)

        # Slice columns (assuming order: t, S, sigma, r, K)
        t = denorm_fn(x_norm[:, 0:1], self.t_range)
        S = denorm_fn(x_norm[:, 1:2], self.S_range)
        sigma = denorm_fn(x_norm[:, 2:3], self.sigma_range)
        r = denorm_fn(x_norm[:, 3:4], self.r_range)
        K = denorm_fn(x_norm[:, 4:5], self.K_range)
        
        return t, S, sigma, r, K

    def normalize_price(self, price: Union[np.ndarray, torch.Tensor], K: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalizes the Option Price.
        Standard practice: V_norm = V / K
        """
        return price / (K + 1e-8) # Add epsilon to avoid div by zero

    def denormalize_price(self, price_norm: Union[np.ndarray, torch.Tensor], K: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalizes the Option Price.
        V = V_norm * K
        """
        return price_norm * K