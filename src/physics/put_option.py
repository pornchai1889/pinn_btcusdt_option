# src/physics/put_option.py
import numpy as np
import torch
from scipy.stats import norm
from .base_option import OptionPhysics
from typing import Union


class PutOption(OptionPhysics):
    """
    Concrete implementation for European Put Options.
    """

    def analytical_solution(
        self,
        t: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the Black-Scholes price for a European Put Option.
        Formula: P = K * e^(-rt) * N(-d2) - S * N(-d1)
        """
        # Ensure numerical stability (avoid division by zero or log of zero)
        t = np.maximum(t, 1e-10)
        S = np.maximum(S, 1e-10)
        K = np.maximum(K, 1e-10)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        # Put Option Formula
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def payoff(
        self, S: Union[torch.Tensor, np.ndarray], K: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Put Option Payoff: max(K - S, 0)
        """
        # Handle both PyTorch Tensors and NumPy arrays
        if isinstance(S, torch.Tensor):
            # Explicitly cast K to Tensor to satisfy Mypy strict type checking for torch.maximum
            K_t = torch.as_tensor(K, device=S.device, dtype=S.dtype)
            zero_t = torch.tensor(0.0, device=S.device, dtype=S.dtype)
            return torch.maximum(K_t - S, zero_t)

        return np.maximum(K - S, 0)

    def boundary_condition_lower(
        self,
        t: Union[torch.Tensor, np.ndarray],
        r: Union[torch.Tensor, np.ndarray],
        K: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Lower Boundary Condition (S -> Minimum).
        Formula: K * e^(-rt)
        """
        if isinstance(t, torch.Tensor):
            # Explicitly cast operands to ensure PyTorch operations satisfy Mypy
            r_t = torch.as_tensor(r, device=t.device, dtype=t.dtype)
            K_t = torch.as_tensor(K, device=t.device, dtype=t.dtype)
            return K_t * torch.exp(-r_t * t)

        return K * np.exp(-r * t)

    def boundary_condition_upper(
        self,
        t: Union[np.ndarray, torch.Tensor],
        S_max: Union[np.ndarray, torch.Tensor, float],
        K: Union[np.ndarray, torch.Tensor],
        r: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Upper Boundary Condition (S -> Maximum (Infinity in theoy)).
        For a Put Option, as Spot Price increases (Deep OTM), the Option Price approaches 0.
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        return np.zeros_like(t)  # Return a zero-filled array (or list) of length t.
