import numpy as np
import torch
from scipy.stats import norm
from .base_option import OptionPhysics

class PutOption(OptionPhysics):
    """
    Concrete implementation for European Put Options.
    """
    
    def analytical_solution(self, t, S, K, r, sigma):
        """
        Computes the Black-Scholes price for a European Put Option.
        Formula: P = K * e^(-rt) * N(-d2) - S * N(-d1)
        """
        # Ensure numerical stability (avoid division by zero or log of zero)
        t = np.maximum(t, 1e-10)
        S = np.maximum(S, 1e-10)
        K = np.maximum(K, 1e-10)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Put Option Formula
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def payoff(self, S, K):
        """
        Put Option Payoff: max(K - S, 0)
        """
        # Handle both PyTorch Tensors and NumPy arrays
        if isinstance(S, torch.Tensor):
            return torch.maximum(K - S, torch.tensor(0.0, device=S.device))
        return np.maximum(K - S, 0)
    
    def boundary_condition_lower(self, t, r, K):
        """
        Lower Boundary Condition (S -> 0).
        For a Put Option, as Spot Price becomes 0 (Deep ITM), the Option Price approaches the Present Value of K.
        Formula: K * e^(-rt)
        """
        if isinstance(t, torch.Tensor):
            return K * torch.exp(-r * t)
        return K * np.exp(-r * t)

    def boundary_condition_upper(self, t, S_max, K, r):
        """
        Upper Boundary Condition (S -> Infinity).
        For a Put Option, as Spot Price increases (Deep OTM), the Option Price approaches 0.
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        return np.zeros_like(t)