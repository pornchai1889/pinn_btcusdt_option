# src/physics/call_option.py
import numpy as np
import torch
from scipy.stats import norm
from .base_option import OptionPhysics

class CallOption(OptionPhysics):
    """
    Concrete implementation for European Call Options.
    """
    
    def analytical_solution(self, t, S, K, r, sigma):
        """
        Computes the Black-Scholes price for a European Call Option.
        Formula: C = S * N(d1) - K * e^(-rt) * N(d2)
        """
        # Ensure numerical stability (avoid division by zero or log of zero)
        t = np.maximum(t, 1e-10)
        S = np.maximum(S, 1e-10)
        K = np.maximum(K, 1e-10)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        return price

    def payoff(self, S, K):
        """
        Call Option Payoff: max(S - K, 0)
        """
        # Handle both PyTorch Tensors and NumPy arrays
        if isinstance(S, torch.Tensor):
            return torch.maximum(S - K, torch.tensor(0.0, device=S.device))
        return np.maximum(S - K, 0)
    
    def boundary_condition_lower(self, t, r, K):
        """
        Lower Boundary Condition (S -> Minimum).
        For a Call Option, as Spot Price becomes Minimum, the Option Price becomes 0.
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        return np.zeros_like(t) # Return a zero-filled array (or list) of length t.

    def boundary_condition_upper(self, t, S_max, K, r):
        """
        Upper Boundary Condition (S -> Maximum).
        For a Call Option, as Spot Price increases (Maximum), Price -> S - K * exp(-rt).
        """
        if isinstance(t, torch.Tensor):
            return S_max - K * torch.exp(-r * t)
        return S_max - K * np.exp(-r * t)