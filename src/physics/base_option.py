# src/physics/base_option.py
import torch
from abc import ABC, abstractmethod

class OptionPhysics(ABC):
    """
    Abstract Base Class for Option Pricing Physics.
    Defines the standard interface and shared PDE logic for financial derivatives.
    """
    def __init__(self, config):
        self.config = config
        self.market_config = config['market']

    @abstractmethod
    def analytical_solution(self, t, S, K, r, sigma):
        """
        Calculates the exact price using the analytical formula (e.g., Black-Scholes).
        Used for validation and benchmarking.
        """
        pass

    @abstractmethod
    def payoff(self, S, K):
        """
        Calculates the payoff at maturity (t=0).
        Corresponds to the Initial Value Problem (IVP) condition.
        """
        pass
    
    @abstractmethod
    def boundary_condition_lower(self, t, r, K):
        """
        Calculates the value when S approaches 0 (Lower Boundary).
        """
        pass

    @abstractmethod
    def boundary_condition_upper(self, t, S_max, K, r):
        """
        Calculates the value when S approaches Infinity (Upper Boundary).
        """
        pass

    def compute_pde_residual(self, model, x_pde):
        """
        Computes the Black-Scholes PDE residual (physics loss).
        Equation: dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0
        
        Args:
            model: The Neural Network model.
            x_pde: Normalized input tensor [Batch, 5] (t, S, sigma, r, K).
            
        Returns:
            pde_residual: The residual error from the physics equation.
        """
        # 1. Enable Gradient Tracking for Automatic Differentiation
        x_pde.requires_grad = True
        
        # 2. Forward Pass
        # The model outputs Normalized Option Price (V/K)
        v_pred_norm = model(x_pde)
        
        # 3. Denormalize Inputs (Restore to physical units for the equation)
        # x_pde columns: [0]t, [1]S, [2]sigma, [3]r, [4]K
        t_min, t_max = self.market_config['t_range']
        S_min, S_max = self.market_config['S_range']
        K_min, K_max = self.market_config['K_range']
        sig_min, sig_max = self.market_config['sigma_range']
        r_min, r_max = self.market_config['r_range']
        
        # Helper lambda for denormalization inside the computational graph
        denorm = lambda val_norm, v_min, v_max: val_norm * (v_max - v_min) + v_min

        S_pde = denorm(x_pde[:, 1:2], S_min, S_max)
        sigma_pde = denorm(x_pde[:, 2:3], sig_min, sig_max)
        r_pde = denorm(x_pde[:, 3:4], r_min, r_max)
        K_pde = denorm(x_pde[:, 4:5], K_min, K_max)
        
        # Restore Real Option Price: V = ModelOutput * K
        V_real = v_pred_norm * K_pde

        # 4. Compute Gradients using Autograd
        # First derivatives
        grads = torch.autograd.grad(
            v_pred_norm, x_pde, 
            grad_outputs=torch.ones_like(v_pred_norm), 
            create_graph=True
        )[0]
        
        dv_dt_norm = grads[:, 0:1]
        dv_ds_norm = grads[:, 1:2]
        
        # Second derivative (d^2V/dS^2)
        grads2 = torch.autograd.grad(
            dv_ds_norm, x_pde, 
            grad_outputs=torch.ones_like(dv_ds_norm), 
            create_graph=True
        )[0]
        d2v_ds2_norm = grads2[:, 1:2]
        
        # 5. Chain Rule: Scale Gradients to Physical Units
        # dV/dt = (dV_norm/dt_norm) * (dt_norm/dt) * K
        # dt_norm/dt = 1 / (t_max - t_min)
        dV_dt = (K_pde / (t_max - t_min)) * dv_dt_norm
        
        # dV/dS = (dV_norm/dS_norm) * (dS_norm/dS) * K
        # dS_norm/dS = 1 / (S_max - S_min)
        dV_dS = (K_pde / (S_max - S_min)) * dv_ds_norm
        
        # d^2V/dS^2
        d2V_dS2 = (K_pde / (S_max - S_min)**2) * d2v_ds2_norm

        # 6. Calculate PDE Residual
        # Black-Scholes PDE: dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0
        # Note: Sign conventions for time (t) may vary based on Time-to-Maturity definition.
        # Here we assume t is time-to-maturity (tau), so the equation typically has a minus sign or arrangement change.
        # Based on your previous code: dV/dt - (rV - rS... - ...)
        
        pde_res = dV_dt - (0.5 * sigma_pde**2 * S_pde**2 * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_real)
        
        # Normalize residual by K for numerical stability during training
        return pde_res / K_pde