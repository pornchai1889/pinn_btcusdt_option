import sys
import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

# --- Environment Setup ---
# Add the project root to sys.path to ensure imports work correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pinn_net import UniversalPINN
from src.physics.call_option import CallOption

# =============================================================================
# Fixtures (Configuration & Instance Setup)
# =============================================================================
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Standard mock configuration for Model and Physics initialization.
    """
    return {
        'model': {
            'n_input': 5,
            'n_output': 1,
            'n_hidden': 32,      # Small number for fast testing
            'n_layers': 3,
            'hidden_activation': 'tanh',
            'output_activation': 'softplus'
        },
        'market': {
            't_range': [0.0, 1.0],
            'S_range': [0.0, 200.0],
            'K_range': [50.0, 150.0],
            'sigma_range': [0.1, 0.5],
            'r_range': [0.0, 0.1]
        },
        'device': 'cpu'  # Force CPU for unit tests to ensure compatibility
    }

@pytest.fixture
def model(mock_config: Dict[str, Any]) -> UniversalPINN:
    """Creates a fresh instance of the PINN model."""
    return UniversalPINN(mock_config)

@pytest.fixture
def physics_engine(mock_config: Dict[str, Any]) -> CallOption:
    """Creates a physics engine (CallOption) for PDE testing."""
    return CallOption(mock_config)

# =============================================================================
# Test Suite 1: Neural Network Architecture (Smoke Tests)
# =============================================================================
class TestModelArchitecture:
    
    def test_forward_pass_shape(self, model: UniversalPINN) -> None:
        """
        Verifies that the model accepts input of shape (N, 5) 
        and produces output of shape (N, 1).
        """
        batch_size = 16
        # Create dummy 5 input: [t, S, sigma, r, K]
        dummy_input = torch.randn(batch_size, 5) 
        
        # Forward pass
        output = model(dummy_input)
        
        # Assertions
        assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
        assert torch.is_tensor(output), "Output must be a PyTorch Tensor"
        assert not torch.isnan(output).any(), "Model output contains NaNs"

    def test_weight_initialization(self, model: UniversalPINN) -> None:
        """
        Verifies that weights are initialized and not all zeros.
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert torch.std(param) > 0, f"Layer {name} has zero variance (not initialized?)"
                assert not torch.isnan(param).any(), f"Layer {name} contains NaNs"

    def test_activation_config(self, mock_config: Dict[str, Any]) -> None:
        """
        Verifies that changing config actually changes the activation function.
        """
        # Change activation to ReLU
        config_relu = mock_config.copy()
        config_relu['model']['hidden_activation'] = 'relu'
        
        model_relu = UniversalPINN(config_relu)
        
        # Inspect layers to find ReLU
        has_relu = any(isinstance(m, nn.ReLU) for m in model_relu.modules())
        assert has_relu, "Model failed to use ReLU activation from config"

# =============================================================================
# Test Suite 2: Gradient Flow & PDE Residuals (The Critical Tests)
# =============================================================================
class TestGradientFlow:
    
    def test_autograd_enablement(self, model: UniversalPINN) -> None:
        """
        Verifies that the model parameters have gradients enabled.
        Essential for training.
        """
        for param in model.parameters():
            assert param.requires_grad, "Model parameter is frozen (requires_grad=False)"

    def test_first_derivative_computation(self, model: UniversalPINN) -> None:
        """
        Tests if we can compute dV/dX (Gradient w.r.t input) without breaking graph.
        """
        batch_size = 10
        x = torch.randn(batch_size, 5, requires_grad=True)
        
        y = model(x)
        
        # Compute gradient dV/dx
        grads = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True # Important for higher-order derivatives
        )[0]
        
        assert grads.shape == x.shape, "Gradient shape mismatch"
        assert grads.grad_fn is not None, "Gradient computation graph is broken (detached)"
        assert not torch.isnan(grads).any(), "First derivative contains NaNs"

    def test_pde_residual_execution(self, model: UniversalPINN, physics_engine: CallOption) -> None:
        """
        Verifies the full 'compute_pde_residual' pipeline.
        This checks:
        1. Model Forward
        2. Denormalization (differentiable)
        3. First Derivatives (dV/dt, dV/dS)
        4. Second Derivatives (d^2V/dS^2)
        5. PDE Formula combination
        """
        batch_size = 20
        # Inputs must be in normalized range [0, 1] usually
        x_norm = torch.rand(batch_size, 5, requires_grad=True)
        
        # Call the physics engine's PDE function
        pde_residual = physics_engine.compute_pde_residual(model, x_norm)
        
        # Check Output
        assert pde_residual.shape == (batch_size, 1), "PDE residual must be (N, 1)"
        assert torch.is_tensor(pde_residual), "PDE residual must be a tensor"
        
        # Check Graph Connectivity (The most critical check)
        assert pde_residual.grad_fn is not None, "PDE residual lost connection to computational graph!"
        
        # Try a backward pass to ensure no runtime errors in the graph
        loss = torch.mean(pde_residual ** 2)
        try:
            loss.backward()
        except RuntimeError as e:
            pytest.fail(f"Backward pass on PDE residual failed: {e}")

    def test_pde_gradient_nonzero(self, model: UniversalPINN, physics_engine: CallOption) -> None:
        """
        Ensures that optimizing PDE loss actually updates model weights.
        If this fails, the physics loss is 'detached' and won't train the model.
        """
        x_norm = torch.rand(10, 5, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 1. Forward & Loss
        optimizer.zero_grad()
        residual = physics_engine.compute_pde_residual(model, x_norm)
        loss = torch.mean(residual ** 2)
        loss.backward()
        
        # 2. Check if weights have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_grad = True
                break
        
        assert has_grad, "PDE Loss did not generate any gradients for model weights!"