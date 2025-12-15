import torch
import torch.nn as nn

class UniversalPINN(nn.Module):
    """
    Universal Physics-Informed Neural Network (PINN) Architecture.
    A fully connected Deep Neural Network designed to approximate Option Pricing functions.
    
    Architecture:
        Input -> [Linear -> Tanh] x N_Layers -> Linear -> Softplus -> Output
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing model architecture parameters.
                           Expected keys: model['n_input'], model['n_output'], 
                           model['n_hidden'], model['n_layers']
        """
        super(UniversalPINN, self).__init__()
        
        # Extract architecture parameters from config
        model_conf = config['model']
        n_input = model_conf['n_input']
        n_output = model_conf['n_output']
        n_hidden = model_conf['n_hidden']
        n_layers = model_conf['n_layers']
        
        # Build the network
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.Tanh())
        
        # Hidden Layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
            
        # Output Layer
        layers.append(nn.Linear(n_hidden, n_output))
        
        # Enforce positive output (Option prices cannot be negative)
        layers.append(nn.Softplus())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Xavier Normal initialization to Linear layers 
        and sets biases to zero, following standard practices for PINNs.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Normalized input tensor of shape (Batch, n_input)
            
        Returns:
            torch.Tensor: Normalized predicted option price (Batch, n_output)
        """
        return self.net(x)