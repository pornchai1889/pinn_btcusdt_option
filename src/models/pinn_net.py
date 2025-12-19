# src/models/pinn_net.py
import torch.nn as nn
import logging

class UniversalPINN(nn.Module):
    def __init__(self, config):
        super(UniversalPINN, self).__init__()
        
        # Extract params
        model_conf = config['model']
        n_input = model_conf['n_input']
        n_output = model_conf['n_output']
        n_hidden = model_conf['n_hidden']
        n_layers = model_conf['n_layers']
        
        # 1. Activation Registry
        activations = {
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'softplus': nn.Softplus,
            'identity': nn.Identity  # no activation function
        }

        # 2. Get Hidden Activation
        act_name = model_conf.get('hidden_activation', 'tanh').lower()
        HiddenAct = activations.get(act_name, nn.Tanh)
        logging.info(f"Model Hidden Activation: {HiddenAct.__name__}")

        # 3. Get Output Activation
        out_act_name = model_conf.get('output_activation', 'softplus').lower()
        OutputAct = activations.get(out_act_name, nn.Softplus)
        logging.info(f"Model Output Activation: {OutputAct.__name__}")

        # 4. Build Network
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(HiddenAct())
        
        # Hidden Layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(HiddenAct())
            
        # Output Layer
        layers.append(nn.Linear(n_hidden, n_output))
        layers.append(OutputAct())
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)