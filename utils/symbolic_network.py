import numpy as np
import torch
import torch.nn as nn
from utils import functions as functions


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""
    def __init__(self, funcs=None, initial_weight=None, init_stddev=0.1, in_dim=None):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()

        if funcs is None:
            funcs = functions.default_func
        self.initial_weight = initial_weight
        self.W = None       # Weight matrix
        self.built = False  # Boolean whether weights have been initialized

        self.output = None  # tensor for layer output
        self.n_funcs = len(funcs)                       # Number of activation functions (and number of layer outputs)
        self.funcs = [func.torch for func in funcs]     # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)   # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double    # Number of activation functions that take 1 input

        self.out_dim = self.n_funcs + self.n_double

        if self.initial_weight is not None:     # use the given initial weight
            self.W = nn.Parameter(self.initial_weight.clone().detach())  # copies
            self.built = True
        else:
            self.W = torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim))

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""

        g = torch.matmul(x, self.W)         # shape = (?, self.size)
        self.output = []

        in_i = 0    # input index
        out_i = 0   # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            self.output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            self.output.append(self.funcs[out_i](g[:, in_i], g[:, in_i+1]))
            in_i += 2
            out_i += 1

        self.output = torch.stack(self.output, dim=1)

        return self.output

    def get_weight(self):
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SymbolicNet(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, init_stddev=0.1):
        super(SymbolicNet, self).__init__()

        self.depth = symbolic_depth     # Number of hidden layers
        self.funcs = funcs
        layer_in_dim = [1] + self.depth*[len(funcs)]

        if initial_weights is not None:
            layers = [SymbolicLayer(funcs=funcs, initial_weight=initial_weights[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())

        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayer(funcs=self.funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]

            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand((layers[-1].n_funcs, 1)))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        h = self.hidden_layers(input)   # Building hidden layers
        return torch.matmul(h, self.output_weight)  # Final output (no activation units) of network

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
               [self.output_weight.clone()]



