import torch
import numpy as np


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

# there is no analytic solution for the ferrofluid diffeq dx/dt = f(x), so this is just a placeholder function that returns t
def position(t):
    return t

def magnetic_field(x):
    return 1

def magnetic_field_deriv(x):
    return 1

def magnetization(x):
    return 1