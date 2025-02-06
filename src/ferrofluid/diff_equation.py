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

A = 10**13 * -2.586667896908887
B = 10**13 * 0.252612231449641
C = 10**13 * -0.009638840486198
D = 10**13 * 0.000181276716002
E = 10**13 * -0.000001724024410
F = 10**13 * 0.000000007058447

# Magnetic parameters
Ms = 79.57747* 70000  # Saturation magnetization in Oe
k_B = 1.380649e-23  # Boltzmann constant
T = 293  # Room temperature in Kelvin
mu_0 = 4*np.pi*1e-7  # Vacuum permeability
m = 1e-19  # Magnetic moment of particle (this is an estimate, should be from paper)

def langevin(x):
    """Langevin function L(x) = coth(x) - 1/x"""
    # Add small epsilon to prevent division by zero
    eps = 1e-10
    x = x + eps
    return 1/torch.tanh(x) - 1/x

def magnetization(x):
    """M(H) relationship using Langevin function"""
    H = magnetic_field(x)
    alpha = (mu_0 * m * H)/(k_B * T)
    return Ms * langevin(alpha)

def magnetic_field(x):
    """H(x) relationship using polynomial"""
    return 5*A*(x**4) + 4*B*(x**3) + 3*C*(x**2) + 2*D*x + E

def magnetization_deriv(x):
    """dM/dx using chain rule"""
    H = magnetic_field(x)
    alpha = (mu_0 * m * H)/(k_B * T)
    return Ms * (langevin(alpha) - 1/alpha) * (mu_0 * m)/(k_B * T)
