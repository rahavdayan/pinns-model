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

A = 1e13 * -2.586667896908887
B = 1e13 * 0.252612231449641
C = 1e13 * -0.009638840486198
D = 1e13 * 0.000181276716002
E = 1e13 * -0.000001724024410
F = 1e13 * 0.000000007058447

# Magnetic parameters
k_B = 1.380649e-23  # Boltzmann constant
T = 293  # Room temperature in Kelvin
mu_0 = 4*np.pi*1e-7  # Vacuum permeability
d = 76*1e-9  # diameter of magnetic nanoparticle (googled!)
M_d = 4.46*1e5  # Magnetic moment of particle (this is an estimate, should be from paper)
phi = 1/4   # volume fraction of magnetic nanoparticles (25% mentioned somwhere in background paper)

# Droplet parameters
r = 0.002                           # Radius of droplet in [m]
V = (4/3)*np.pi*(r**3)              # Volume of droplet in [m^3]


def langevin(x):
    """Langevin function L(x) = coth(x) - 1/x"""
    # Add small epsilon to prevent division by zero
    eps = 1e-30
    x = x + eps

    return 1/torch.tanh(x) - 1/x

def magnetization(x):
    """M(H) relationship using Langevin function"""
    H = magnetic_field(x)
    alpha = (mu_0 * M_d* d**3* np.pi/6 * H)/(k_B * T)
    alpha_tensor = torch.tensor(alpha)
    return M_d* phi * langevin(alpha_tensor)

def magnetic_field(x):
    """H(x) relationship using polynomial"""
    return 5*A*(x**4) + 4*B*(x**3) + 3*C*(x**2) + 2*D*x + E

def magnetic_field_deriv(x):
    """dH/dx is polynomial"""
    return A*(x**5) + B*(x**4) + C*(x**3) + D*(x**2) + E*x + F

