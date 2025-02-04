import torch
import numpy as np

# H(x) polynomial coefficients
A = (10**13) * -2.586667896908887
B = (10**13) * 0.252612231449641
C = (10**13) * -0.009638840486198
D = (10**13) * 0.000181276716002
E = (10**13) * -0.000001724024410
F = (10**13) * 0.000000007058447

# magnetization/magnetic field scalar
c = 1

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

def position_2mm_droplet():
    return np.array([21.3874, 114.554, 371.812, 551.812, 791.812, 1190.91]), np.array([11.6842, 16.3119, 19.8543, 21.0027, 22.047, 23.2759])
    # list of time values, then position values

def magnetic_field(x):
    return A*(x**5) + B*(x**4) + C*(x**3) + D*(x**2) + E*x + F

def magnetic_field_deriv(x):
    return 5*A*(x**4) + 4*B*(x**3) + 3*C*(x**2) + 2*D*x + E

def magnetization(x):
    return c*magnetic_field(x)