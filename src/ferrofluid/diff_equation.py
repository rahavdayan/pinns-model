import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# H(x) polynomial coefficients, x is measured in m and H in A/m
a_i = [
    (10**13) * -2.586667896908887,
    (10**13) * 0.252612231449641,
    (10**13) * -0.009638840486198,
    (10**13) * 0.000181276716002,
    (10**13) * -0.000001724024410,
    (10**13) * 0.000000007058447
]
n = len(a_i)                        # Number of data points

# Different variables for problem
r = 0.003                           # Radius of droplet in [m]
V = (4/3)*np.pi*(r**3)              # Volume of droplet in [m^3]
mu_0 = 1.256637*(10**-6)            # Permeability of free space [m*kg/(s*A)]
eta = 50                            # Viscosity in [Pa*s]
M_d = 4.46*1e5                      # Domain magnetization of the particles [A/m]
x_0 = 1000                          # x scaling factor [m]
d = 76*1e-9                         # mean diameter of nanoparticles [m]
k_B = 1.3806452*(10**-23)           # Boltzmann constant [m^2*Kg*s^-2*K^-1]
T = 293                             # Absolute temperature [K]
phi = 1/4                           # volume fraction of magnetic nanoparticles (25% mentioned somwhere in background paper)

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

# def position_2mm_droplet():
#     t_train = np.array([21.3874, 114.554, 371.812, 551.812, 791.812, 1190.91])
#     x_train = np.array([11.6842, 16.3119, 19.8543, 21.0027, 22.047, 23.2759])
#     return t_train, x_train, np.log(t_train), x_train / x_0
#     # list of time values in s, then position values in mm

def position_3mm_droplet():
    t_train = np.array([31.9143, 164.784, 396.852, 636.852, 876.852, 1116.85])
    x_train = np.array([14.9303, 19.8436, 22.3973, 23.8782, 25.0633, 26.1438])
    return t_train, x_train, np.log(t_train), x_train / x_0
    # list of time values in s, then position values in mm

def H(x):
    sum = 0
    for i in range(0, n):
        sum += a_i[i]*(x ** i)
    return sum

def dH_dx(x):
    sum = 0
    for i in range(1, n):
        sum += i*a_i[i]*(x ** (i-1))
    return sum

def dx_dt_nondim(t, x):
    xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(x_0*x)
    return ((V*mu_0*phi*M_d*torch.exp(t))/(6*np.pi*r*eta*x_0)) * (1/torch.tanh(xi) - 1/xi) * dH_dx(x_0*x)

def physics_loss(model: torch.nn.Module):
    ts = torch.linspace(3.35, np.log(1200), steps=1200,).view(-1, 1).requires_grad_(True).to(DEVICE)
    xs = model(ts)
    dx = grad(xs, ts)[0]
    pde = dx_dt_nondim(ts, xs) - dx
    
    return torch.mean(pde**2)