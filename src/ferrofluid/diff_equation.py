import torch
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# H(x) polynomial coefficients, x is measured in m and H in A/m
a_i = [
    (10**13) * -2.586667896908887, # [A/m^6]
    (10**13) * 0.252612231449641,  # [A/m^5]
    (10**13) * -0.009638840486198, # [A/m^4]
    (10**13) * 0.000181276716002,  # [A/m^3]
    (10**13) * -0.000001724024410, # [A/m^2]
    (10**13) * 0.000000007058447   # [A/m]
]
n = len(a_i)                        # Number of data points

# Different variables for problem
r = torch.tensor([0.0005, 0.001, 0.0015, 0.002])  # Radius of droplet in [m]
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

def grab_training_data():
    droplet_files_names = ['droplet_1mm.csv', 'droplet_2mm.csv', 'droplet_3mm.csv', 'droplet_4mm.csv']
    droplet_files_list = [None] * len(droplet_files_names)
    col_names_time = [None] * len(droplet_files_names)
    col_names_dist = [None] * len(droplet_files_names)
    max_len = 0

    for i in range (0, len(droplet_files_names)):
        droplet_files_list[i] = pd.read_csv('./droplet_data/' + droplet_files_names[i])
        col_names_time[i] = f'TIME_{i+1}MM'
        col_names_dist[i] = f'TIME_{i+1}MM'
        if len(droplet_files_list[i]) > max_len:
            max_len = len(droplet_files_list[i])
    
    train_t = pd.DataFrame(index=range(max_len), columns=col_names_time)
    train_x = pd.DataFrame(index=range(max_len), columns=col_names_dist)

    for i in range (0, len(droplet_files_list)):
        train_t.loc[i] = droplet_files_list[i][col_names_time[i]]
        train_x.loc[i] = droplet_files_list[i][col_names_dist[i]]
    
    return train_t, train_x

# def position_3mm_droplet():
#     t_train = np.array([31.9143, 164.784, 396.852, 636.852, 876.852, 1116.85])
#     x_train = np.array([14.9303, 19.8436, 22.3973, 23.8782, 25.0633, 26.1438])
#     return t_train, x_train, np.log(t_train), x_train / x_0
#     # list of time values in s, then position values in mm

def H(x):
    sum = 0
    for i in range(n-1, -1, -1):
        sum += a_i[i]*(x ** i)
    return sum

def dH_dx(x):
    sum = 0
    for i in range(n-1, 0, -1):
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

def physics_loss_dimensional(model: torch.nn.Module):
    ts = torch.linspace(0, 1200, steps=1200,).view(-1, 1).requires_grad_(True).to(DEVICE)
    xs = model(ts)
    dx = grad(xs, ts)[0]
    xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(xs)
    pde = (V*mu_0*phi*M_d*dH_dx(xs)*(1/torch.tanh(xi) - 1/xi))/(6*r*np.pi*eta) - dx
    
    return torch.mean(pde**2)
