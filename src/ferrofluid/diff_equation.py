import torch
import numpy as np
import pandas as pd
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use the magnetic field data to create an exponential fit
def exponential_fit():
    # Load the CSV data
    with open('./droplet_data/magnetic_field_data.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader), dtype=float)
    
    # Extract x and y values
    x_target, y_target = data[:, 0], data[:, 1]

    # Log-transform y values (base 20)
    log_y_target = np.log(y_target)

    # Construct the linear system Ax = b
    A = np.vstack([x_target, np.ones(len(x_target))]).T
    m, c = np.linalg.lstsq(A, log_y_target, rcond=None)[0]

    # Define the exponential fit function
    def f(x):
        return np.e ** (c + m * x)
    
    # Define its derivative
    def f_deriv(x):
        return m*np.e ** (c + m * x)
    
    return f, f_deriv

# H(x) polynomial coefficients, x is measured in m and H in A/m
a_i = [
    (10**13) * -2.586667896908887,
    (10**13) * 0.252612231449641,
    (10**13) * -0.009638840486198,
    (10**13) * 0.000181276716002,
    (10**13) * -0.000001724024410,
    (10**13) * 0.000000007058447
]

# Different variables for problem
n = len(a_i)                                                    # Number of data points
r = torch.tensor([0.0005, 0.001, 0.0015, 0.002]).to(DEVICE)     # List of radii of droplet sizes in [m]
V = (4/3)*torch.pi*(r**3).to(DEVICE)                            # List of volumes of droplet sizes in [m^3]
mu_0 = 1.256637*(10**-6)                                        # Permeability of free space [m*kg/(s*A)]
eta = 50                                                        # Viscosity in [Pa*s]
M_d = 4.46*1e5                                                  # Domain magnetization of the particles [A/m]
x_0 = 1000                                                      # x scaling factor [m]
d = 76*1e-9                                                     # mean diameter of nanoparticles [m]
k_B = 1.3806452*(10**-23)                                       # Boltzmann constant [m^2*Kg*s^-2*K^-1]
T = 293                                                         # Absolute temperature [K]
phi = 1/4                                                       # volume fraction of magnetic nanoparticles (25%, mentioned somwhere in background paper)
exp, exp_deriv = exponential_fit()                              # Exponential fit to the magnetic field data and its derivtive

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
    # get training data csvs
    droplet_files_names = ['droplet_1mm.csv', 'droplet_2mm.csv', 'droplet_3mm.csv', 'droplet_4mm.csv']
    dim_data = [None] * len(droplet_files_names)
    nondim_data = [None] * len(droplet_files_names)
    
    # store csv for each size droplet into an array, one for dimensionalized data and one for nondimensionalized
    for i in range (0, len(droplet_files_names)):
        dim_data[i] = pd.read_csv('./droplet_data/' + droplet_files_names[i])
        
        # Grab every tenth row
        dim_data[i] = dim_data[i][dim_data[i].index % 10 == 0]
        
        # exclude first row because of issues with (0, 0)
        dim_data[i] = dim_data[i].loc[1:]

        # perform nondimensionalization on nondimensionalized copy
        nondim_data[i] = dim_data[i].copy()
        nondim_data[i]['TIME'] = np.log(nondim_data[i]['TIME'])
        nondim_data[i]['DISTANCE'] = nondim_data[i]['DISTANCE'] / x_0
    
    return dim_data, nondim_data

# Langevin function
def L(xi):
    return 1/torch.tanh(xi) - 1/xi

# Magnetization
def M(x):
    xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(x)
    return phi*M_d*L(xi)

# Magnetic field
def H(x):
    out = torch.zeros_like(x).to(DEVICE)
    
    # Mask for elements where x < 20
    mask_poly = x < 20
    mask_exp = ~mask_poly
    
    # Convert a_i list to a tensor
    a_i_tensor = torch.tensor(a_i, dtype=torch.float32, device=DEVICE).repeat(len(out), 1)

    # Polynomial part for elements < 20
    if mask_poly.any():
        # does tensor multipication
        mult_tensor = a_i_tensor[mask_poly.view(-1)] * (x[mask_poly].unsqueeze(-1) ** torch.arange(n-1, -1, -1, device=DEVICE, dtype=torch.float32))
        # Perform element-wise multiplication with a_i (broadcasted) and sum along the rows
        out[mask_poly] = torch.sum(mult_tensor, dim=1)
    
    # Exponential part for elements >= 20
    if mask_exp.any():
        out[mask_exp] = torch.exp(x[mask_exp])
    
    return out

# Magnetic field derivative
def dH_dx(x):
    out = torch.zeros_like(x).to(DEVICE)
    
    # Mask for elements where x < 20
    mask_poly = x < 20
    mask_exp = ~mask_poly
    
    # Convert a_i list to a tensor
    a_i_tensor = torch.tensor(a_i[1:], dtype=torch.float32, device=DEVICE).repeat(len(out), 1)
    # Convert range from n to 1 to a tensor
    i_tensor = torch.arange(n-1, 0, -1, device=DEVICE, dtype=torch.float32).repeat(len(out), 1)

    # Polynomial derivative part for elements < 20
    if mask_poly.any():
        # does tensor multipication
        mult_tensor = i_tensor[mask_poly.view(-1)] * a_i_tensor[mask_poly.view(-1)] * (x[mask_poly].unsqueeze(-1) ** torch.arange(n-2, -1, -1, device=DEVICE, dtype=torch.float32))
        # Perform element-wise multiplication with a_i (broadcasted) and sum along the rows
        out[mask_poly] = torch.sum(mult_tensor, dim=1)
    
    # Exponential derivative part for elements >= 20
    if mask_exp.any():
        out[mask_exp] = torch.exp(x[mask_exp])
    
    return out

# nondimensional version of dx/dt (dx*/dt* instead of dx/dt)
def dx_dt_nondim(t_star, x_star, droplet_size_idx):
    return (V[droplet_size_idx]*M(x_0*x_star)*mu_0*dH_dx(x_0*x_star)*torch.exp(t_star)) / (6*np.pi*r[droplet_size_idx]*eta*x_0)

def physics_loss_nondim(model: torch.nn.Module):
    ts = torch.linspace(np.log(1e-50), np.log(600), steps=600,).view(-1, 1).requires_grad_(True).to(DEVICE)
    xs = model(ts)
    dx = grad(xs, ts)[0]
    pde = dx_dt_nondim(ts, xs, model.droplet_size_idx) - dx
    return torch.mean(pde**2)

def physics_loss_dimensional(model: torch.nn.Module):
    ts = torch.linspace(0, 600, steps=600,).view(-1, 1).requires_grad_(True).to(DEVICE)
    xs = model(ts)
    dx = grad(xs, ts)[0]
    xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(xs)
    pde = (V*mu_0*phi*M_d*dH_dx(xs)*(1/torch.tanh(xi) - 1/xi))/(6*r*np.pi*eta) - dx
    return torch.mean(pde**2)