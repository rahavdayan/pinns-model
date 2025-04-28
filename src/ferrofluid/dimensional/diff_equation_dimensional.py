import torch
import numpy as np
import pandas as pd
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use the magnetic field data to create an exponential fit
def exponential_fit():
    # Load the CSV data
    with open('../droplet_data/magnetic_field_data.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader), dtype=float)
    
    # Extract x and y values
    x_target, y_target = data[:, 0], data[:, 1]

    # Log-transform y values
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

# come up with a domain surrounding the dimensionalized training data points
def get_domain_dim(droplet_size_idx):
    min_value = dim_data[droplet_size_idx]["DISTANCE"].min()
    max_value = dim_data[droplet_size_idx]["DISTANCE"].max()
    # this extends the domain by 20% the original interval to the right
    return min_value, max_value + (max_value - min_value)*(dom_ext / 100)

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

def grab_training_data(real = False):
    # get training data csvs
    droplet_file_names = []
    if real:
        droplet_file_names = ['averages_1mm.csv', 'averages_2mm.csv', 'averages_3mm.csv', 'averages_4mm.csv']
    else:
        droplet_file_names = ['droplet_1mm.csv', 'droplet_2mm.csv', 'droplet_3mm.csv', 'droplet_4mm.csv']
    file_length = len(droplet_file_names)
    dim_data = [None] * file_length
    train_data = [None] * file_length
    test_data = [None] * file_length
    
    # store csv for each size droplet into an array, one for dimensionalized data and one for nondimensionalized
    for i in range (0, file_length):
        dim_data[i] = pd.read_csv('./droplet_data/' + droplet_file_names[i])

        # Split into train and test sets
        train_data[i] = dim_data[i][dim_data[i]['DISTANCE'] < 0.014]
        test_data[i] = dim_data[i][dim_data[i]['DISTANCE'] >= 0.014]

    return train_data, test_data

# H(x) polynomial coefficients from highest deg to lowest, x is measured in m and H in A/m
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
d = 76*1e-9                                                     # mean diameter of nanoparticles [m]
k_B = 1.3806452*(10**-23)                                       # Boltzmann constant [m^2*Kg*s^-2*K^-1]
T = 293                                                         # Absolute temperature [K]
phi = 1/4                                                       # volume fraction of magnetic nanoparticles (25%, mentioned somwhere in background paper)
exp, exp_deriv = exponential_fit()                              # Exponential fit to the magnetic field data and its derivtive
dim_data = grab_training_data()
x_c = 0.02                                                      # Cutoff value for piecewise H(x) and dH_dx(x) in default units, mm
k = 0.003                                                       # Value of k for M(x) = k*H(x)
dom_ext = 40                                                    # Percentage in which domain of evaluation of physics loss is extended from the domain of the training data

def H_noexp(x):
    return a_i[0] * x**(5) + a_i[1] * x**(4) + a_i[2] * x**(3) + a_i[3] * x**(2) + a_i[4] * x + a_i[5]

def dH_dx_noexp(x):
    return 5*a_i[0] * x**(4) + 4*a_i[1] * x**(3) + 3*a_i[2] * x**(2) + 2*a_i[3] * x + a_i[4]

# H(x) coefficients, it is in the form H(x) = m*(x+c)^(-n)
m_rahav = 0.15412679903407128234604783756367
c_rahav = 0.0077646377575943583554396454360358
n_rahav = 2.6895013909079388270129129523411

# Different variables for problem
dim_data = grab_training_data()
k = 245  # 0.35 for paper 245 for experimental                  # Value of k for M(x) = k*H(x)
dom_ext = 100                                                   # Percentage in which domain of evaluation of physics loss is extended from the domain of the training data
num_eval_points = 50                                            # Number of evaluation points points when determining physics loss

# Magnetization
def M(x):
    return k*H(x)

# Magnetic field
def H(x):
    return (m_rahav * torch.pow(x + c_rahav, -n_rahav)).to(DEVICE)

# Magnetic field derivative
def dH_dx(x):
    return (-n_rahav*m_rahav * torch.pow(x + c_rahav, -n_rahav - 1)).to(DEVICE)

# dimensional differential equation dt/dx, used in dimensional physics loss
def dt_dx_dim(t, x, droplet_size_idx):
    return -(6*np.pi*r[droplet_size_idx]*eta) / (V[droplet_size_idx]*M(x)*mu_0*dH_dx(x))

def physics_loss_dim(model: torch.nn.Module):
    xs_min, xs_max = 0, 0.03
    xs = torch.linspace(xs_min, xs_max, steps=num_eval_points,).view(-1, 1).requires_grad_(True).to(DEVICE)
    ts = model(xs)
    dx = grad(ts, xs)[0]
    pde = dt_dx_dim(ts, xs, model.droplet_size_idx) - dx
    return torch.mean(pde**2)

# Magnetization
# def M(x):
#     return 350 * H_noexp(x)
#     # xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(x, x_c)
#     # return phi*M_d*L(xi)

# def M_Langevin(x, x_c):
#     xi = ((torch.pi*mu_0*M_d*(d**3))/(6*k_B*T)) * H(x, x_c)
#     return phi*M_d*L(xi)

# Magnetic field
# def H(x, x_c):
#     out = torch.zeros_like(x).to(DEVICE)
    
#     # Mask for elements where x < 20
#     mask_poly = x < x_c
#     mask_exp = ~mask_poly
    
#     # Convert a_i list to a tensor
#     a_i_tensor = torch.tensor(a_i, dtype=torch.float32, device=DEVICE).repeat(len(out), 1)

#     # Polynomial part for elements < 20
#     if mask_poly.any():
#         # does tensor multipication
#         mult_tensor = a_i_tensor[mask_poly.view(-1)] * (x[mask_poly].unsqueeze(-1) ** torch.arange(n-1, -1, -1, device=DEVICE, dtype=torch.float32))
#         # Perform element-wise multiplication with a_i (broadcasted) and sum along the rows
#         out[mask_poly] = torch.sum(mult_tensor, dim=1)
    
#     # Exponential part for elements >= 20
#     if mask_exp.any():
#         out[mask_exp] = exp(x[mask_exp])
    
#     return out


# Magnetic field derivative
# def dH_dx(x, x_c):
#     out = torch.zeros_like(x).to(DEVICE)
    
#     # Mask for elements where x < 20
#     mask_poly = x < x_c
#     mask_exp = ~mask_poly
    
#     # Convert a_i list to a tensor
#     a_i_tensor = torch.tensor(a_i[:-1], dtype=torch.float32, device=DEVICE).repeat(len(out), 1)
#     # Convert range from n-1 to 1 to a tensor
#     i_tensor = torch.arange(n-1, 0, -1, device=DEVICE, dtype=torch.float32).repeat(len(out), 1)

    # Polynomial derivative part for elements < 20
#     if mask_poly.any():
#         # does tensor multipication
#         mult_tensor = i_tensor[mask_poly.view(-1)] * a_i_tensor[mask_poly.view(-1)] * (x[mask_poly].unsqueeze(-1) ** torch.arange(n-2, -1, -1, device=DEVICE, dtype=torch.float32))
#         # Perform element-wise multiplication with a_i (broadcasted) and sum along the rows
#         out[mask_poly] = torch.sum(mult_tensor, dim=1)
    
#     # Exponential derivative part for elements >= 20
#     if mask_exp.any():
#         out[mask_exp] = exp_deriv(x[mask_exp])
    
#     return out

# dimensional differential equation dx/dt, used in dimensional physics loss
# def dt_dx_dim(x, x_c, droplet_size_idx):
#     return  -(6*np.pi*r[droplet_size_idx]*eta) / (V[droplet_size_idx]*M(x)*mu_0*dH_dx_noexp(x))

# def physics_loss_dim(model: torch.nn.Module):
#     xs_min, xs_max = [0, 0.02]
#     # xs_min, xs_max = get_domain_dim(model.droplet_size_idx)
#     xs = torch.linspace(xs_min, xs_max, steps=100,).view(-1, 1).requires_grad_(True).to(DEVICE)
#     ts = model(xs)
#     dt = grad(ts, xs)[0]
#     pde = dt_dx_dim(xs, x_c, model.droplet_size_idx) - dt
#     return torch.mean(pde**2)
