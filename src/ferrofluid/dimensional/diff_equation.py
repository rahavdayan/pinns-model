import torch
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_domain_dim():
    min_value = train[0]['DISTANCE'].min()
    max_value = train[0]['DISTANCE'].max()
    # Extend the domain by a percentage (dom_ext) of the original interval
    return min_value, max_value + (max_value - min_value) * (dom_ext / 100)

# gradient computation for physics loss
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

# @param experimental_data (bool): If True, loads experimental droplet data. If False, loads paper data. Default is True.
# @param split_fn (function): A function that splits a pandas DataFrame into training and test data. 
#                              If None, defaults to splitting where 'DISTANCE' < 0.014.
def grab_training_data(experimental_data=True, split_fn=None):
    droplet_file_names = []
    legend = []

    if experimental_data:
        legend = ['averages_1mm.csv', 'averages_2mm.csv', 'averages_3mm.csv', 'averages_4mm.csv']
        droplet_file_names = ['../droplet_data/experimental_data/' + item for item in legend]
    else:
        legend = ['droplet_1mm.csv', 'droplet_2mm.csv', 'droplet_3mm.csv', 'droplet_4mm.csv']
        droplet_file_names = ['../droplet_data/paper_data/' + item for item in legend]

    file_length = len(droplet_file_names)
    dim_data = [None] * file_length
    train_data = [None] * file_length
    test_data = [None] * file_length

    for i in range(file_length):
        dim_data[i] = pd.read_csv(droplet_file_names[i])

        # Use custom split function if provided
        if split_fn is not None:
            train_data[i], test_data[i] = split_fn(dim_data[i])
        else:
            train_data[i] = dim_data[i][dim_data[i]['DISTANCE'] < 0.014]
            test_data[i] = dim_data[i][dim_data[i]['DISTANCE'] >= 0.014]

    return train_data, test_data, legend

r = torch.tensor([0.0005, 0.001, 0.0015, 0.002]).to(DEVICE)     # List of radii of droplet sizes in [m]
V = (4/3)*torch.pi*(r**3).to(DEVICE)                            # List of volumes of droplet sizes in [m^3]
mu_0 = 1.256637*(10**-6)                                        # Permeability of free space [m*kg/(s*A)]
eta = 50                                                        # Viscosity in [Pa*s]
k = 245                                                         # Value of k for M(x) = k*H(x)
dom_ext = 40                                                    # Percentage in which domain of evaluation of physics loss is extended from the domain of the training data
num_eval_points = 50                                            # Number of evaluation points points when determining physics loss
train, test, legend = grab_training_data()                      # Grab training data
lb, ub = get_domain_dim()                                       # Define interval over which to take physics loss

# H(x) coefficients, it is in the form H(x) = m*(x+c)^(-n)
m = 0.15412679903407128234604783756367
c = 0.0077646377575943583554396454360358
n = 2.6895013909079388270129129523411

# Magnetization
def M(x):
    return k*H(x)

# Magnetic field, in the following form: m*(x+c)^(-n)
def H(x):
    return (m * torch.pow(x + c, -n)).to(DEVICE)

# Magnetic field derivative
def dH_dx(x):
    return (-n*m * torch.pow(x + c, -n - 1)).to(DEVICE)

# dimensional differential equation dt/dx, used in dimensional physics loss
def dt_dx_dim(t, x, droplet_size_idx):
    return -(6*np.pi*r[droplet_size_idx]*eta) / (V[droplet_size_idx]*M(x)*mu_0*dH_dx(x))

# calculates physics loss for model to learn from
def physics_loss_dim(model: torch.nn.Module):
    xs = torch.linspace(lb, ub, steps=num_eval_points,).view(-1, 1).requires_grad_(True).to(DEVICE)
    ts = model(xs)
    dx = grad(ts, xs)[0]
    pde = dt_dx_dim(ts, xs, model.droplet_size_idx) - dx
    return torch.mean(pde**2)