import torch
import numpy as np
import pandas as pd
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define start and endpoints of the domain of the training data
domain_start = 0
domain_end = 2

# measures how far to extend the domain when plugging in points for physics loss as a percentage
# ex. if domain_start=0 and domain_end=2, if this value is 20 then the points chosen to evaluate physics loss are chosen from 0-2.4 instead of 0-2
# useful to test how far the model can predict beyond the original domain
dom_ext = 20
domain_end_ext = domain_end + (domain_end - domain_start)*(dom_ext / 100)

# number of evaluation points used for physics loss
num_eval_points = 600

# coefficient in front of e^-x
c = 0.5

# will create training data based on equation y=c*e^-x with random noise with standard deviation noise_spread if noise is turned on.
def grab_e_data(noise=False, noise_spread=1, n_points=5):
    dim_data_t = torch.linspace(domain_start, domain_end, n_points)
    dim_data_x = dim_data_t.detach().clone()
    dim_data_x = c * torch.exp(-dim_data_x)
    if noise:
        dim_data_x += torch.normal(mean=0.0, std=noise_spread, size=(n_points,))
    return dim_data_t, dim_data_x

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

def physics_loss_e(model: torch.nn.Module):
    ts_min, ts_max = domain_start, domain_end_ext
    ts = torch.linspace(ts_min, ts_max, steps=num_eval_points,).view(-1, 1).requires_grad_(True).to(DEVICE)
    xs = model(ts)
    dx = grad(xs, ts)[0]
    pde = -xs - dx # dx/dt = -x
    return torch.mean(pde**2)