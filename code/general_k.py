import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.PDE_module import *


def pde_residual(X: torch.Tensor, output: torch.Tensor, k) -> torch.Tensor:
    """
    compute pde residual function, which will be use in the PINN class, it should be rewritten for a new problem.
    :param k: params
    :param X: input point data [x,y,t]
    :param output: press
    :return: e.g. pde: L(U) = f, return res = L(U) - f, res should converge to 0, L is an operator related to pde.
    """
    p_x = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 0]
    p_y = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 1]
    p_xx = torch.autograd.grad(p_x, X, grad_outputs=torch.ones_like(p_x),
                               retain_graph=True,
                               create_graph=True)[0][:, 0]
    p_yy = torch.autograd.grad(p_y, X, grad_outputs=torch.ones_like(p_y),
                               retain_graph=True,
                               create_graph=True)[0][:, 1]

    return k * (p_xx + p_yy)


devices = check_devices()
nx, ny = 100, 100
shape = (nx, ny)
domain = Square(nx=nx, ny=ny, bound_cond='2points', bound_val=[0, 1])

layer_size = [2, 50, 50, 50, 1]
k = torch.rand(shape) * (20 - 0) + 0
params = {'k': k.flatten().to(devices[0])}
model = PINN(solver_layers=layer_size, domain=domain,
             device=devices[0], params=params,
             log_dir='./logs/general_k', pde_func=pde_residual)

model.train_solver(max_iter=20000, interval=1000)
