import os
import sys

import numpy as np
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.PDE_module import *
from src.Data import *

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
    p_t = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 2]
    p_xx = torch.autograd.grad(p_x, X, grad_outputs=torch.ones_like(p_x),
                               retain_graph=True,
                               create_graph=True)[0][:, 0]
    p_yy = torch.autograd.grad(p_y, X, grad_outputs=torch.ones_like(p_y),
                               retain_graph=True,
                               create_graph=True)[0][:, 1]

    return k.flatten() * (p_xx + p_yy) - p_t


devices = check_devices()
nx, ny, nt = 40, 40, 10
shape = (nx, ny)
domain = Square(nx=nx, ny=ny)
Time_space = TimeSpaceDomain(0, 10, nt=nt, nx=nx, ny=ny, cond='2points', bound_val=[0, 10], init_val=2)
Time_space.set_permeability(k=[3, 9, 15, 20])
layer_size = [3, 100, 100, 100, 1]

model = PINN(solver_layers=layer_size, domain=Time_space,
             device_ids=[0], log_dir='../logs/exp_1.2', pde_func=pde_residual, lr=3e-3)
#
model.train_solver(max_iter=20000, interval=1000)
#
# TS = TimeSpaceDomain(0, 1, 10)
