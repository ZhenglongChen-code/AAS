# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/8 12:27
@Auth ： ChenZL
@File ：exp2.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
from torch import nn
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src import *
import os
import argparse

def pde_residual(X: torch.Tensor, output: torch.Tensor, k) -> torch.Tensor:
    """
    compute pde residual function, which will be use in the PINN class, it should be rewritten for a new problem.
    in this exp2 program, we design a DNN with input: X=[x,y,t,k], therefore k in here is useless
    :param k: permeability
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

    k = k.flatten()
    res_vec = k * (p_xx + p_yy) - p_t
    if k.shape != res_vec.shape:
        raise RuntimeError('the shape of permeability is different from residual vector, '
                           'which will cause broadcast error!')
    else:
        return res_vec


# os.environ['CUDA_VISIBLE_DEVICES'] == '1,4'
devices = check_devices()
# physical condition settings
x_min, x_max, y_min, y_max, t_max = 0, 100, 0, 100, 1e+5
dx, dy, dt = 5, 5, 1e+4
nx, ny, nt = int((x_max-x_min)//dx + 1), int((y_max-y_min)//dy + 1), int(t_max//dt +1)
# shape = (nx, ny)
domain = Square(x_min=x_min, x_max=x_max, nx=nx, y_min=y_min, y_max=y_max, ny=ny)
Time_space = TimeSpaceDomain(0, 10, nt=nt, x_min=x_min, x_max=x_max, nx=nx,
                             y_min=y_min, y_max=y_max, ny=ny,
                             cond='2points', bound_val=np.array([20, 20]) * 1e+8, init_val=30 * 1e+8 )
k = np.array([3, 6, 10, 14]) * 5 * 1e-15
Time_space.set_permeability(k=k)
layer_size = [3, 100, 100, 100, 100, 1]

model = PINN(solver_layers=layer_size, domain=Time_space,
             device_ids=[0], log_dir='../logs/exp_real', pde_func=pde_residual, lr=0.1)
#
model.train_solver(max_iter=1000, interval=100)

torch.save(model.best_solver, '../model/pde_t_20000.pt')
# def main():



