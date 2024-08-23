import os
import sys

import numpy as np
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
    p_t = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 2]
    p_xx = torch.autograd.grad(p_x, X, grad_outputs=torch.ones_like(p_x),
                               retain_graph=True,
                               create_graph=True)[0][:, 0]
    p_yy = torch.autograd.grad(p_y, X, grad_outputs=torch.ones_like(p_y),
                               retain_graph=True,
                               create_graph=True)[0][:, 1]

    return k * (p_xx + p_yy) - p_t


class TimeSpaceDomain(Square):
    def __init__(self, t_min, t_max, nt, x_min=0, x_max=1, nx=50, y_min=0, y_max=1, ny=50,
                 cond='2points', bound_val=[0, 1], init_val=0.5):
        super().__init__(x_min=x_min, x_max=x_max, nx=nx, y_min=y_min, y_max=y_max, ny=ny)
        # data set
        self.t_min, self.t_max = t_min, t_max
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        t = np.linspace(t_min, t_max, nt)
        self.shape = (nx, ny, nt)
        self.X, self.Y, self.T = np.meshgrid(x,y,t,  indexing='ij')
        self.x_array = self.X.flatten()
        self.y_array = self.Y.flatten()
        self.t_array = self.T.flatten()

        self.point = np.vstack((self.x_array, self.y_array, self.t_array)).T

        self.index = dict({'bound_idx': np.zeros(1), 'init_idx': np.zeros(1)})
        # if self.index is not None:
        #     print('self.index:', self.index)

        # bound and initial
        self.bound_val = None
        self.set_bound_val(cond=cond, val=bound_val)

        self.init_point = None
        self.init_val = None
        self.set_initial_val(init_val)



    def set_bound_val(self, cond='2points', val=[0, 1]):
        if cond == '2lines':
            lb_idx = np.where(self.x_array == self.x_min)[0]
            rb_idx = np.where(self.x_array == self.x_max)[0]
            lb_cond = np.ones(lb_idx.size) * val[0]
            rb_cond = np.ones(rb_idx.size) * val[1]
        elif cond == '2points':
            lb_idx = np.where(np.all((self.x_array == self.x_min, self.y_array == self.y_min), axis=0))[0]
            rb_idx = np.where(np.all((self.x_array == self.x_max, self.y_array == self.y_max), axis=0))[0]
            lb_cond = np.ones(lb_idx.size) * val[0]
            rb_cond = np.ones(rb_idx.size) * val[1]

        lb, rb = self.point[lb_idx], self.point[rb_idx]

        self.bound_val = torch.tensor(np.concatenate((lb_cond, rb_cond), axis=0),
                                      dtype=torch.float32).reshape(-1, 1)
        bound_idx = np.append(lb_idx, rb_idx)
        self.index['bound_idx'] = bound_idx
        inner_idx = [i for i in range(self.nx * self.ny) if i not in bound_idx]
        self.inner_point = torch.tensor(self.point[inner_idx], dtype=torch.float32, requires_grad=True)
        self.bound_point = torch.tensor(np.concatenate((lb, rb), axis=0),
                                        dtype=torch.float32, requires_grad=True)  # coord of boundary points
        self.point = torch.tensor(self.point, dtype=torch.float32, requires_grad=True)

    def set_initial_val(self, val):
        init_idx = np.where(self.t_array==self.t_min)[0]
        init_idx = [i for i in init_idx if i not in self.index['bound_idx'] ]
        self.index['init_idx'] = init_idx
        self.init_val = torch.ones((len(init_idx), 1), dtype=torch.float32) * val
        # self.init_val = torch.tensor(np.ones(len(init_idx)) * val, dtype=torch.float32).reshape(-1, 1)
        self.init_point = self.point[init_idx].clone().detach().requires_grad_(True)


devices = check_devices()
nx, ny, nt = 40, 40, 10
shape = (nx, ny)
domain = Square(nx=nx, ny=ny)
Time_space = TimeSpaceDomain(0, 10, nt=nt, nx=nx, ny=ny, cond='2points', bound_val=[0, 10], init_val=2 )
# Time_space.set_bound_val(cond='2points')
# Time_space.set_initial_val(val=0.5)
layer_size = [3, 50, 50, 50, 1]
# k = torch.rand(shape) * (20 - 0) + 0
params = {'k': 1}
model = PINN(solver_layers=layer_size, domain=Time_space,
             device=devices[0], params=params,
             log_dir='./logs/exp_1', pde_func=pde_residual)
#
model.train_solver(max_iter=20000, interval=1000)
#
# TS = TimeSpaceDomain(0, 1, 10)
