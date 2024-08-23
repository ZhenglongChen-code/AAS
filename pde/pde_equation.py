import numpy as np
import torch


def convection_1d(model, X):
    """
    a simple convection equation: u_t + a*u_x =0, when t=0, u = u0(x)
    :param model: a nn model to solve pde
    :param X: input X=(x,t), model.net(X) = u
    :return: u_t + a*u_x
    """
    x = X[:, 0:1]
    t = X[:, 1:]
    u = model.net(X)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    return u_t + model.a * u_x
