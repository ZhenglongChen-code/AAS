import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.PDE_module import *
from src.Data import *
from torch.optim.lr_scheduler import StepLR


def pde_residual(X: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
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

    k = X[:, -1]
    res = k * (p_xx + p_yy) - p_t
    if res.shape == k.shape:
        return res
    else:
        raise RuntimeError('shape of  res is not equal to shape of k')



devices = check_devices()
nx, ny, nt = 40, 40, 10
shape = (nx, ny)
domain = Square(nx=nx, ny=ny)
Time_space = TimeSpaceDomain(0, 10, nt=nt, nx=nx, ny=ny, cond='2points', bound_val=[0, 10], init_val=2)
Time_space.set_permeability()
device_ids = [0]

points, bound_points, bound_val, init_point, init_val, permeability = Time_space.array2tensor(device_ids)
model = DNN(layers=[4, 50, 50, 50, 1]).cuda(device=devices[0])
if len(devices) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

# NN params
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
criterion = nn.MSELoss()

#
train(model, Time_space, pde_residual,
      optimizer, criterion, scheduler, max_iter=1000, interval=100,
      device_ids=device_ids, logdir='./logs/exp_3', embed_k=True)




