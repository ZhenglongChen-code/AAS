from torch.utils.tensorboard import SummaryWriter
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.PDE_module import *
from src.PDE_module import pde_residual


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices


devices = check_devices()
layer_size = [2, 50, 50, 50, 1]
nx, ny = 100, 100
# s = Square(nx=nx, ny=ny, cond='2point')
s = Square(nx=nx, ny=ny)
s.set_bound_val()
model = PINN(layer_size, s, {'k': 5}, devices[0], pde_func=pde_residual, log_dir='./logs/uniform_k/2lines')
model.train_solver(max_iter=20000, interval=1000)
model.visualize()

p_pred = model.predict()