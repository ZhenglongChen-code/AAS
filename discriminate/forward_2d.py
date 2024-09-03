import os
import sys
import time
import copy
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
from src.Net_structure import *

# init params
mu_o = 2e-3
ct = 5e-8
p_init = 30.0 * 1e6
nt = 180
dt = 30000
qw = 0.0005
chuk = 5e-15
poro = 0.1

rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14*rw*rw*length
bhp_constant = 20e6
nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_data = np.load('../dataset/train_data_20x20.npz')
train_press, train_permeability = train_data['train_press'], train_data['train_permeability']
test_data = np.load('../dataset/test_data_20x20.npz')
test_press, test_permeability = test_data['test_press'], test_data['test_permeability']

# save data
writer = SummaryWriter(comment='PINNS_data', log_dir='../logs/forward_2d_2')
# init and condition
p_init, p_bc = 30 * 1e+6, 20 * 1e+6
with open('../dataset/samplesperm.txt') as f:
    data = f.readlines()
    permeability = list(map(float, data))
    permeability = np.array(permeability) * 1e-15

mesh = MeshGrid(nx, ny, nz, train_permeability[5].flatten())
# mesh = MeshGrid(nx, ny, nz, permeability.flatten())

# NN params
criterion = nn.MSELoss()
b1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
# b5 = nn.Sequential(*resnet_block(40,80,2,first_block=False))

input_size = nx * ny * nz  # 输入为单元渗透率或trans，后者更好
# 实例化模型
model = nn.Sequential(b1, b2, b3, b4,
                      nn.Flatten(), nn.Linear(1000, input_size)).to(torch.float64).cuda()

class CNN(torch.nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 100*3*20*20 -> 100*25*18*18: (100) 对每一个batch即time step (2) channel
            torch.nn.Conv2d(1, 25, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # # ->100*25*9*9
        )

        self.conv2 = torch.nn.Sequential( # ->100*50*7*7
            torch.nn.Conv2d(25, 50, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # ->100*50*3*3
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, output_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(600, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)   # x: tensor (time steps (100), 50*3*3)
        x = self.fc(x) # x: tensor (time steps (100), output size)
        return x


# model = CNN(input_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
# reshape data
batch_size = 1
dt = 10000

# constant value when permeability have been set
trans_matrix = mesh.trans_matrix
neighbor_idx = mesh.neighbor_vectors


def solve_func(p_last, p_next, trans_matrix, neighbor_idx):
    """
    :param p_last: scaled pressure, p_scaled = p / p_init
    :param trans_matrix:
    :param neighbor_idx:
    :return:
    """
    # p_input = p_last.reshape(batch_size, nz, nx, ny)
    # p_next = model(p_input)[0]  # 先测试单条数据
    # p_next == outputs, (p_next - p_last) * p_init 与函数中的(u[:] - press1[:]*p_init) 不同, 有计算精度误差
    res = torch.zeros_like(p_last)
    res += mesh.cell_volume * mesh.porosity * mesh.ct / dt * (p_next * p_init - p_last * p_init)
    for d in range(4):
        res -= trans_matrix[d] * (p_next[neighbor_idx[d]] * p_init - p_next * p_init)
    res[0] += mesh.PI * (p_next[0]*p_init - mesh.pwf)
    loss = criterion(res, torch.zeros_like(res))
    return loss, p_next


p_last = mesh.press / p_init
torch.autograd.set_detect_anomaly(True)


nt = 100
model_list = []
for t in range(nt):
    if t % 10 == 0:
        fig = plt.figure(dpi=600, figsize=(5, 5))

    min_loss = 1e+5
    loss = 1e+6
    for i in tqdm(range(1000), desc='training'):
        optimizer.zero_grad()
        model.train()
        p_input = p_last.reshape(batch_size, nz, nx, ny)
        p_next = model(p_input)[0]

        res = torch.zeros_like(p_next)
        res = res + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next*p_init - p_last*p_init)
        for d in range(4):
            res = res - trans_matrix[d] * (p_next[neighbor_idx[d]] * p_init - p_next * p_init)
        res[0] = res[0] + mesh.PI * (p_next[0] * p_init - mesh.pwf)
        loss = criterion(res, torch.zeros_like(res))
        # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
        loss.backward()  # 仅在必要时使用 retain_graph=True
        optimizer.step()
        scheduler.step()
        if (i+1) % 100 == 0:
            print('time step:{}, train step:{:d}, loss{}'.format(t, i+1, loss.item()))
        if loss < 1e-17:
            print('time step:{}, train step:{:d}, loss{}'.format(t, i + 1, loss.item()))
            break

    p_last = p_next.detach()  # 计算下一个时间步时，使用 detach()分离数据
    p_last[0] = p_bc/p_init
    model_list.append(copy.deepcopy(model))

    # save figure
    axs = fig.add_subplot(4, 3, (t % 10)+1)
    out = p_next.detach().cpu().numpy().reshape(nx, ny)
    gca = axs.imshow(out, origin='lower', cmap='viridis')
    axs.set_xlabel('X', fontsize=5)
    axs.set_ylabel('Y', fontsize=5)
    axs.set_title('press_t_{}'.format(t+1), fontsize=5)
    axs.tick_params(axis='both', labelsize=5)
    cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                        ticks=np.linspace(out.min(), out.max(), 5, endpoint=True), format='%.2f') # ,label='Press Values'
    # 设置 colorbar 的刻度标签大小
    cbar.ax.tick_params(labelsize=2)
    if (t+1) % 10 == 0:
        plt.suptitle('forward1_t_{}'.format(t+1))
        plt.savefig('../figure/forward1_t_{}.png'.format(t+1))
        plt.show()
        writer.add_figure('press solution', fig, global_step=t+1)


