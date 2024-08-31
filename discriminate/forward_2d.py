import os
import sys
import time
import copy
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
from src.Net_structure import *

# class Sequential_model(nn.Module):
#     def __init__(self, model_list):
#         self.model_list = model_list
#
#     def

nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
# train_data = np.load('../dataset/train_data_10x10.npz')
# train_press, train_permeability = train_data['train_press'], train_data['train_permeability']
# test_data = np.load('../dataset/test_data_10x10.npz')
# test_press, test_permeability = test_data['test_press'], test_data['test_permeability']
# init and condition
p_init, p_bc = 30 * 1e+6, 20 * 1e+6
with open('../dataset/samplesperm.txt') as f:
    data = f.readlines()
    permeability = list(map(float, data))
    permeability = np.array(permeability) * 1e-15

mesh = MeshGrid(nx, ny, nz, permeability.flatten())


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
                      nn.Flatten(), nn.Linear(1000, input_size)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

# fig, axes = plt.subplots(1, 1)
nt = 4
model_list = []
for t in range(nt):
    loss = 1e+6
    for i in tqdm(range(500), desc='training'):
        optimizer.zero_grad()
        model.train()
        p_input = p_last.reshape(batch_size, nz, nx, ny)
        p_next = model(p_input)

        res = torch.zeros_like(p_next)
        res = res + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next*p_init - p_last*p_init)
        for d in range(4):
            res = res - trans_matrix[d] * (p_next[:, neighbor_idx[d]] * p_init - p_next * p_init)
        res[0] = res[0] + mesh.PI * (p_next[0] * p_init - mesh.pwf)
        loss = criterion(res, torch.zeros_like(res))
        # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
        loss.backward()  # 仅在必要时使用 retain_graph=True
        optimizer.step()
        if (i+1) % 100 == 0:
            print('time step:{}, train step:{:d}, loss{}'.format(t, i+1, loss.item()))
        if loss < 1e-12:
            plt.imshow(p_next.detach().cpu().numpy().reshape(nx, ny))
            plt.show()
            break

    p_last = p_next.detach()  # 计算下一个时间步时，使用 detach()分离数据
    model_list.append(copy.deepcopy(model))

    plt.imshow(p_next.detach().cpu().numpy().reshape(nx, ny))
    plt.show()


