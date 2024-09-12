import os
import sys
import time
import copy
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
from src.Net_structure import *
from collections import OrderedDict


class SequenceModel(nn.Module):
    def __init__(self, model_list):
        super(SequenceModel, self).__init__()
        self.time_step = len(model_list)
        layer_list = []
        for t in range(self.time_step):
            layer_list.append(('step_%d' % t, model_list[t]))

        self.model_list = nn.Sequential(OrderedDict(layer_list))

    def forward(self, input_fig):
        return self.model_list(input_fig).detach()

    def predict(self, time_step, input_fig):

        press_out = self.model_list[time_step-1](input_fig)
        return press_out.detach()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices


devices = check_devices()
select_gpu = devices[0]
# init and condition
p_init, p_bc = 30 * 1e+6, 20 * 1e+6
mu_o = 2e-3
ct = 5e-8
p_init = 30.0 * 1e6
nt = 100
dt = 15000
qw = 0.0005
chuk = 5e-15
poro = 0.1
rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14 * rw * rw * length
bhp_constant = 20 * 1e6
nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_data = np.load('../dataset/train_perm10d_20x20.npz')
train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']

# save data
writer = SummaryWriter(comment='discriminate_data', log_dir='../logs/press+perme_10d')

# constant value
trans_matrix = []
for perm in tqdm(train_permeability, desc='data preparing'):
    mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                    poro, p_init, p_bc, bhp_constant, devices=[select_gpu])
    #  trans_matrix.append(mesh.trans_matrix)

    if len(trans_matrix) == 0:
        trans_matrix = mesh.trans_matrix.detach().flatten()
        permeability = torch.tensor(perm * 1e15)
    else:
        trans_matrix = torch.vstack((trans_matrix, mesh.trans_matrix.detach().flatten()))
        permeability = torch.vstack((permeability, torch.tensor(perm * 1e15)))


batch_size = 500
press_input = torch.tile(mesh.press, (batch_size, 1)) / p_init  # scaled press
scaled_params = train_permeability.mean()
train_loader = DataLoader(MyDataset(permeability / scaled_params, trans_matrix),
                          batch_size=batch_size, shuffle=True)

neighbor_idx = mesh.neighbor_vectors  # constant

# NN params
criterion = nn.MSELoss()
b1 = nn.Sequential(nn.Conv2d(2, 20, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
# b5 = nn.Sequential(*resnet_block(40,80,2,first_block=False))

input_size = nx * ny * nz  # 输入为单元渗透率或trans，后者更好
# 实例化模型
model = nn.Sequential(b1, b2, b3, b4,
                      nn.Flatten(), nn.Linear(1000, input_size)).to(torch.float64).to(select_gpu)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = StepLR(optimizer, step_size=500, gamma=0.95)

# data prepare
nt = 50
model_list = []

for t in range(nt):
    if t % 10 == 0:
        fig_train = plt.figure(dpi=600, figsize=(5, 5))
        # fig_test = plt.figure(dpi=600, figsize=(5, 5))

    batch = 0
    for k, trans in train_loader:
        trans = trans.to(select_gpu)

        if t == 0:
            p_last = torch.tile(mesh.press / p_init, (batch_size, 1)).to(select_gpu)
            input_fig = torch.hstack((p_last, k.to(select_gpu))).reshape(-1, 2, nx, ny).to(select_gpu)
        else:
            p_last = torch.tile(mesh.press / p_init, (batch_size, 1)).to(select_gpu)
            for i in range(t):
                input_fig = torch.hstack((p_last, k.to(select_gpu))).reshape(-1, 2, nx, ny).to(select_gpu)
                p_next = model_list[i](input_fig)
                p_last = p_next.detach()
                p_last[:, 0] = p_bc / p_init

            input_fig = torch.hstack((p_last, k.to(select_gpu))).reshape(-1, 2, nx, ny).to(select_gpu)

        for i in tqdm(range(5000), desc='training'):
            optimizer.zero_grad()
            model.train()
            p_next = model(input_fig)  # size: [batch_size, 400]

            res = torch.zeros_like(p_next)
            res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:] * p_init - p_last[:] * p_init)
            # for j in range(batch_size):  # 把trans_matrix 改成4个方向的tensor可以加速。
            for d in range(4):
                res[:] = (res[:] - trans[:, d*400:(d+1)*400] *
                          (p_next[:, neighbor_idx[d]] * p_init - p_next[:] * p_init))

            res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
            loss = criterion(res, torch.zeros_like(res))
            # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
            loss.backward()  # 仅在必要时使用 retain_graph=True
            optimizer.step()
            scheduler.step()
            if (i + 1) % 500 == 0:
                print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i + 1, loss.item()))
            if loss < 1e-17:
                print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i + 1, loss.item()))
                break
        batch += 1

    p_last = p_next.detach()  # 计算下一个时间步时，使用 detach()分离数据
    p_last[:, 0] = p_bc / p_init
    model_list.append(copy.deepcopy(model))

    # save test  figure
    axs_train = fig_train.add_subplot(4, 3, (t % 10)+1)
    # axs_test = fig_test.add_subplot(4, 3, (t % 10) + 1)
    # test_perm_tensor = torch.tensor(test_permeability / test_permeability.mean(),
    #                                 dtype=torch.float64).reshape(batch_size, nz, nx, ny).to(devices[0])
    # test_p_next = model(test_perm_tensor)

    # out = test_p_next[0].detach().cpu().numpy().reshape(nx, ny)
    out_train = p_next[0].detach().cpu().numpy().reshape(nx, ny)
    # out_test = test_p_next[0].detach().cpu().numpy().reshape(nx, ny)
    # train fig
    gca_train = axs_train.imshow(out_train, origin='lower', cmap='viridis')
    axs_train.set_xlabel('X', fontsize=5)
    axs_train.set_ylabel('Y', fontsize=5)
    axs_train.set_title('press_t_{}'.format(t+1), fontsize=5)
    axs_train.tick_params(axis='both', labelsize=5)
    cbar_train = fig_train.colorbar(gca_train, ax=axs_train, orientation='vertical', extend='both',
                                    ticks=np.linspace(out_train.min(), out_train.max(), 5, endpoint=True),
                                    format='%.2f')
    # test fig
    # gca_test = axs_test.imshow(out_test, origin='lower', cmap='viridis')
    # axs_test.set_xlabel('X', fontsize=5)
    # axs_test.set_ylabel('Y', fontsize=5)
    # axs_test.set_title('press_t_{}'.format(t + 1), fontsize=5)
    # axs_test.tick_params(axis='both', labelsize=5)
    # cbar_test = fig_test.colorbar(gca_test, ax=axs_test, orientation='vertical', extend='both',
    #                               ticks=np.linspace(out_test.min(), out_test.max(), 5, endpoint=True), format='%.2f')

    # 设置 colorbar 的刻度标签大小
    cbar_train.ax.tick_params(labelsize=2)
    # cbar_test.ax.tick_params(labelsize=2)

    if (t + 1) % 10 == 0:
        fig_train.suptitle('train_permeability_press_t_{}'.format(t+1))
        # fig_test.suptitle('test_permeability_press_t_{}'.format(t + 1))
        plt.savefig('../figure/perme_t_{}.png'.format(t+1))
        plt.show()
        writer.add_figure('train_press solution', fig_train, global_step=t+1)
        # writer.add_figure('test_press solution', fig_test, global_step=t + 1)

sequence_model = SequenceModel(model_list)
torch.save(sequence_model.model_list, '../model/sequence_press+perm.pth')
