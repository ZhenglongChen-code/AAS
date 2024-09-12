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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


class SequenceModel(nn.Module):
    def __init__(self, model_list):
        super(SequenceModel,self).__init__()
        self.time_step = len(model_list)
        layer_list = []
        for t in range(self.time_step):
            layer_list.append(('step_%d' %t, model_list[t]))

        self.model_list = nn.Sequential(OrderedDict(layer_list))

    def forward(self, press_in):
        return self.model_list(press_in)

    def predict(self, time_step, press_in):
        for i in range(time_step):
            press_out = self.model_list[i](press_in)
            press_in = press_out

        return press_out


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
dt = 30000
qw = 0.0005
chuk = 5e-15
poro = 0.1
rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14*rw*rw*length
bhp_constant = 20*1e6
nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_data = np.load('../dataset/train_perm10d_20x20.npz')
train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']

scaled_trans_params = 1e10
# save data
writer = SummaryWriter(comment='discriminate_data', log_dir='../logs/press+trans')

# constant value
trans_matrix = []
for perm in tqdm(train_permeability, desc='prepare data'):
    mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                    poro, p_init, p_bc, bhp_constant)
    if len(trans_matrix) == 0:
        trans_matrix = mesh.trans_matrix.detach().flatten()
    else:

        trans_matrix = torch.vstack((trans_matrix, mesh.trans_matrix.detach().flatten()))


neighbor_idx = mesh.neighbor_vectors
batch_size = 500

train_loader = DataLoader(trans_matrix * scaled_trans_params, batch_size=batch_size, shuffle=True)

# NN params
criterion = nn.MSELoss()
b1 = nn.Sequential(nn.Conv2d(5, 20, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
b5 = nn.Sequential(*resnet_block(40, 80, 2, first_block=False))

input_size = nx * ny * nz  # 输入为单元渗透率或trans，后者更好
# 实例化模型
model = nn.Sequential(b1, GAM_Attention(20,20), b2,GAM_Attention(20,20),
                      b4, GAM_Attention(40,40), b5, GAM_Attention(80, 80),
                      nn.Flatten(), nn.Linear(720, input_size)).to(torch.float64).to(devices[0])

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.98)

# data prepare
nt = 100
model_list = []
press_history = []
for t in range(nt):
    if t % 10 == 0:
        fig = plt.figure(dpi=600, figsize=(5, 5))
    batch_press = []
    batch = 0
    for trans in train_loader:
        trans = trans.to(select_gpu)
        if t == 0:
            p_last = torch.tile(mesh.press / p_init, (len(trans), 1)).to(select_gpu)
            input_fig = torch.hstack((p_last, trans)).reshape(-1, 5, nx, ny)
        else:
            p_last = press_history[-1][batch]
            input_fig = torch.hstack((p_last, trans)).reshape(-1, 5, nx, ny)

        for i in tqdm(range(5000), desc='training'):
            optimizer.zero_grad()
            model.train()
            p_next = model(input_fig)
            res = torch.zeros_like(p_next)
            res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:]*p_init - p_last[:]*p_init)
            for d in range(4):
                res[:] = (res[:] - trans[:, d * 400:(d + 1) * 400] / scaled_trans_params *
                          (p_next[:, neighbor_idx[d]] * p_init - p_next[:] * p_init))

            res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
            loss = criterion(res, torch.zeros_like(res))
            # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
            loss.backward()  # 仅在必要时使用 retain_graph=True
            optimizer.step()
            scheduler.step()
            if (i+1) % 500 == 0:
                print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i+1, loss.item()))
            if loss < 1e-15:
                print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i + 1, loss.item()))
                break

        batch += 1
        p_last = p_next.detach()  # 计算下一个时间步时，使用 detach()分离数据
        p_last[:, 0] = p_bc/p_init
        batch_press.append(p_last)

    model_list.append(copy.deepcopy(model))
    press_history.append(batch_press)

    # save figure
    axs = fig.add_subplot(4, 3, (t % 10)+1)
    out = p_next[5].detach().cpu().numpy().reshape(nx, ny)
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
        plt.savefig('../figure/press+trans_t_{}.png'.format(t+1))
        plt.show()
        writer.add_figure('press solution', fig, global_step=t+1)


sequence_model = SequenceModel(model_list)
torch.save(sequence_model.model_list, '../model/sequence.pth')

### test file
with open('../dataset/samplesperm.txt') as f:
    data = f.readlines()
    permeability_t = list(map(float, data))
    permeability_t = np.array(permeability_t) * 1e-15

mesh_t = MeshGrid(nx, ny, nz, train_permeability[-1].flatten(), mu_o, ct,poro, p_init, p_bc, bhp_constant)
trans_t = mesh_t.trans_matrix.flatten().to(select_gpu)

for t in range(nt):
    if t % 10 == 0:
        fig = plt.figure(dpi=600, figsize=(5, 5))

    if t == 0:
        test_p_last = mesh_t.press.to(select_gpu) / p_init

    test_fig = torch.hstack((test_p_last, trans_t * scaled_trans_params)).reshape(-1, 5, nx, ny)
    test_p_next = model_list[t](test_fig)
    test_p_last = test_p_next.detach()
    test_p_last[:, 0] = p_bc/p_init

    # plot setting
    axs = fig.add_subplot(4, 3, (t % 10) + 1)
    out = test_p_last[0].detach().cpu().numpy().reshape(nx, ny)
    gca = axs.imshow(out, origin='lower', cmap='viridis')
    axs.set_xlabel('X', fontsize=5)
    axs.set_ylabel('Y', fontsize=5)
    axs.set_title('press_t_{}'.format(t + 1), fontsize=5)
    axs.tick_params(axis='both', labelsize=5)
    cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                        ticks=np.linspace(out.min(), out.max(), 5, endpoint=True),
                        format='%.2f')  # ,label='Press Values'
    # 设置 colorbar 的刻度标签大小
    cbar.ax.tick_params(labelsize=2)

    if (t + 1) % 10 == 0:
        plt.savefig('../figure/press+trans_test_t_{}.png'.format(t + 1))
        plt.show()
        writer.add_figure('test press solution', fig, global_step=t + 1)