import os
import sys
import time
import copy
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
# from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
from src.Net_structure import *
from collections import  OrderedDict

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
selected_gpu = devices[0]
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
cs = ct * 3.14*rw*rw*length
bhp_constant = 20*1e6
nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_data = np.load('../dataset/train_perm10d_20x20.npz')
train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']

scaled_trans_params = 1e11
# save data
writer = SummaryWriter(comment='discriminate_data', log_dir='../logs/trans')

# constant value
# trans_matrix = []
X = []
trans_w, trans_e, trans_s, trans_n = [], [], [], []
for perm in tqdm(train_permeability, desc='prepare data'):
    mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                    poro, p_init, p_bc, bhp_constant, devices=[selected_gpu])
    #  trans_matrix.append(mesh.trans_matrix)
    x_i = torch.vstack((mesh.trans_matrix[0].detach(),
                        mesh.trans_matrix[1].detach(),
                        mesh.trans_matrix[2].detach(),
                        mesh.trans_matrix[3].detach())).reshape(1, 4, 400)
    if len(X) == 0:
        X = x_i * scaled_trans_params
        permeability = torch.tensor(perm)
    else:
        X = torch.vstack((X, x_i * scaled_trans_params))
        permeability = torch.vstack((permeability, torch.tensor(perm)))


neighbor_idx = mesh.neighbor_vectors
batch_size = 500
train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)

# NN params
criterion = nn.MSELoss()
b1 = nn.Sequential(nn.Conv2d(4, 20, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
b5 = nn.Sequential(*resnet_block(40,80,2,first_block=False))

input_size = nx * ny * nz  # 输入为单元渗透率或trans，后者更好
# 实例化模型
model = nn.Sequential(b1, b2, b4, b5,
                      nn.Flatten(), nn.Linear(720, input_size)).to(torch.float64).to(selected_gpu)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = StepLR(optimizer, step_size=500, gamma=0.98)

# data prepare

nt = 100
model_list = []
for t in range(nt):
    if t % 10 == 0:
        fig = plt.figure(dpi=600, figsize=(5, 5))

    for trans in train_loader:
        trans = trans.to(selected_gpu)
        input_fig = trans.reshape(-1, 4, nx, ny)

        if t == 0:
            p_last = torch.tile(mesh.press/p_init, (batch_size, 1))
        else:
            p_last = model_list[t-1](input_fig).detach()
            p_last[:, 0] = p_bc/p_init

        for i in tqdm(range(5000), desc='training'):
            optimizer.zero_grad()
            model.train()

            p_next = model(input_fig)  # size: [m, 400]
            res = torch.zeros_like(p_next)
            res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:]*p_init - p_last[:]*p_init)
            
            for d in range(4):
                # res[:] = res[:] - trans[:, d*400:(d+1)*400] * (p_next[:, neighbor_idx[d]] * p_init - p_next[:] * p_init)
                res[:] = (res[:] - trans[:, d, :] / scaled_trans_params *
                          (p_next[:, neighbor_idx[d]] * p_init - p_next[:] * p_init))

            res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
            loss = criterion(res, torch.zeros_like(res))
            # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
            loss.backward()  # 仅在必要时使用 retain_graph=True
            optimizer.step()
            scheduler.step()
            if (i+1) % 500 == 0:
                print('time step:{}, train step:{:d}, loss{}'.format(t, i+1, loss.item()))
            if loss < 1e-15:
                print('time step:{}, train step:{:d}, loss{}'.format(t, i + 1, loss.item()))
                break

    model_list.append(copy.deepcopy(model))

    # save train figure
    axs = fig.add_subplot(4, 3, (t % 10)+1)
    out = p_next[-1].detach().cpu().numpy().reshape(nx, ny)
    gca = axs.imshow(out, origin='lower', cmap='viridis')
    axs.set_xlabel('X', fontsize=5)
    axs.set_ylabel('Y', fontsize=5)
    axs.set_title('press_t_{}'.format(t+1), fontsize=5)
    axs.tick_params(axis='both', labelsize=5)
    cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                        ticks=np.linspace(out.min(), out.max(), 5, endpoint=True), format='%.2f')
    # ,label='Press Values'
    # 设置 colorbar 的刻度标签大小
    cbar.ax.tick_params(labelsize=2)
    if (t+1) % 10 == 0:
        plt.savefig('../figure/trans_t_{}.png'.format(t+1))
        plt.show()
        writer.add_figure('press solution', fig, global_step=t+1)


sequence_model = SequenceModel(model_list)
torch.save(sequence_model.model_list, '../model/sequence_trans.pth')
