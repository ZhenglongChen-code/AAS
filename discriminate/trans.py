import os
import sys
import time
import copy
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import Dataloader
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


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices

devices = check_devices()
# init and condition
p_init, p_bc = 30 * 1e+6, 20 * 1e+6
mu_o = 2e-3
ct = 5e-8
p_init = 30.0 * 1e6
nt = 100
dt = 10000
qw = 0.0005
chuk = 5e-15
poro = 0.1
rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14*rw*rw*length
bhp_constant = 20*1e6
nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_data = np.load('../dataset/train_data_20x20.npz')
train_press, train_permeability = train_data['train_press'], train_data['train_permeability']
test_data = np.load('../dataset/test_data_20x20.npz')
test_press, test_permeability = test_data['test_press'], test_data['test_permeability']

scaled_trans_params = 1e11
# save data
writer = SummaryWriter(comment='discriminate_data', log_dir='../logs/trans')

# constant value
# trans_matrix = []
X = []
trans_w, trans_e, trans_s, trans_n = [], [], [], []
for perm in train_permeability:
    mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                    poro, p_init, p_bc, bhp_constant)
    #  trans_matrix.append(mesh.trans_matrix)
    x_i = torch.vstack((mesh.trans_matrix[0].detach() ,
                        mesh.trans_matrix[1].detach() ,
                        mesh.trans_matrix[2].detach() ,
                        mesh.trans_matrix[3].detach() ))
    if len(X) == 0:
        X = x_i
        permeability = torch.tensor(perm)
    else:
        X = torch.vstack((X, x_i))
        permeability = torch.vstack((permeability, torch.tensor(perm)))

    # press_input.append(mesh.press)

# trans_w = torch.tensor(trans_w).to(devices[0])
# trans_e = torch.tensor(trans_e).to(devices[0])
# trans_n = torch.tensor(trans_n).to(devices[0])
# trans_s = torch.tensor(trans_s).to(devices[0])

neighbor_idx = mesh.neighbor_vectors
batch_size = 100

input_tensor = X.reshape(batch_size, 4, nx, ny).to(devices[0])
# perm_loader = Dataloader()

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
                      nn.Flatten(), nn.Linear(720, input_size)).to(torch.float64).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.98)

# data prepare
p_last = torch.tile(mesh.press, (batch_size, 1))

trans_w = input_tensor[:, 0, :, :].reshape(-1, 400)
trans_e = input_tensor[:, 1, :, :].reshape(-1, 400)
trans_n = input_tensor[:, 2, :, :].reshape(-1, 400)
trans_s = input_tensor[:, 3, :, :].reshape(-1, 400)
nt = 100
model_list = []
for t in range(nt):
    if t % 10 == 0:
        fig = plt.figure(dpi=600, figsize=(5, 5))

    for i in tqdm(range(2000), desc='training'):
        optimizer.zero_grad()
        model.train()
        # p_input = p_last.reshape(batch_size, nz, nx, ny)
        p_next = model(input_tensor)  # size: [m, 400]
        res = torch.zeros_like(p_next)
        res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:]*p_init - p_last[:]*p_init)

        res[:] = res[:] - trans_w * (p_next[:, neighbor_idx[0]] * p_init - p_next[:] * p_init)
        res[:] = res[:] - trans_e * (p_next[:, neighbor_idx[1]] * p_init - p_next[:] * p_init)
        res[:] = res[:] - trans_n * (p_next[:, neighbor_idx[2]] * p_init - p_next[:] * p_init)
        res[:] = res[:] - trans_s * (p_next[:, neighbor_idx[3]] * p_init - p_next[:] * p_init)

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

    p_last = p_next.detach()  # 计算下一个时间步时，使用 detach()分离数据
    p_last[:, 0] = p_bc/p_init
    input_tensor[:, 0, :, :] = p_last.reshape(-1, 20, 20)
    p_last = input_tensor[:, 0, :, :].reshape(-1, 400)  # already scaled in to [0,1]
    trans_w = input_tensor[:, 1, :, :].reshape(-1, 400)
    trans_e = input_tensor[:, 2, :, :].reshape(-1, 400)
    trans_n = input_tensor[:, 3, :, :].reshape(-1, 400)
    trans_s = input_tensor[:, 4, :, :].reshape(-1, 400)
    model_list.append(copy.deepcopy(model))

    # save figure
    axs = fig.add_subplot(4, 3, (t % 10)+1)
    out = p_next[-1].detach().cpu().numpy().reshape(nx, ny)
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
        plt.savefig('../figure/trans_t_{}.png'.format(t+1))
        plt.show()
        writer.add_figure('press solution', fig, global_step=t+1)


sequence_model = SequenceModel(model_list)
torch.save(sequence_model.model_list, '../model/sequence.pth')
