import os
import sys
import time
import copy
import torch
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    def predict(self, time_step, input_fig):
        press_out = model_list[time_step-1](input_fig)
        return press_out


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


def read_file(filepath):
    train_data = np.load(filepath)
    train_permeability, test_permeability = train_data['train_permeability'], test_data['test_permeability']
    return train_permeability, test_permeability


parser = argparse.ArgumentParser(prog='train_perm', description='train a permeability NN',
                                 epilog='2024.09')
parser.add_argument('--id', default=0, type=int,
                    help='select a gpu to train NN')
parser.add_argument('-f', '--logdir', default='../logs/train_permeability')
parser.add_argument('-t', '--dt', default=10000, type=int,
                    help='time step of PDE model')
parser.add_argument('--max_iter', default=5000, type=int,
                    help='max iters for training process of NN')
parser.add_argument('--train_data', default='../dataset/train_data_20x20.npz', type=str)
parser.add_argument('-n', '--nt', default=100, type=int,
                    help='how long will the NN model predict')
parser.add_argument('--bs', default=1000, type=int, help='batch size')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')

args = parser.parse_args()

if __name__ == '__main__':
    devices = check_devices()
    selected_gpu = devices[args.id]

    # init and condition
    nt = args.nt
    dt = args.dt
    max_iter = args.max_iter
    p_init, p_bc = 30 * 1e+6, 20 * 1e+6
    mu_o = 2e-3
    ct = 5e-8
    p_init = 30.0 * 1e6

    qw = 0.0005
    chuk = 5e-15
    poro = 0.1
    rw = 0.05
    SS = 3
    length = 3000
    cs = ct * 3.14 * rw * rw * length
    bhp_constant = 20 * 1e6
    nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个

    # load data
    train_permeability, test_permeability = read_file('../dataset/train_perm10d_20x20.npz')

    # save data
    writer = SummaryWriter(comment='discriminate_data', log_dir='../logs/perme_2d')

    # constant value
    trans_matrix = []
    for perm in tqdm(train_permeability[:1000], desc='data preparing'):
        mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                        poro, p_init, p_bc, bhp_constant, devices=[selected_gpu])
        #  trans_matrix.append(mesh.trans_matrix)

        if len(trans_matrix) == 0:
            trans_matrix = mesh.trans_matrix.detach().flatten()
            permeability = torch.tensor(perm * 1e15)
        else:
            trans_matrix = torch.vstack((trans_matrix, mesh.trans_matrix.detach().flatten()))
            permeability = torch.vstack((permeability, torch.tensor(perm * 1e15)))

        # press_input.append(mesh.press)

    # trans_w = torch.tensor(trans_w)
    # trans_e = torch.tensor(trans_e)
    # trans_n = torch.tensor(trans_n)
    # trans_s = torch.tensor(trans_s)

    batch_size = args.bs
    press_input = torch.tile(mesh.press, (batch_size, 1)) / p_init  # scaled press
    scaled_params = train_permeability.mean()
    train_loader = DataLoader(MyDataset(permeability / scaled_params, trans_matrix),
                              batch_size=batch_size, shuffle=True)

    # perm_input = torch.tensor(train_permeability / scaled_params, dtype=torch.float64).to(devices[0])
    neighbor_idx = mesh.neighbor_vectors  # constant

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
    model = nn.Sequential(b1, GAM_Attention(20,20), b2, b3, b4, GAM_Attention(40,40),
                          nn.Flatten(), nn.Linear(1000, input_size)).to(torch.float64).to(selected_gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.95)
    scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)
    # data prepare
    nt = args.nt
    model_list = []

    for t in range(nt):
        if t % 10 == 0:
            fig_train = plt.figure(dpi=600, figsize=(5, 5))
            # fig_test = plt.figure(dpi=600, figsize=(5, 5))

        bach = 0
        for k, trans in train_loader:
            input_fig = k.reshape(batch_size, nz, nx, ny).to(selected_gpu)
            trans = trans.to(selected_gpu)

            if t == 0:
                p_last = torch.tile(mesh.press / p_init, (batch_size, 1)).to(selected_gpu)
            else:
                p_last = model_list[t-1](input_fig)
                p_last = p_last.detach()
                p_last[:, 0] = p_bc / p_init

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
                    print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, bach, i + 1, loss.item()))
                if loss < 1e-17:
                    print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, bach, i + 1, loss.item()))
                    break
            bach += 1

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
    torch.save(sequence_model.model_list, '../model/sequence_perm.pth')
