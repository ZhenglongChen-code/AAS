import argparse
import copy
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Data import *
from src.PDE_module import check_devices, pde_residual
from src.statistics import DiagGaussian
from src.layers import *


def select_point(x_range, y_range, z):
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    idx1 = np.where(np.all([x_range[0] <= z[:, 0], z[:, 0] <= x_range[1]], axis=0))[0]
    idx2 = np.where(np.all([y_range[0] <= z[:, 1], z[:, 1] <= y_range[1]], axis=0))[0]

    idx = list(set(idx1) & set(idx2))

    return z[idx]


def system_resample(X, weights, resample_num, devise_ids=0):

    cumulative_sum = torch.cumsum(weights/weights.sum(), dim=0)  # 求权重数组的前缀和
    # 将权重数组转化为一个区间数组，其中每个区间的长度与对应粒子的权重成比例
    cumulative_sum[-1] = 1.  # avoid round-off error 避免舍入误差 确保前缀和的最后一个元素确实等于1
    rn = torch.rand(resample_num).cuda(device=devise_ids)
    # 找到前缀和数组中第一个大于随机数 rn[i] 的元素的索引值，将其放入indexs数组
    indexes = torch.searchsorted(cumulative_sum, rn)
    # resample according to indexes
    new_samples = X[indexes]  # 把indexs数组对应的索引所代表的粒子赋值给新的x_P

    return new_samples.detach().cpu().numpy()

class DNN(torch.nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        """
        :param layers: size of each layer of nn, such as [2, 5, 5, 2]
        :param activation: activation function, default: Tanh
        """
        super(DNN, self).__init__()
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):  # last layer do not need activation layer
            layer_list.append(
                ('hidden_layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('batch_norm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))

        layer_list.append(
            ('hidden_layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PINN():
    def __init__(self, solver_layers, domain, device_ids, log_dir='./logs', 
                 pde_func=pde_residual, lr=1e-2, args=None):
        """
        :param solver_layers: pde solver layers;
        :param domain: data point domain;
        :param device_ids:
        """
        # time & space params
        self.domain = domain
        self.device_ids = device_ids
        self.nt = self.domain.shape[-1]
        self.args = args

        # setup init condition
        if self.domain.type == 'time_space':
            self.X, self.bound_point, self.bound_val, self.init_point, self.init_val, self.k = domain.array2tensor(
                device_ids)
        else:
            self.X, self.bound_point, self.bound_val, self.k = domain.array2tensor(device_ids)
            self.init_point = None
            self.init_val = None

        self.original_X = self.X

        # PINN params
        self.solver = DNN(solver_layers).cuda(device=device_ids[0])
        if len(device_ids) > 1:
            self.solver = nn.DataParallel(self.solver, device_ids=device_ids)

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr=lr)
        # self.optimizer = torch.optim.LBFGS(self.solver.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.criterion = nn.MSELoss()

        # adaptive sample parameters
        # self.n_components = list(range(2, 8))
        self.samplers = [GaussianMixture(n_components=i, covariance_type='full', max_iter=2000) for i in list(range(2, 8))]
        
        self.generate_num = len(self.X)//nt // 5

        # pde parameters
        self.residual_func = pde_func

        # other params
        self.min_loss = 1e+5
        self.best_solver = None
        self.writer = SummaryWriter(comment='PINNS_data', log_dir=log_dir)
        self.train_step = 0

    def data_update(self, add_point):
        # if not isinstance(add_point, torch.Tensor):
        #     add_point = add_point.detach()

        k = self.domain.k_list
        n = len(k)
        x_i = np.linspace(self.domain.x_min, self.domain.x_max, n + 1, endpoint=True)
        new_permeability = np.ones(len(add_point)) * k[0]
        if n > 1:
            for i in range(n):
                if i == n - 1:
                    idx = np.where(np.all((add_point[:, 0] >= x_i[i], add_point[:, 0] <= x_i[i + 1]), axis=0))[0]
                    # x <= x_max
                else:
                    idx = np.where(np.all((add_point[:, 0] >= x_i[i], add_point[:, 0] < x_i[i + 1]), axis=0))[0]
                new_permeability[idx] = k[i]
        new_permeability = torch.tensor(new_permeability, dtype=torch.float32).cuda(device=self.device_ids[0]).reshape(-1, 1)
        add_point = torch.tensor(add_point, dtype=torch.float32).cuda(device=self.device_ids[0])
        self.X = torch.vstack((self.X, add_point))
        self.k = torch.vstack((self.k, new_permeability))

    def train_loss(self):
        """
        :return: loss_bound + residual
        """
        bound_pre = self.solver(self.bound_point)
        loss_bound = self.criterion(bound_pre, self.bound_val.reshape(bound_pre.shape))
        loss_init = 0
        if self.init_point is not None:
            init_pre = self.solver(self.init_point)
            loss_init = self.criterion(init_pre, self.init_val.reshape(init_pre.shape))
            self.writer.add_scalar('init_loss', scalar_value=loss_init.detach(), global_step=self.train_step)

        res_vec = self.get_pde_residual()
        residual = self.criterion(res_vec, torch.zeros_like(res_vec))
        # self.loss_history.append([loss_bound.detach().cpu(), residual.detach().cpu()])
        self.train_step += 1
        # self.writer.add_scalar(tag='bound_loss', scalar_value=loss_bound.detach(), global_step=self.train_step)
        # self.writer.add_scalar(tag='residual', scalar_value=residual.detach(), global_step=self.train_step)
        self.writer.add_scalars('Loss', {'bound_loss': loss_bound, 'residual': residual}, self.train_step)
        # bound loss and init loss are difficult to decrease
        return loss_bound + residual + loss_init, res_vec

    def get_pde_residual(self):
        X = self.X
        output = self.solver(X)
        res = self.residual_func(X, output, self.k)
        return res

    def train_solver(self, max_iter=1000, interval=100):
        self.solver.train()
        for epoch in trange(max_iter, desc='training'):
            self.optimizer.zero_grad()
            # using LBFGS optimizer
            # def closure():
            #     loss = self.train_loss()
            #     loss.backward()

            #     self.writer.add_scalar(tag='sum_loss', scalar_value=loss.detach(), global_step=self.train_step)
            #     if loss < self.min_loss:
            #         self.best_solver = copy.deepcopy(self.solver)

            #     if (epoch + 1) % interval == 0:
            #         print('train {:d} steps, train loss: {}'.format(epoch + 1, loss.detach().cpu()))
            #         fig = self.visualize_t()
            #         self.writer.add_figure(tag='solution', figure=fig, global_step=(epoch + 1))
    
            #     return loss
            
            # using Adam optimizer
            loss, res_vec = self.train_loss()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar(tag='sum_loss', scalar_value=loss.detach(), global_step=self.train_step)

            if loss < self.min_loss:
                self.best_solver = copy.deepcopy(self.solver)

            if (epoch + 1) % interval == 0:
                print('train {:d} steps, train loss: {}'.format(epoch + 1, loss.detach().cpu()))

                fig = self.visualize_t()
                self.writer.add_figure(tag='solution', figure=fig, global_step=(epoch + 1))
                fig2 = self.visualize_res(res_vec.abs())  # plot abs residual
                self.writer.add_figure(tag='residual', figure=fig2, global_step=(epoch + 1))

                # using adaptive sample: only use 2 dim Gaussian Mitrix model (x, y)

                fig3 = self.visualize_res(res_vec.abs(), add_point=True)  # plot abs residual and add_point
                self.writer.add_figure(tag='residual and add_point', figure=fig3, global_step=(epoch + 1))
                # update self.X
                # self.data_update(add_point=X_new)

        self.writer.add_graph(self.solver, input_to_model=self.X)
        self.writer.close()


    @torch.no_grad()
    def predict(self, point=None, c='last'):
        if c == 'last':
            solver = self.solver
        elif c == 'best':
            solver = self.best_solver

        solver.eval()
        if point is not None:
            if isinstance(point, np.ndarray):
                point = torch.tensor(point, dtype=torch.float32, requires_grad=False, device=self.device)

            out_pre = solver(point)
            return out_pre.detach().cpu().numpy()
        else:
            # using default point
            out_pre = solver(self.X)
            return out_pre.detach().cpu().numpy()

    def visualize(self):
        out = self.predict(point=self.original_X)
        shape = self.domain.shape
        fig = plt.figure(dpi=300, figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]
        x, y = self.domain.bound_point[:, 0], self.domain.bound_point[:, 1]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        im = ax.imshow(out.reshape(shape), cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
        ax.set_title('press_solution')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both',
                            ticks=[0, 0.25, 0.5, 0.75, 1], format='%.2f', label='Press Values')
        # 设置 colorbar 的刻度标签大小
        cbar.ax.tick_params(labelsize=10)
        # plt.show()
        return fig

    def visualize_t(self):
        out = self.predict(point=self.original_X)
        out = out[list(range(len(self.original_X)))]
        shape = self.domain.shape
        out = out.reshape(shape)
        nt = shape[-1]

        fig = plt.figure(dpi=600, figsize=(15, 15))
        axs = [fig.add_subplot(3, 4, i + 1) for i in range(nt)]

        x, y = self.domain.bound_point[:, 0], self.domain.bound_point[:, 1]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

        for i in range(nt):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')

            # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]
            value = out[:, :, i]
            im = axs[i].imshow(value, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()],
                               origin='lower')

            axs[i].set_xlim(x.min(), x.max())
            axs[i].set_ylim(y.min(), y.max())
            axs[i].plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
            axs[i].set_title('press_t{}'.format(i))
            if i == nt-1:
                cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', extend='both',
                                    ticks=np.linspace(value.min(), value.max(), 5, endpoint=True), format='%.2f',
                                    label='Press Values')
                # 设置 colorbar 的刻度标签大小
                cbar.ax.tick_params(labelsize=10)
        # plt.show()
        return fig

    def visualize_res(self, res_vector, add_point=False):
        out = res_vector.detach().cpu().numpy()
        out = out[list(range(len(self.original_X)))]
        res_min, res_max = np.min(out), np.max(out)
        shape = self.domain.shape
        # print('domain shape:'.format(shape))

        out = out.reshape(shape)
        nt = shape[-1]

        fig = plt.figure(dpi=600, figsize=(10, 10))
        axs = [fig.add_subplot(3, 4, i + 1) for i in range(nt)]
        X, Y = self.domain.X, self.domain.Y
        t_array = self.domain.t_array
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()

        for i in range(nt):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            value = out[:, :, i]
            gca = axs[i].pcolormesh(X[:, :, 0], Y[:, :, 0], out[:, :, i],
                                    shading='auto', cmap=plt.cm.jet)
            if add_point:
                X_new = self.adaptive_sample(i, t_array, res_vector)
                if X_new is not None:
                    if isinstance(X_new, torch.Tensor):
                        X_new = X_new.detach().cpu().numpy()
                    axs[i].plot(X_new[:, 0], X_new[:, 1],  'kx', markersize=3,
                                label='add {} points'.format(len(X_new)), alpha=1.0)

                    # update data in adaptive_sample function
                    # self.data_update(add_point=np.hstack((X_new, t_array[i] * np.ones((len(X_new), 1)))))

            axs[i].set_aspect('equal')
            axs[i].set_xlabel('X', fontsize=5)
            axs[i].set_ylabel('Y', fontsize=5)
            axs[i].set_xlim(self.domain.x_min, self.domain.x_max)
            axs[i].set_ylim(self.domain.y_min, self.domain.y_max)
            axs[i].set_title('residual{}'.format(i), fontsize=5)

            cbar = fig.colorbar(gca, ax=axs[i], orientation='vertical', extend='both',
                                ticks=np.linspace(value.min(), value.max(), 5, endpoint=True), format='%.2f',
                                label='Residual Values')
            # 设置 colorbar 的刻度标签大小
            cbar.ax.tick_params(labelsize=2)

        return fig
    
    def adaptive_sample(self, i, t_array, res_vector):
        # sample datas in each time step, we firstly use system_sample then use Gaussian Mixture
        idx_t = torch.where(self.X[:, -1] == t_array[i])[0]
        X_t = self.X[idx_t, 0:2]  # only sample (x,y)
        residual_t = res_vector[idx_t].abs()

        # 这里可以残差分段多次采样，比如设res_list= [res.min(), res().max]分5段，采样res>reslist[1]
        # 部分的样本点即可
        if residual_t.max() - residual_t.min() < 1e-3:
            print('residual_max - res_min = {}, residual is uniform'.format(residual_t.max() - residual_t.min()))
            return None

        # 1. using Iterative way to resample
        # res_threshold = torch.linspace(residual_t.min().item(), residual_t.max().item(), 5)
        #
        # # sample large residual coord
        # idx_res = torch.where(residual_t > res_threshold[1])[0].detach().cpu().numpy()
        # X_samples = X_t[idx_res]
        #
        # for j in range(2,5):
        #     idx_res_new = torch.where(residual_t > res_threshold[j])[0].detach().cpu().numpy()
        #     idx_res = np.append(idx_res, idx_res_new)
        #
        # X_samples = X_t[idx_res].detach().cpu().numpy()
        # print('sample nums:{}'.format(len(X_samples)))

        # 2. using system resample can ensure get more training data
        # X_samples = system_resample(X_t, residual_t, resample_num=self.generate_num, devise_ids=self.device_ids[0])
        # if len(X_samples) > 10:
        #     # 这里可以分别用多个sampler拟合，然后计算采样点对数概率最大的索引，使用对应sampler
        #     [sampler.fit(X_samples) for sampler in self.samplers]
        #     score_list = [sampler.score_samples(X_samples).sum() for sampler in self.samplers]
        #     best_idx = np.argmax(score_list)
        #     best_sampler = self.samplers[best_idx]
        #     Z_t, Z_t_label = best_sampler.sample(self.generate_num)
        #     x_range = [self.domain.x_min, self.domain.x_max]
        #     y_range = [self.domain.y_min, self.domain.y_max]
        #     X_new = select_point(x_range, y_range, Z_t)
        #     while len(X_new) < self.generate_num // 2:
        #         Z_t, Z_t_label = best_sampler.sample(self.generate_num//2)
        #         Z_t = select_point(x_range, y_range, Z_t)
        #         X_new = np.vstack((X_new, select_point(x_range, y_range, Z_t)))

        #     self.data_update(add_point=np.hstack((X_new, t_array[i] * np.ones((len(X_new), 1)))))
        #     print('add {0} points, now there are {1} points'.format(len(X_new), len(self.X)))
        #     return X_new
        # else:
        #     return None

        # using das 

        X_new = self.das(i, t_array, res_vector)

        return X_new
    
    def das(self,i, t_array, res_vector):
        # sample datas in each time step, we firstly use system_sample then use Gaussian Mixture
        idx_t = torch.where(self.X[:, -1] == t_array[i])[0]
        X_t = self.X[idx_t, 0:2]  # only sample (x,y)
        residual_t = res_vector[idx_t].abs().detach()

        if residual_t.max() - residual_t.min() < 1e-3:
            print('residual_max - res_min = {}, residual is uniform'.format(residual_t.max() - residual_t.min()))
            return None
        
        device = torch.device('cuda:' + str(self.device_ids[0]))
        # model
        # Define the prior distribution, usually diagonal Gaussian, this distribution takes cpu tensors as inputs and outputs
        p_z0 = DiagGaussian(torch.tensor([0.0, 0.0]), torch.tensor([[1., 0.0], [0.0, 1.]])) 
        # initialize a KRnet, a normalizing flow model
        args = self.args
        flow = KRnet(p_z0, args.input_size, args.n_step, args.n_depth, args.width, 2, device=device).to(device=device)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

        # loss_meter = RunningAverageMeter()
        scaled_res = residual_t/residual_t.sum()
        x = X_t.detach() # must detach data,  
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            log_px = flow.log_prob(x)
            loss = - torch.mean(scaled_res / torch.exp(log_px) * log_px)
            # loss = -log_px.mean()
            loss.backward()
            # nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()

            # loss_meter.update(loss.item())
            if itr%100==0:
                lr_scheduler.step()
        
        z_t_samples = flow.sample(self.generate_num)
        x_range = [self.domain.x_min, self.domain.x_max]
        y_range = [self.domain.y_min, self.domain.y_max]
        X_new = select_point(x_range, y_range, z_t_samples)
        while len(X_new) < self.generate_num // 2:
            z_t_samples = flow.sample(self.generate_num)
            z_t_samples = select_point(x_range, y_range, z_t_samples)
            X_new = np.vstack((X_new, select_point(x_range, y_range, z_t_samples)))

        self.data_update(add_point=np.hstack((X_new, t_array[i] * np.ones((len(X_new), 1)))))
        print('add {0} points, now there are {1} points'.format(len(X_new), len(self.X)))

        return X_new

    




def pde_residual(X: torch.Tensor, output: torch.Tensor, k) -> torch.Tensor:
    """
    compute pde residual function, which will be use in the PINN class, it should be rewritten for a new problem.
    :param k: params
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

    return k.flatten() * (p_xx + p_yy) - p_t


devices = check_devices()

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--n_step', type=int, default=1)  # 1
parser.add_argument('--n_depth', type=int, default=8)
parser.add_argument('--width', type=int, default=24)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


nx, ny, nt = 31, 31, 11
shape = (nx, ny)
domain = Square(nx=nx, ny=ny)
Time_space = TimeSpaceDomain(0, 10, nt=nt, nx=nx, ny=ny, cond='2points', bound_val=[0, 10], init_val=5)
Time_space.set_permeability(k=[3, 9, 15, 20])
visualize_k(Time_space)
layer_size = [3, 50, 100, 200, 100, 50, 1]

model = PINN(solver_layers=layer_size, domain=Time_space,
             device_ids=[0], log_dir='../logs/das_sample_littile', pde_func=pde_residual, lr=3e-3,
             args=args)

# model = PINN(solver_layers=layer_size, domain=Time_space,
#              device_ids=[0], log_dir='../logs/exp_not-add-point', pde_func=pde_residual, lr=1e-2)
#
print('model original data nums:{}'.format(len(model.original_X)))
model.train_solver(max_iter=1000, interval=100)
print('model last data nums:{}'.format(len(model.X)))