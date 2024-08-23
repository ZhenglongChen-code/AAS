import torch
import matplotlib
from torch import nn
import numpy as np


class FNN(nn.Module):
    def __init__(self, layer_size, activation=nn.Tanh):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # 数组长n，则网络的全连接层有n-1层

        for i in range(1, len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i - 1], layer_size[i]))
            self.layers.append(self.activation())

        # 这是最后一层输出层
        self.layers.append(nn.Linear(layer_size[-2], layer_size[-1]))

    def forward(self, inputs):
        X = inputs
        for layer in self.layers:
            X = layer(X)
        return X


class PointSet():
    def __init__(self, scaled_coord, size, region=[[0, 10], [0, 10]]):
        """
        :param scaled_coord: PINN output
        :param size: generated point shape
        :param region: [X_lim, Y_lim]
        """
        self.new_coord = None
        self.new_point = None
        self.region = np.array(region)
        # self.region = torch.tensor(region, dtype=torch.float32)
        # suppose coord data are scaled in [0, 1], and then restore these data
        self.coord = scaled_coord * (self.region[:, 1] - self.region[:, 0]).T + self.region[:, 0].T
        self.initial_point = np.random.Generator.uniform(self.region[:, 0].T, self.region[:, 1].T, size=size)

    def generate_point(self, generator):
        self.new_point = generator(self.initial_point)  # 这里希望生成器网络输出的新的点介于定义域内。但是很难控制，所以我们将输出的数据归一化后再计算坐标

        # scaled_point =
        self.new_coord = self.new_point * (self.region[:, 1] - self.region[:, 0]).T + self.region[:, 0].T
        idx = [True for i in range(len(self.new_coord))]
        for i, point in enumerate(self.new_coord):
            if any(point < self.region[:, 0].T) or any(point > self.region[:, 1].T):
                idx[i] = False
        for i in self.new_coord:
            if i < self.region[:, 0].T or i > self.region[:, 1].T:
                idx = np.delete(idx, np.where(idx == i))
        self.new_coord = self.new_coord[idx]
        return self.new_coord[idx]


# def pde(x, y):



def residual(net, in_put, devices):
    # 计算残差
    x = torch.tensor(in_put, dtype=torch.float32).to(devices)
    u = net(x)

    return x

# def train(model, data_loader, criterion, optimizer, device, epoches):
#     for epoch in epoches:
#         for data in data_loader:
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion()



model = FNN(layer_size=[2]+[20]*3+[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()

