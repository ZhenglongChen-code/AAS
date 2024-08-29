import os
import sys
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import *
from src.Net_structure import *

nx, ny, nz = 10, 10, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
train_press, train_permeability = np.load('../dataset/train_data_10x10.npz')
test_press, test_permeability = np.load('../dataset/test_data_10x10.npz')

# init and condition
p_init, p_bc = 30 * 1e+6, 20 * 1e+6


b1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
# b5 = nn.Sequential(*resnet_block(40,80,2,first_block=False))

input_size = nx * ny * nz  # 输入为单元渗透率或trans，后者更好
# 实例化模型
model = nn.Sequential(b1, b2, b3, b4,
                      nn.Flatten(), nn.Linear(40 * 5 * 5, input_size))