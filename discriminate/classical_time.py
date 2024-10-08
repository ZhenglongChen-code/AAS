from scipy.sparse.linalg import cg
from tqdm import tqdm
import time
import random
import numpy as np
from src.Mesh import *

def compute_press_t(cell_list, nx, ny):
    def compute_press(cell_list, nx, ny):
        # 矩阵方程Ax = b系数矩阵初始化
        row2cell = list(range(1, nx * ny - 1))  # 待求cell的编号,因为已知两个cell的 Dirichlet 条件，所以现在只有 nx*ny -2
        nrow = nx * ny - 2  # -2
        A = np.zeros((nrow, nrow))
        b = np.zeros(nrow)
        for i in range(nrow):
            cell_i = cell_list[row2cell[i]]
            neighbor = cell_i.neighbors  # 当前cell的邻接cell编号
            A[i, i] = - sum(cell_i.trans[:4])
            for j in range(4):
                if neighbor[j] != -1:
                    # 加这个判断的原因是如果这个方向没有邻接cell  cell_list[-1]能被索引
                    cell_j = cell_list[neighbor[j]]
                else:
                    continue
                if cell_j.markbc == 1:
                    b[i] = - cell_i.trans[j] * cell_j.press

                if neighbor[j] in row2cell:
                    A[i, neighbor[j] - 1] = cell_i.trans[j]

        press, exit_code = cg(A, b, x0=None, tol=1e-10)
        press = np.insert(press, [0], cell_list[0].press)  # p_0
        press = np.append(press, cell_list[-1].press)  # p_{-1}
        return press


kl = [1, 3, 5, 6, 8]
dist = [[1, 3], [5, 19]]  # 均匀分布的参数,
dist_dim = len(dist)  # 有几个均匀分布
full_set = []  # 单元组全集
start_time = time.time()
while len(full_set) < 200:  # 100个训练，100个测试
    # tqdm.write('current processing: {:.2f}%'.format(len(full_set)/5e3 * 100))
    test_k = set()  # 使用集合防止5列数据重复, 如果实际的均匀分布参数没有重合部分可以不写这一步。
    while len(test_k) < dist_dim:
        for dis_u in dist:
            rv = random.uniform(dis_u[0], dis_u[1])
            test_k.add(rv)
    test_k = list(test_k)
    if test_k not in full_set:
        full_set.append(test_k)
end_time = time.time()
print("create permeability set cost {:.5f} s".format(end_time - start_time))

nx, ny, nz = 10, 10, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
p_init, p_bc = 30 * 1e+6, 20 * 1e+6
all_press = []  # 存放所有的Press
all_permeability = []
for k in tqdm(full_set, desc='loading permeability'):
    perme = np.zeros((nx, ny))
    for i in range(dist_dim):
        ra = nx//dist_dim
        perme[:, list(range(ra*i, ra*(i+1)))] = k[i] * 1e-15  # mD 单位换算

    # generate 10x10 meshgrid
    mesh = MeshGrid(nx, ny, nz, perme.flatten())
    cell_list = mesh.cell_list

    # boundary condition
    cell_list[0].markbc = 1  # Dirichlet BC
    cell_list[0].press = p_bc
    cell_list[-1].markbc = 1
    cell_list[-1].press = p_bc / 2
    press = compute_press(cell_list, nx, ny)
    all_press.append(press)
    all_permeability.append(perme.reshape(1, -1))

all_press = np.array(all_press)
all_permeability = np.array(all_permeability)
train_press, test_press = all_press[:100], all_press[100:]
train_permeability, test_permeability = all_permeability[:100], all_permeability[100:]

# save data
np.savez('../dataset/train_data_10x10.npz', train_press=train_press, train_permeability=train_permeability)
np.savez('../dataset/test_data_10x10.npz', test_press=test_press, test_permeability=test_permeability)