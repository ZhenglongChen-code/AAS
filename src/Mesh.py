# this is a module including some basic class to generate mesh_grids

import math

import numpy as np
import torch
from scipy.sparse.linalg import cg
from torch.utils.data import DataLoader


class Node:
    def __init__(self):
        self.coord = [0, 0, 0]  # node coordinate (x, y, z)


class Cell:
    def __init__(self):
        self.coord = [0, 0, 0]  # cell center coordinate (x_c, y_c, z_c)
        self.vertices = [-1 for i in range(8)]  # cell 顶点的编号
        self.neighbors = [-1 for i in range(6)]  # 相邻cell的编号. 依次左右前后上下(以x轴向右, y轴向前, z轴向下建立坐标系)
        self.dx = 0  # 每个cell 立方体的长宽高
        self.dy = 0
        self.dz = 0
        self.volume = 0  # cell的体积
        self.porosity = 0  # 当前cell的孔隙率
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0 for i in range(6)]  # # 6个邻接cell的T_ij
        self.markbc = 0  # boundary condition mark, if this cell is boundary cell, set it to 1
        self.markwell = 0  # judge weather it is a well
        self.press = 0


# MeshGrid's initialization need many physical parameters, you can change these parameters in the class directly so that
# you don't need to impart too many parameters to the class init function
class MeshGrid(Node, Cell):
    def __init__(self, nx, ny, nz, permeability,
                 mu_o=2e-3, ct=5e-8, porosity=0.1, p_init=30.0*1e6, p_bc=20.0*1e6,
                 bhp_constant=0):
        # Define cell num and Length of side
        self.ls_x, self.ls_y, self.ls_z = 5, 5, 5  # x, y, z 方向的长宽高
        self.node_list = []  # 存放节点, 按编号索引
        self.cell_list = []
        self.cell_volume = None
        self.ncell = None
        self.nx, self.ny, self.nz = nx, ny, nz

        # well coondition
        self.mu_o = mu_o  # 2e-3 参数
        self.ct = ct
        self.porosity = porosity

        # generate nodes and cell
        self.generate_node()
        # print('nodes generate successfully')
        self.generate_cell(permeability)
        # print('cells generate successfully')

        # compute transmissibility(Tij) for every cell
        self.trans_matrix = []
        self.neighbor_vectors = []
        self.compute_trans_matrix()

        # compute delta P in [w e n s] directions
        # other well conditions
        self.rw = 0.05
        self.SS = 3  # 论文里的S
        self.length = 3000
        self.cs = self.ct * 3.14 * self.rw ** 2 * self.length
        self.bhp_constant = bhp_constant
        self.ddx = 10
        self.ddy = 10
        self.re = 0.14 * (self.ddx ** 2 + self.ddy ** 2) ** 0.5
        # PI 分子还要乘以渗透率场K
        self.PI = - torch.tensor(2 * torch.pi * self.ddx * 2.5e-15 / self.mu_o / (math.log(self.re / self.rw)
                                                                                + self.SS)).cuda()
        # bottom-hole pressure
        self.pwf = torch.tensor(self.bhp_constant).cuda()
        # boundary condition
        self.cell_list[0].markwell = 1
        # initial condition
        self.p_init = torch.tensor(p_init).cuda()
        self.p_bc = torch.tensor(p_bc).cuda()
        self.press = torch.ones(len(self.cell_list)).cuda() * self.p_init
        self.press[0] = self.p_bc  #
        self.q = (self.press[0] - self.pwf) * self.PI
        self.sum_delta_p = None  # 计算公式里面的 \sum_j  T_{ij} * (p_j -p_i)
        self.compute_sum_delta_p()

    def update_data(self, press_new):
        """update self.press and sum_delta_p and q"""
        # 注意不能更新边界条件
        self.press = press_new
        self.compute_sum_delta_p()
        self.press[0] = self.p_bc  # 先计算sum_delta_p 再把边界条件赋值上
        # self.q = (self.press[0] - self.pwf) * self.PI

    def generate_node(self):
        """
        :param nx, ny, nz:  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
        :return: node_list
        """
        # current_x, current_y, current_z :当前node的(x,y,z)坐标值
        for i in range(self.nz + 1):
            current_z = i * self.ls_z
            for j in range(self.ny + 1):
                current_y = j * self.ls_y
                for k in range(self.nx + 1):
                    current_x = k * self.ls_x
                    node = Node()
                    node.coord = [current_x, current_y, current_z]
                    self.node_list.append(node)

    def generate_cell(self, permeability):
        """  build connectivity and neighbors, and store permeability of each cell
        :param nx, ny, nz:  # x, y, z 方向的cell数目
        :param permeability: cell单元的渗透率
        :return: cell_list
        """
        for i in range(self.nz):
            for j in range(self.ny):
                for k in range(self.nx):
                    cell = Cell()
                    idx = i * self.nx * self.ny + j * self.nx + k  # number every cell

                    # Record 6 neighbor cells idx number
                    cell.neighbors[0] = idx - 1 if k > 0 else cell.neighbors[
                        0]  # 如果x方向k>0, 也就是x减少方向还有cell,记录 y,z 不动，x 减少方向标记为0
                    cell.neighbors[1] = idx + 1 if k < self.nx - 1 else cell.neighbors[
                        1]  # 如果x增大方向还有cell, 记录 y,z不动，x 增大方向cell标记为1
                    cell.neighbors[2] = idx - self.nx if j > 0 else cell.neighbors[2]
                    cell.neighbors[3] = idx + self.nx if j < self.ny - 1 else cell.neighbors[3]
                    cell.neighbors[4] = idx - self.nx * self.ny if i > 0 else cell.neighbors[4]
                    cell.neighbors[5] = idx + self.nx * self.ny if i < self.nz - 1 else cell.neighbors[5]

                    # Record the 8 vertices of current cell
                    i0 = i * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + k  # cell左上方的顶点编号
                    i1 = i0 + 1
                    i2 = i0 + (self.nx + 1)
                    i3 = i2 + 1

                    i4 = i0 + (self.nx + 1) * (self.ny + 1)
                    i5 = i4 + 1
                    i6 = i4 + (self.nx + 1)
                    i7 = i6 + 1
                    cell.vertices = [i0, i1, i2, i3, i4, i5, i6, i7]

                    cell.dx = self.node_list[i1].coord[0] - self.node_list[i0].coord[0]
                    cell.dy = self.node_list[i2].coord[1] - self.node_list[i0].coord[1]
                    cell.dz = self.node_list[i4].coord[2] - self.node_list[i0].coord[2]

                    # compute cell center coordinate and volume
                    temp = [self.node_list[i].coord for i in cell.vertices]
                    cell.coord = np.mean(temp, axis=0)
                    cell.volume = cell.dx * cell.dy * cell.dz

                    # add the cell to cell_list
                    self.cell_list.append(cell)

        self.cell_volume = self.cell_list[0].volume
        self.ncell = len(self.cell_list)

        # print("try to define properties")
        # mu_o = 2e-3
        # ct = 5e-8
        # porosity = 0.1
        for i in range(self.ncell):
            self.cell_list[i].porosity = self.porosity
            self.cell_list[i].kx, self.cell_list[i].ky = permeability[i], permeability[i]  # 这里设定cell沿x,y,z方向渗透率k相同

    def compute_trans_matrix(self):
        """
        use permeability to  compute Tij for cell_list[i]
        :return: self.trans_matrix; a (4, len(cell_list)) dim matrix,
        [ [Tij_w]; [Tij_e]; [Tij_n]; [Tij_s] ], [Tij_w] : (1,400) tensor
        neighbor_vector: neighbor cell of cell_list[i] in each direction.
        """
        # trans_matrix = torch.zeros((4, self.nx * self.ny)).cuda()  # 最终返回的tensor
        neighbor_vector = np.zeros((4, self.nx * self.ny)).tolist()
        # neighbor_w, neighbor_e, neighbor_n, neighbor_s
        # = neighbor_vector[0], neighbor_vector[1], neighbor_vector[2], neighbor_vector[3]
        for i in range(self.ncell):
            current_cell = self.cell_list[i]
            dxi, dyi, dzi = current_cell.dx, current_cell.dy, current_cell.dz  # cell 边长
            current_cell_trans_matrix = []  # 当前cell的w e n s四个方向Tij 数组

            for j in range(4):  # 不考虑z轴方向的邻接cell
                neighbor = current_cell.neighbors[j]

                if neighbor < 0:  # 当前cell j方向没有邻接cell
                    current_cell.trans[j] = 0
                    current_cell_trans_matrix.append(0)
                    # 当前cell j方向没有邻接cell, 但是防止等会索引Press 矩阵出错, 将这个邻接cell编号设成0，
                    # 但是不影响后面计算 Tij * delta_p, 因为没有邻接cell处的Tij = 0
                    neighbor_vector[j][i] = int(0)
                else:
                    neighbor_vector[j][i] = neighbor
                    dxj, dyj, dzj = self.cell_list[neighbor].dx, self.cell_list[neighbor].dy, self.cell_list[
                        neighbor].dz
                    if j <= 1:  # j=0 或j=1, x轴方向, 截面面积为A = dyi * dzi
                        Ti = current_cell.kx * dyi * dzi / (dxi / 2 * self.mu_o)
                        # neighbor cell is self.cell_list[neighbor]
                        Tj = self.cell_list[neighbor].kx * dyj * dzj / (dxj / 2 * self.mu_o)

                    elif j >= 2:  # j=2, j=3, y轴方向, 截面面积A = dxi * dzi

                        Ti = current_cell.kx * dxi * dzi / (dyj / 2 * self.mu_o)
                        Tj = self.cell_list[neighbor].kx * dxj * dzj / (dyj / 2 * self.mu_o)

                    Tij = 1 / (1 / Ti + 1 / Tj)
                    current_cell.trans[j] = Tij
                    current_cell_trans_matrix.append(Tij)

            self.trans_matrix.append(current_cell_trans_matrix)  # 注意这个时候的trans_matrix是(len(cell_list), 4) 数组

        self.trans_matrix = torch.tensor(self.trans_matrix).T.cuda()
        self.neighbor_vectors = neighbor_vector  # [4, nx*ny] shape matrix, each colum represent 4 neighbor cell index.
        # print('self.trans_matrix has been computed, shape is :{}'.format(self.trans_matrix.shape))

    def compute_sum_delta_p(self):
        """
        compute w,e,n,s 4 directions delta p, neighbor_w is an idx array. suppose P is pressure tensor ,
        delta_p_w = P[neighbor_w] - P
        :return:a 1_D matrix sum_delta_p . delta_p = [[delta_p_w]; [delta_p_e]; [delta_p_n]; [delta_p_s]].
        then \sum_j T_{ij}*(p_j - p_i) = sum(trans_matrix * delta_p , axis = 0)  and its shape is (1, len(cell_list))
        """
        # neighbor_w, neighbor_e, neighbor_n, neighbor_s = self.neighbor_vectors[0], ...[1], ...[2], ...[3]
        delta_p = torch.zeros((4, self.ncell)).cuda()  # same shape with self.trans_matrix
        for c in range(4):
            delta_p[c, :] = (self.press[self.neighbor_vectors[c]] - self.press) * self.trans_matrix[c]
        self.sum_delta_p = torch.sum(delta_p, dim=0)  # 不要忘记\laplace_p 与 trans_matrix 做点乘

    def solve_press(self):
        # 矩阵方程Ax = b系数矩阵初始化
        row2cell = list(range(1, self.nx * self.ny - 1))  # 待求cell的编号,因为已知两个cell的 Dirichlet 条件，所以现在只有
        nrow = self.nx * self.ny - 2
        A = np.zeros((nrow, nrow))
        b = np.zeros(nrow)
        for i in range(nrow):
            cell_i = self.cell_list[row2cell[i]]
            neighbor = cell_i.neighbors  # 当前cell的邻接cell编号
            A[i, i] = - sum(cell_i.trans[:4])
            for j in range(4):
                if neighbor[j] != -1:
                    # 加这个判断的原因是如果这个方向没有邻接cell  cell_list[-1]能被索引
                    cell_j = self.cell_list[neighbor[j]]
                else:
                    continue

                if cell_j.markbc == 1:
                    b[i] = - cell_i.trans[j] * cell_j.press

                if neighbor[j] in row2cell:
                    A[i, neighbor[j] - 1] = cell_i.trans[j]

        press, exit_code = cg(A, b, x0=None, tol=1e-10)
        press = np.insert(press, [0], self.cell_list[0].press)  # p_0
        press = np.append(press, self.cell_list[-1].press)  # p_{-1}
        return press


def set_batch(in_array, batch_size):
    """
    this function aims to set n samples to m little batch, m = int(n/batch_size) + 1
    :param in_array: list or array type samples,
    :param batch_size:
    :return: loader1 with higher dim array  (batch_size, channels=1, w=20, h=20)  and loader2: (batch_size, 400)
            each element in loader1 is a 2_D tensor, while in loader2 is a 1_D tensor
    """
    if not isinstance(in_array, torch.Tensor):
        data = torch.tensor(in_array, dtype=torch.float32).cuda()
    else:
        data = in_array
    fig_tensor = data.view(-1, 1, 20, 20)
    loader1 = DataLoader(dataset=fig_tensor, batch_size=batch_size, shuffle=False, num_workers=0)
    loader2 = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader1, loader2
