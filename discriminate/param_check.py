import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
import numpy as np

with open('../dataset/samplesperm.txt') as f:
    data = f.readlines()
    permeability = list(map(float, data))


############

import torch
import torch.utils.data


# aa=torch.tensor([1, 2, 3, 4])
# bb=aa.reshape(2,2)
# f=open('qwt_1000h_dt10_bhp.txt','r')
# data = f.readlines()  # 将txt中所有字符串读入data
# qwt0 = list(map(float, data))
# qwt0=torch.tensor(qwt0).cuda()
f=open('../dataset/samplesperm.txt','r')
data = f.readlines()  # 将txt中所有字符串读入data
permvec = list(map(float, data))
f.close()
for i in range(400):
    # if permvec[i]>20:
    #     permvec[i]=20e-15
    # elif permvec[i]<1:
    #     permvec[i]=1e-15
    # else:
    permvec[i]=permvec[i]*1e-15
# rewrite FD codes
class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self): # 不加self就变成了对所有类对象同时更改
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.markbc = 0
        self.press = 0


print("define properties")
mu_o = 2e-3
ct = 5e-8
p_init=10.0*1e6
nt = 180
dt=10000
chuk = 5e-15
poro = 0.1

bhp_constant = 0

########using Meshgrida class ######
nx, ny, nz = 20, 20, 1
permeability = np.array(permeability)* 1e-15
mesh = MeshGrid(nx, ny, nz, permeability, mu_o=mu_o, p_init=p_init, p_bc=bhp_constant)
batch_size = 1
trans_matrix = mesh.trans_matrix
neighbor_idx = mesh.neighbor_vectors

def solve_func(p_last, trans_matrix, neighbor_idx):
    """
    :param p_last: scaled pressure, p_scaled = p / p_init
    :param trans_matrix:
    :param neighbor_idx:
    :return:
    """
    p_input = p_last.reshape(batch_size, nz, nx, ny)
    p_next = model(p_input)[0]  # 先测试单条数据
    # p_next == outputs, (p_next - p_last) * p_init 与函数中的(u[:] - press1[:]*p_init) 不同, 有计算精度误差
    res = mesh.cell_volume * mesh.porosity * mesh.ct / dt * (p_next * p_init - p_last * p_init)
    for d in range(4):
        res -= trans_matrix[d] * (p_next[neighbor_idx[d]] * p_init - p_next * p_init)
    res[0] += mesh.PI * (p_next[0]*p_init - mesh.pwf)
    # loss = criterion(res, torch.zeros_like(res))
    return res, p_next
#################

print("build Grid")
dxvec=[0]
for i in range(0, 20):
    dxvec.append(5)

dyvec=[0]
for i in range(0, 20):
    dyvec.append(5)
dzvec=[0,5]


nx=len(dxvec)-1
ny=len(dyvec)-1
nz=len(dzvec)-1
nodelist=[]
llz = 0
for k in range(0, nz+1):
    llz = llz + dzvec[k]
    lly=0
    for j in range(0, ny+1):
        lly = lly + dyvec[j]
        llx = 0
        for i in range(0, nx+1):
            llx = llx + dxvec[i]
            node=NODE()
            node.x=llx
            node.y=lly
            node.z=llz
            nodelist.append(node)

# build connectivity and neighbors
celllist=[]

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            if i>0:
                cell.neighbors[0] = nc - 1
            if i<nx-1:
                cell.neighbors[1] = nc + 1
            if j>0:
                cell.neighbors[2] = nc - nx
            if j<ny-1:
                cell.neighbors[3] = nc + nx
            if k>0:
                cell.neighbors[4] = nc - nx*ny
            if k<nz-1:
                cell.neighbors[5] = nc + nx * ny
            i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            cell.dx = nodelist[i1].x - nodelist[i0].x
            cell.dy = nodelist[i2].y - nodelist[i0].y
            cell.dz = nodelist[i4].z - nodelist[i0].z
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7
            cell.xc = 0.125 * (nodelist[i0].x+nodelist[i1].x+nodelist[i2].x+nodelist[i3].x+nodelist[i4].x+nodelist[i5].x+nodelist[i6].x+nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
            cell.volume=cell.dx*cell.dy*cell.dz
            celllist.append(cell)

cellvolume=celllist[0].volume

ncell=len(celllist)
print("set properties to grid")
for i in range(0, ncell):
    celllist[i].porosity=poro
    celllist[i].kx = permvec[i]
    celllist[i].ky = permvec[i]
    celllist[i].kz = permvec[i]

print("compute transmissibility")
transvec = []  # for input into network
for ie in range(0, ncell):
    dx1=celllist[ie].dx
    dy1=celllist[ie].dy
    dz1=celllist[ie].dz
    for j in range(0, 4):
        je=celllist[ie].neighbors[j]
        if je>=0:
            dx2 = celllist[je].dx
            dy2 = celllist[je].dy
            dz2 = celllist[je].dz
            mt1=1.0/mu_o
            mt2=1.0/mu_o
            if j==0 or j == 1:
                mt1 = mt1 * dy1 * dz1
                mt2 = mt2 * dy2 * dz2
                k1 = celllist[ie].kx
                k2 = celllist[je].kx
                dd1=dx1/2.
                dd2=dx2/2.
            elif j==2 or j==3:
                mt1 = mt1 * dx1 * dz1
                mt2 = mt2 * dx2 * dz2
                k1 = celllist[ie].ky
                k2 = celllist[je].ky
                dd1 = dy1 / 2.
                dd2 = dy2 / 2.
            # elif j == 4 or j == 5:
            #     mt1 = mt1 * dx1 * dy1
            #     mt2 = mt2 * dx2 * dy2
            #     k1 = celllist[ie].kz
            #     k2 = celllist[je].kz
            #     dd1 = dz1 / 2.
            #     dd2 = dz2 / 2.
            t1 = mt1 * k1 / dd1
            t2 = mt2 * k2 / dd2
            tt = 1 / (1 / t1 + 1 / t2)
            celllist[ie].trans[j]=tt
            transvec.append(tt)
        else:
            transvec.append(0)

print("define pde")
# define pde and bc

transvec_n = torch.zeros(ncell).cuda()
transvec_s = torch.zeros(ncell).cuda()
transvec_e = torch.zeros(ncell).cuda()
transvec_w = torch.zeros(ncell).cuda()
neiborvec_w = torch.zeros(ncell).type(torch.long).cuda()
neiborvec_e = torch.zeros(ncell).type(torch.long).cuda()
neiborvec_s = torch.zeros(ncell).type(torch.long).cuda()
neiborvec_n = torch.zeros(ncell).type(torch.long).cuda()

for ie in range(ncell):
    neibor_w = celllist[ie].neighbors[0]
    neibor_e = celllist[ie].neighbors[1]
    neibor_s = celllist[ie].neighbors[2]
    neibor_n = celllist[ie].neighbors[3]
    if neibor_w < 0:
        neibor_w = 0
    if neibor_e < 0:
        neibor_e = 0
    if neibor_s < 0:
        neibor_s = 0
    if neibor_n < 0:
        neibor_n = 0
    neiborvec_w[ie] = neibor_w
    neiborvec_e[ie] = neibor_e
    neiborvec_n[ie] = neibor_n
    neiborvec_s[ie] = neibor_s
    transvec_w[ie] = celllist[ie].trans[0]
    transvec_e[ie] = celllist[ie].trans[1]
    transvec_s[ie] = celllist[ie].trans[2]
    transvec_n[ie] = celllist[ie].trans[3]

ddx=5

pwf = bhp_constant
# weightvec = torch.zeros(ncell).cuda()
# for ie in range(ncell):
#     rr = ((100 - celllist[ie].xc) ** 2 + (100 - celllist[ie].yc) ** 2)
#     weightvec[ie] = rr
def diffusionExplicit(u):  #(cells)  Explicit
    u = u*p_init  # u: scaled output pressure
    press2 = press1 * p_init
    diffu = torch.zeros_like(u).cuda()
    diffu[:] = diffu[:] - transvec_w[:] * (press2[neiborvec_w[:]] - press2[:])
    diffu[:] = diffu[:] - transvec_e[:] * (press2[neiborvec_e[:]] - press2[:])
    diffu[:] = diffu[:] - transvec_s[:] * (press2[neiborvec_s[:]] - press2[:])
    diffu[:] = diffu[:] - transvec_n[:] * (press2[neiborvec_n[:]] - press2[:])
    diffu[:] = diffu[:] + (u[:] - press2[:]) * poro * cellvolume * ct / dt
    diffu[0] = u[0]-pwf
    return diffu



def diffusionImplicit(u):  #(cells)  Implicit
    u=u*p_init
    diffu=torch.zeros_like(u).cuda()
    diffu[:] = diffu[:] - transvec_w[:] * (u[neiborvec_w[:]] - u[:])
    diffu[:] = diffu[:] - transvec_e[:] * (u[neiborvec_e[:]] - u[:])
    diffu[:] = diffu[:] - transvec_s[:] * (u[neiborvec_s[:]] - u[:])
    diffu[:] = diffu[:] - transvec_n[:] * (u[neiborvec_n[:]] - u[:])
    diffu[:] = diffu[:] + (u[:] - press1[:]*p_init) * poro * cellvolume * ct / dt
    # (u[:] - press1[:]*p_init) 与 p_next-p_last 不同
    # poro * cellvolume * ct / dt 与mesh中参数相同
    diffu[0] = diffu[0] + mesh.PI * (u[0]-pwf)
    return diffu



output_size = ncell

print("define NN model, criterion, optimizer and scheduler")


class CNN(torch.nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 100*3*20*20 -> 100*25*18*18: (100) 对每一个batch即time step (2) channel
            torch.nn.Conv2d(1, 25, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # # ->100*25*9*9
        )

        self.conv2 = torch.nn.Sequential( # ->100*50*7*7
            torch.nn.Conv2d(25, 50, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # ->100*50*3*3
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, output_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(600, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)   # x: tensor (time steps (100), 50*3*3)
        x = self.fc(x) # x: tensor (time steps (100), output size)
        return x



input_size = ncell  # 输入为单元渗透率或trans，后者更好
hidden_size = 600
# 实例化模型
model = CNN(output_size)
model.cuda()
# 设置损失函数和优化器
learning_rate = 0.001
criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.991)
f = open('loss1stephetehighly.txt','w')
print("训练模型")
num_epochs = 300
print("construct input tensor for NN")
inputtensor = torch.ones(1, 1, ny, nx).cuda()
# 前向传播计算模型预测值
pressout=torch.ones(ncell).cuda()
pressout[0]=pwf/p_init
presssave=torch.zeros(nt+1,ncell).cuda()
presssave[0] = pressout
######press_in
# press_input = mesh.press

nt=1
for t in range(nt):
    print(t)
    lowestloss = 10000000000
    press1 = pressout
    inputtensor[0][0] = pressout.reshape(ny, nx)
    # qwt[t]=PI*(press1[0]-pwf)
    for epoch in range(num_epochs):
        outputs = model(inputtensor)  # on gpu
        diff = diffusionImplicit(outputs[0])
        # 计算损失并利用反向传播计算损失对各参数梯度
        loss = criterion(diff, diff * 0)
        # loss = criterion(res, torch.zeros_like(res))
        f.write("%0.3f\n" % loss)
        pressnext = outputs[0].clone().detach()
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(torch.ones_like(loss))
        optimizer.step()
        # scheduler.step()
        if loss < lowestloss:
            lowestloss = loss
            pressout = pressnext

    print(lowestloss)
    presssave[t+1] = pressout


import matplotlib.pyplot as plt
plt.imshow(pressout.detach().cpu().numpy().reshape(nx, ny))
plt.savefig('chek.png')
plt.show()
ppp=pressout.cpu()*p_init
for ie in range(len(celllist)):
    celllist[ie].press=ppp[ie].detach().numpy()
