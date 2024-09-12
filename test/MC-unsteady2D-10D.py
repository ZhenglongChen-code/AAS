
import torch
import torchvision
import torch.utils.data
import math
import numpy as np
import time
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import cg
from scipy.stats import qmc
import random
import copy

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
        self.markwell = 0
        self.press = 0

class BLOCK: # grid cell in the parametric space
    def __init__(self):
        self.samples = []
        self.pvol=0

class SAMPLE:
    def __init__(self):
        self.value=np.zeros(2)
        self.p=0

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallingCounter
def singlephase_unsteady_imp(chukvec):
    for i in range(0, ncell):  # set chuk
        celllist[i].kx = chukvec[i]
        celllist[i].ky = chukvec[i]
        celllist[i].kz = chukvec[i]*0.1
    for ie in range(0, ncell):  # compute transmissibility
        dx1 = celllist[ie].dx
        dy1 = celllist[ie].dy
        dz1 = celllist[ie].dz
        for j in range(0, 4):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                dx2 = celllist[je].dx
                dy2 = celllist[je].dy
                dz2 = celllist[je].dz
                mt1 = 1.0 / mu_o
                mt2 = 1.0 / mu_o
                if j == 0 or j == 1:
                    mt1 = mt1 * dy1 * dz1
                    mt2 = mt2 * dy2 * dz2
                    k1 = celllist[ie].kx
                    k2 = celllist[je].kx
                    dd1 = dx1 / 2.
                    dd2 = dx2 / 2.
                elif j == 2 or j == 3:
                    mt1 = mt1 * dx1 * dz1
                    mt2 = mt2 * dx2 * dz2
                    k1 = celllist[ie].ky
                    k2 = celllist[je].ky
                    dd1 = dy1 / 2.
                    dd2 = dy2 / 2.
                elif j == 4 or j == 5:
                    mt1 = mt1 * dx1 * dy1
                    mt2 = mt2 * dx2 * dy2
                    k1 = celllist[ie].kz
                    k2 = celllist[je].kz
                    dd1 = dz1 / 2.
                    dd2 = dz2 / 2.
                t1 = mt1 * k1 / dd1
                t2 = mt2 * k2 / dd2
                tt = 1 / (1 / t1 + 1 / t2)
                celllist[ie].trans[j] = tt
    qwt=np.zeros(nt)   # record flow rate with time
    for i in range(0, ncell):  # initial condition
        celllist[i].press = p_init
    Acoef = np.zeros((ncell, ncell))
    RHS = np.zeros(ncell)
    for t in range(nt):
        for ie in range(ncell):
            icell = ie
            p_i = celllist[icell].press
            Acoef[ie, ie] = celllist[icell].porosity * ct * celllist[icell].volume / dt
            RHS[ie] = celllist[icell].porosity * ct * celllist[icell].volume / dt * p_i
            if celllist[icell].markwell > 0:
                qw = PI * (celllist[icell].press - pwf)
                RHS[ie] = RHS[ie] - qw
                qwt[t] = qw
            for j in range(6):
                je = celllist[icell].neighbors[j]
                if je >= 0:
                    Acoef[ie, je] = -celllist[ie].trans[j]
                    Acoef[ie, ie] = Acoef[ie, ie] + celllist[ie].trans[j]
        # press=np.dot(np.linalg.inv(Acoef),RHS)
        # press, exit_code=minres(Acoef, RHS, x0=None, shift=0.0, tol=1e-10, maxiter=None, M=None, callback=None, show=False, check=False)
        # endtime1 = time.time()
        press, exit_code = cg(Acoef, RHS, x0=None, tol=1e-05)
        # endtime = time.time()
        # print('iteration:', t, ' exit code: ', exit_code, ' time cost ', endtime - endtime1)
        for ie in range(ncell):
            icell = ie
            celllist[icell].press = press[ie]
            # if press[ie] < 0:
            #     print('negative press at ', icell, press[ie])
    return qwt

print("build Grid")
dxvec=[0]
for i in range(0, 20):
    dxvec.append(10)

dyvec=[0]
for i in range(0, 20):
    dyvec.append(10)
dzvec=[0,10]


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


print("define properties")
mu_o = 2e-3
ct = 5e-8
poro = 0.1
for i in range(0, ncell):
    celllist[i].porosity = poro
print("define well condition")
rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14 * rw * rw * length
ddx = 10
re = 0.14 * (ddx * ddx + ddx * ddx) ** 0.5
PI = 2 * 3.14 * ddx * 2.5e-15 / mu_o / (math.log(re / rw) + SS)
pwf = 20e6  # bottom-hole pressure
celllist[0].markwell = 1
# simulation settings same as the qwt_real which is synthetic
p_init=30.0*1e6
nt = 120
dt = 100000
nsamples=20000
print("read samples")
allperms = np.loadtxt('allsamples10d-mc32768.txt',skiprows=0)
allperms = allperms[:nsamples]
allperms=allperms*1e-15
allpermvec=np.zeros((nsamples, ncell))
for isample in range(nsamples):
    k10d = allperms[isample]
    for j in range(ny):
        for i in range(nx):
            id = j * nx + i
            ix = int(i / 2)
            allpermvec[isample, id] = k10d[ix]

print("start qmc")
allqwt=np.zeros((nsamples, nt))
starttime = time.time()
for isample in range(nsamples):
    print("simulating sample ", isample)
    allqwt[isample]=singlephase_unsteady_imp(allpermvec[isample])

endtime=time.time()
print('time cost: ',endtime-starttime)
g = open('timecost-mc20000.txt','w')
g.write("Time Cost: %e" % (endtime-starttime))
#output QoI
f = open('qwtofallsamples-mc20000.txt','w')
for i in range(nsamples):
    for j in range(nt):
        f.write("%e " % (allqwt[i,j]))
    f.write("\n")


print("output to vtk")
f = open('result_unsteady_2D_10D-mc20000.vtk','w')
f.write("# vtk DataFile Version 2.0\n")
f.write( "Unstructured Grid\n")
f.write( "ASCII\n")
f.write("DATASET UNSTRUCTURED_GRID\n")
f.write("POINTS %d double\n" % (len(nodelist)))
for i in range(0, len(nodelist)):
    f.write("%0.3f %0.3f %0.3f\n" % (nodelist[i].x, nodelist[i].y, nodelist[i].z))
f.write("\n")
f.write("CELLS %d %d\n" % (len(celllist), len(celllist)*9))
for i in range(0, len(celllist)):
    f.write("%d %d %d %d %d %d %d %d %d\n" % (8, celllist[i].vertices[0], celllist[i].vertices[1], celllist[i].vertices[3], celllist[i].vertices[2], celllist[i].vertices[4], celllist[i].vertices[5], celllist[i].vertices[7], celllist[i].vertices[6]))
f.write("\n")
f.write("CELL_TYPES %d\n" % (len(celllist)))
for i in range(0, len(celllist)):
    f.write("12\n")
f.write("\n")
f.write("CELL_DATA %d\n" % (len(celllist)))
f.write("SCALARS Permeability1_mD double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (celllist[i].kx))
f.write("SCALARS Permeability2-mD double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (allpermvec[11,i]*1e15))
f.write("SCALARS Pressure-MPa double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (celllist[i].press/1000000))
f.close()






