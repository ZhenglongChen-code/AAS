import torch
import torchvision
import torch.utils.data
import math
import numpy as np
import copy
import time
from itertools import combinations

nrealall=1100
nreal=1000
ncell=400
nt=60
# For inputs
f=open('allperms1100.txt','r')
data = f.readlines()  # 将txt中所有字符串读入data
datavecall=list(map(float, data))
f.close()
datavecall=np.array(datavecall)
datavec=np.zeros((nrealall, ncell))
for ie in range(ncell):
    for ir in range(nrealall):
        datavec[ir, ie]=datavecall[ie*nrealall+ir]*0.1
f.close()
permvec=datavec
f=open('qwtall1100dt10k.txt','r')
data = f.readlines()
qwtall = list(map(float, data))
qwtall=np.array(qwtall)
qwtvec=np.zeros((nrealall,nt))
for ir in range(nrealall):
    for ie in range(nt):
        qwtvec[ir, ie] = qwtall[ir * 180 + ie*3]*10000

f = open('perms-trainset.txt','w')
for ig in range(nreal):
    for ie in range(ncell):
        f.write("%e " % (permvec[ig,ie]))
    f.write("\n")

f = open('qwts-trainset.txt','w')
for ig in range(nreal):
    for ie in range(60):
        f.write("%e " % (qwtvec[ig, ie]))
    f.write('\n')

f = open('perms-testset.txt','w')
for ig in range(nreal,nrealall):
    for ie in range(ncell):
        f.write("%e " % (permvec[ig,ie]))
    f.write("\n")

f = open('qwts-testset.txt','w')
for ig in range(nreal,nrealall):
    for ie in range(60):
        f.write("%e " % (qwtvec[ig, ie]))
    f.write('\n')








