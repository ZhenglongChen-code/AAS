
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

print("define properties and random parameter")
ndim=10
minvec=np.zeros(10)
maxvec=np.zeros(10)
for i in range(10):
    minvec[i]=(i+1)*1.0
    maxvec[i] = (i + 1) * 5.0

minvec2=maxvec*2.0
maxvec2=maxvec*3.0

samplelist = np.random.uniform(low=0, high=1, size=(32768,10))
nsamples=len(samplelist)
for i in range(nsamples):
    r2 = random.uniform(0, 1)
    if r2 < 0.5:
        samplelist[i] = samplelist[i] * (maxvec - minvec) + minvec
    else:
        samplelist[i] = samplelist[i] * (maxvec2 - minvec2) + minvec2


f = open('allsamples10d-mc32768.txt','w')
for isample in range(nsamples):
    for i in range(ndim):
        f.write("%e " % (samplelist[isample,i]))
    f.write("\n")







