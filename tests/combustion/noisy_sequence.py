from tensorflow.keras.utils import Sequence
import numpy as np
import random
from copy import deepcopy
import math

def flip_rotate(f, idx_x, flip=False, rad=0):
    f = deepcopy(f)
    if flip:
        f[..., idx_x] = -f[...,idx_x]
    fnew = deepcopy(f)

    fnew[..., idx_x] = math.cos(rad)*f[...,idx_x] - math.sin(rad) * f[...,idx_x+1] 
    fnew[..., idx_x+1] = math.sin(rad)*f[...,idx_x] + math.cos(rad) * f[...,idx_x+1] 
    return fnew

class NoisySequence(Sequence):
    def __init__(self, nf, ef, y, nstd, res_norm, shuffle, batch_size=1, aug=False):
        self.nf = nf
        self.ef = ef
        self.y = y
        
        self.nstd = nstd
        self.res_norm = res_norm
        self.data_size = nf.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.p = np.arange(self.data_size)
        self.aug = aug
        
    def __len__(self):
        if self.shuffle:
            self.p = np.random.permutation(self.data_size)
        return int(np.ceil(self.data_size / self.batch_size))
    
    def __getitem__(self, batch_num):
        if batch_num >= self.__len__():
            raise IndexError(
                "Mapper: batch_num larger than number of esstaimted  batches for this epoch."
            )
        idx = self.p[batch_num:batch_num+self.batch_size]
        noise = np.random.normal(loc=0.0, scale=self.nstd, size=self.nf[idx,].shape)
        
        nf = self.nf[idx,] + noise
        ef = self.ef[idx,]
        y = self.y[idx,] - noise/self.res_norm

        if self.aug:
            flip = random.uniform(0, 1)>0.5
            rad = random.uniform(0, 2*math.pi)
            nf = flip_rotate(nf, idx_x=1, rad=rad, flip=flip)
            y = flip_rotate(y, idx_x=1, rad=rad, flip=flip)
            ef = flip_rotate(ef, idx_x=0, rad=rad, flip=flip)

        
        return [nf, ef], y
        
    def on_epoch_end(self):
        if self.shuffle:
            self.p = np.random.permutation(self.data_size)
