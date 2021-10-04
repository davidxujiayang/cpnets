import scipy.io
import numpy as np

mat = scipy.io.loadmat(f'data/hires.mat')
data_dns = mat['udns_matrix']
nt = data_dns.shape[0]
nx = 32
ratio = int(data_dns.shape[1]/nx)

data = np.zeros((nt, nx))
for i in range(nx):
    data[:, i] = np.mean(data_dns[:, i*ratio:(i+1)*ratio], axis=1)

np.save(f'data/lowres.npy', data)

mat = scipy.io.loadmat(f'data/hires_newic.mat')
data_dns = mat['udns_matrix']
nt = data_dns.shape[0]
ratio = int(data_dns.shape[1]/nx)

data = np.zeros((nt, nx))
for i in range(nx):
    data[:, i] = np.mean(data_dns[:, i*ratio:(i+1)*ratio], axis=1)

np.save(f'data/lowres_newic.npy', data)
