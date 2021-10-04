import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append('../../src/')

from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MSE
from tensorflow.keras.layers import Input, Dense, Conv1D
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cond_dense import ConditionedDense
from misc import pickle_save
import matplotlib
matplotlib.use('Agg')


if not os.path.exists('models'):
    os.mkdir('models')
if not os.path.exists('images'):
    os.mkdir('images')


run_train = True
load_existing = False

width = 16
ntrain = 27
epochs = 300
learning_rate = 1e-3

# Load data
mat = scipy.io.loadmat(f'data/hires.mat')
data = np.expand_dims(np.load(f'data/lowres.npy'), -1)
nx = data.shape[1]
fx = data
t_matrix = mat['t_matrix'][0,]
nt = data.shape[0]
nu = mat['nu'][0,]
dx = mat['dxc'][0,]
dt = t_matrix[1]-t_matrix[0]

# Process network input & output
u = data
ub = np.zeros((nt, nx+2, 1)) # padded with periodic boundary
ub[:,0,] = u[:,-1,]
ub[:,-1,] = u[:,0,]
ub[:,1:-1,] = u
adv = 0.5/dx*u*(ub[:,2:,]-ub[:,:-2,])
diffu = nu/dx**2*(ub[:,2:,]-2*u+ub[:,:-2,])
du = u[1:,] - u[:-1,]
c = du/dt+adv[:-1,]-diffu[:-1,]
p = np.concatenate((adv, diffu), axis=-1)

c_p = np.abs(p[:ntrain,]).max(axis=(0,1))
c_r = np.abs(c[:ntrain,]).max(axis=(0,1))

pn = p/c_p
cn = c/c_r

pn_train = pn[:ntrain,]
cn_train = cn[:ntrain,]
pn_test = pn[ntrain:ntrain+10,]
cn_test = cn[ntrain:ntrain+10,]

# Build network
def network_model():
    p = Input(shape=(None, 2))
    x = p
    x = ConditionedDense(width, use_bias=True, use_x_bias=True)([x, x])
    c = Conv1D(filters=1, kernel_size=3, use_bias=False, padding='same')(x)
    return Model(p, c)

model = network_model()
model.summary()

if not run_train or load_existing:
    model.load_weights(f'models/cpcnn')

if run_train:
    optimizer = optimizers.Adam(lr=learning_rate)

    def myloss(y_true, y_pred):
        loss = MSE(y_true[:,2:-2,], y_pred[:,2:-2,])
        return loss

    model.compile(optimizer=optimizer, loss=myloss)

    history = model.fit(
        pn_train, 
        cn_train,
        batch_size=1,
        epochs=epochs,
        validation_data=(pn_test, cn_test),
        shuffle=True,
        verbose=2)

    model.save_weights(f'models/cpcnn')
    pickle_save(history.history, f'models/cpcnn_history.pkl')

# Online prediction
ui = u[0, :, 0]
snapshot = np.zeros((nt, nx))
snapshot[0, ] = ui
for i in range(1, nt):
    ubi = np.zeros(nx+2)
    ubi[1:-1] = ui
    ubi[0] = ui[-1]
    ubi[-1] = ui[0]
    advi = np.expand_dims(0.5/dx*ui*(ubi[2:,]-ubi[:-2,]), axis = -1)
    diffui = np.expand_dims(nu/dx**2*(ubi[2:,]-2*ui+ubi[:-2,]), axis = -1)
    pi = np.expand_dims(np.concatenate((advi, diffui), axis=-1), axis=0)
    ci = model.predict(pi/c_p)*c_r
    ui = ui + dt*(-advi[:,0] + diffui[:,0]+ci[0,:,0])
    snapshot[i, ] = ui

snapshot = np.expand_dims(snapshot, -1)
np.save(f'data/cpcnn_pred.npy', snapshot)

# Single step closure
c_pred = np.zeros((nt-1, nx))
for i in range(0, nt-1):
    ui = u[i,:,0]
    ubi = np.zeros(nx+2)
    ubi[1:-1] = ui
    ubi[0] = ui[-1]
    ubi[-1] = ui[0]
    advi = np.expand_dims(0.5/dx*ui*(ubi[2:,]-ubi[:-2,]), axis = -1)
    diffui = np.expand_dims(nu/dx**2*(ubi[2:,]-2*ui+ubi[:-2,]), axis = -1)
    pi = np.expand_dims(np.concatenate((advi, diffui), axis=-1), axis=0)
    ci = model.predict(pi/c_p)*c_r
    c_pred[i, ] = ci[:,0]

c_pred = np.expand_dims(c_pred, -1)
np.save(f'data/cpcnn_c.npy', c_pred)


fig, ax = plt.subplots()

ax.plot(np.linspace(0, 1, nx), data[0, :, 0], 'k-', label='Initial')
ax.plot(np.linspace(0, 1, nx), data[50, :, 0], 'b-', linewidth=1, label='i=50, truth')
ax.plot(np.linspace(0, 1, nx), snapshot[50, :, 0], 'b--', linewidth=1, label='i=50, pred.')
ax.plot(np.linspace(0,1,nx), data[-1,:,0], 'r-', linewidth=1, label='i=267, truth')
ax.plot(np.linspace(0,1,nx), snapshot[-1,:,0], 'r--', linewidth=1, label='i=267, pred.')
# ax.set_ylim([0, 1.25*Thot])

ax.legend()
plt.savefig(f'images/cpcnn_pred.png', dpi=300)

# Second test case from a new initial condition
mat = scipy.io.loadmat(f'data/hires_newic.mat')
data = np.expand_dims(np.load(f'data/lowres_newic.npy'), -1)
nx = data.shape[1]
fx = data
t_matrix = mat['t_matrix'][0,]
nt = data.shape[0]
nu = mat['nu'][0,]
dx = mat['dxc'][0,]
dt = t_matrix[1]-t_matrix[0]

# Process network input & output
u = data
ub = np.zeros((nt, nx+2, 1)) # padded with periodic boundary
ub[:,0,] = u[:,-1,]
ub[:,-1,] = u[:,0,]
ub[:,1:-1,] = u
adv = 0.5/dx*u*(ub[:,2:,]-ub[:,:-2,])
diffu = nu/dx**2*(ub[:,2:,]-2*u+ub[:,:-2,])
du = u[1:,] - u[:-1,]
c = du/dt+adv[:-1,]-diffu[:-1,]
p = np.concatenate((adv, diffu), axis=-1)

c_p = np.abs(p[:ntrain,]).max(axis=(0,1))
c_r = np.abs(c[:ntrain,]).max(axis=(0,1))

ui = u[0, :, 0]
snapshot = np.zeros((nt, nx))
snapshot[0, ] = ui
for i in range(1, nt):
    ubi = np.zeros(nx+2)
    ubi[1:-1] = ui
    ubi[0] = ui[-1]
    ubi[-1] = ui[0]
    advi = np.expand_dims(0.5/dx*ui*(ubi[2:,]-ubi[:-2,]), axis = -1)
    diffui = np.expand_dims(nu/dx**2*(ubi[2:,]-2*ui+ubi[:-2,]), axis = -1)
    pi = np.expand_dims(np.concatenate((advi, diffui), axis=-1), axis=0)
    ci = model.predict(pi/c_p)*c_r
    ui = ui + dt*(-advi[:,0] + diffui[:,0]+ci[0,:,0])
    snapshot[i, ] = ui

snapshot = np.expand_dims(snapshot, -1)
np.save(f'data/cpcnn_pred_newic.npy', snapshot)

# Single step closure
c_pred = np.zeros((nt-1, nx))
for i in range(0, nt-1):
    ui = u[i,:,0]
    ubi = np.zeros(nx+2)
    ubi[1:-1] = ui
    ubi[0] = ui[-1]
    ubi[-1] = ui[0]
    advi = np.expand_dims(0.5/dx*ui*(ubi[2:,]-ubi[:-2,]), axis = -1)
    diffui = np.expand_dims(nu/dx**2*(ubi[2:,]-2*ui+ubi[:-2,]), axis = -1)
    pi = np.expand_dims(np.concatenate((advi, diffui), axis=-1), axis=0)
    ci = model.predict(pi/c_p)*c_r
    c_pred[i, ] = ci[:,0]

c_pred = np.expand_dims(c_pred, -1)
np.save(f'data/cpcnn_c_newic.npy', c_pred)


fig, ax = plt.subplots()

ax.plot(np.linspace(0, 1, nx), data[0, :, 0], 'k-', label='Initial')
ax.plot(np.linspace(0, 1, nx), data[50, :, 0], 'b-', linewidth=1, label='i=50, truth')
ax.plot(np.linspace(0, 1, nx), snapshot[50, :, 0], 'b--', linewidth=1, label='i=50, pred.')
ax.plot(np.linspace(0,1,nx), data[-1,:,0], 'r-', linewidth=1, label=f'i={nt}, truth')
ax.plot(np.linspace(0,1,nx), snapshot[-1,:,0], 'r--', linewidth=1, label=f'i={nt}, pred.')
# ax.set_ylim([0, 1.25*Thot])

ax.legend()
plt.savefig(f'images/cpcnn_pred_newic.png', dpi=300)