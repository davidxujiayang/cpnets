import sys
sys.path.append('../../src/')
sys.path.append('utils')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from misc import pickle_save, pickle_load, Timer, FeatureScaler

from tensorflow.keras import optimizers
import networkx as nx
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from cond_dense import ConditionedDense
from noisy_sequence import NoisySequence
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Layer, LayerNormalization, ReLU, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from graph_ops import gauss_linear_gradient, weight_mul
import setGPU

tf.keras.backend.set_floatx('float32')

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

run_train = True
load_existing = False


width = 128
depth = 15
nstd = 1.3e-3
epochs = 100
learning_rate = 2e-3

ntrain_max = 400
ntrain = 400
ntest = 400
batch_size = 1

filename = f'gnet-w{width}-d{depth}-n{nstd}-nt{ntrain}'
print(filename)

data = np.float32(np.load('data/deepblue_snapshots_int5_len6000.npy')[ntrain_max-ntrain:ntrain_max+ntest+1, ])
nvar = data.shape[-1]

fs = FeatureScaler(shift=[0]*nvar, norm=[5e5, 200, 200, 2.5e3, 1, 1, 1, 1], separate_channel=True)
nf = fs.normalize(data)
nf_std = nf.std(axis=(0,1))
print('STD NF: ', nf_std)

res = nf[1:,] - nf[:-1,]
res_norm = 1e-2
rf = res/res_norm
rf_std = rf.std(axis=(0,1))
print('STD RF: ', rf_std)

G = pickle_load('data/graph.pkl')
n_edges = G.number_of_edges()
aidx = np.zeros((n_edges, 2), dtype=np.int32)
ef = np.zeros((n_edges, 2))
areas = np.zeros((n_edges, 1))
edge_noslip = []
edge_sym = []
edge_inner = []
for i, (u, v, e) in enumerate(G.edges.data()):
    idx = e['idx']
    ef[idx, :2] = e['vec']
    areas[idx, ] = e['area']
    aidx[idx, 0] = u
    aidx[idx, 1] = v

    if G.nodes[v]['ghost'] == 1:
        edge_noslip.append(idx)
    elif G.nodes[v]['ghost'] == 2:
        edge_sym.append(idx)
    else:
        edge_inner.append(idx)

print('NUMBER OF EDGES: ', n_edges)

elem_inout = []
elem_noslip = []
elem_sym = []
volumes = []
for i in G.nodes():
    volumes.append(G.nodes[i]['volume'])
    if G.nodes[i]['is_boundary'] == 1:
        elem_inout.append(i)
    if G.nodes[i]['ghost'] == 1:
        elem_noslip.append(i)
    if G.nodes[i]['ghost'] == 2:
        elem_sym.append(i)

volumes = np.float32(np.array(volumes))

N = G.number_of_nodes() # Number of nodes in the graph including ghost ones
N0 = nf.shape[1]     # Number of nodes in the mesh domain
F = nf.shape[-1]    # Dimension of node features
S = ef.shape[-1]    # Dimension of edge features
n_edges = ef.shape[-2]

elem_inner = list(set(range(N)) - set(elem_inout) - set(elem_noslip) - set(elem_sym))

areas = areas/areas[edge_inner, ].max()

ef = np.repeat(np.expand_dims(ef, axis=0), nf.shape[0], axis=0)

index_i = tf.convert_to_tensor(aidx[:,0])
index_j = tf.convert_to_tensor(aidx[:,1])
index_i_inner = tf.convert_to_tensor(aidx[edge_inner,0])
index_j_inner = tf.convert_to_tensor(aidx[edge_inner,1])
index_i_noslip = tf.convert_to_tensor(aidx[edge_noslip,0])
index_j_noslip = tf.convert_to_tensor(aidx[edge_noslip,1])
index_i_sym = tf.convert_to_tensor(aidx[edge_sym,0])
index_j_sym = tf.convert_to_tensor(aidx[edge_sym,1])

nf_train = nf[:ntrain, ]
ef_train = ef[:ntrain, ]
y_train = rf[:ntrain, ]

v_inv = 1/volumes
v_inv = v_inv/v_inv[elem_inner, ].max()

rij = tf.convert_to_tensor(np.float32(areas[:,0])*v_inv[aidx[:,0]])
rji = tf.convert_to_tensor(np.float32(areas[:,0])*v_inv[aidx[:,1]])
v_inv = tf.convert_to_tensor(v_inv)

def get_model(width, depth):
    X_in = Input(shape=(N0, F,))
    E_in = Input(shape=(n_edges, S,))

    x = X_in

    e = E_in
    
    e_inner = tf.gather(e, edge_inner, axis=1)
    e_noslip = tf.gather(e, edge_noslip, axis=1)
    e_sym = tf.gather(e, edge_sym, axis=1)
    
    x = Dense(width, name=f'input_x_conddense_0', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(width, name=f'input_x_conddense_1', activation='relu')(x)
    x = LayerNormalization()(x)

    for i in range(depth):
        x0 = x

        x = Dense(width, activation='relu', name=f'x_dense_{i}')(x)
        x = LayerNormalization()(x)

        xi = tf.gather(x, index_i_inner, axis=1) 
        xj = tf.gather(x, index_j_inner, axis=1)

        concat_inner = Concatenate()([xi, xj, e_inner])
        flux_inner = Dense(width, name=f'flux_dense_{i}')(concat_inner)
        x_noslip = tf.gather(x, index_i_noslip, axis=1)
        concat_noslip = Concatenate()([x_noslip, e_noslip])
        flux_noslip = Dense(width, name=f'flux_noslip_dense_{i}')(concat_noslip)
        x_sym = tf.gather(x, index_i_sym, axis=1)
        concat_sym = Concatenate()([x_sym, e_sym])
        flux_sym = Dense(width, name=f'flux_sym_dense_{i}')(concat_sym)

        flux = Concatenate(axis=1)([flux_inner, flux_noslip, flux_sym])
        flux = ReLU()(flux)

        fluxi = weight_mul(flux, rij)
        fluxj = weight_mul(flux, rji)

        fluxi = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(fluxi, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
        fluxj = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(fluxj, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])
        flux = fluxi-fluxj
        flux = flux[:,:N0,]
        flux = LayerNormalization()(flux)
        
        s = Dense(width, activation='relu', name=f'source_dense0_{i}')(x)
        s = LayerNormalization()(s)
        s = Dense(width, name=f'source_dense1_{i}')(s)
        s = LayerNormalization()(s)

        r = flux+s
        x = x0+r

    r = Dense(width, activation='relu', name=f'output_r_dense0')(r)
    r = Dense(width//2, activation='relu', name=f'output_r_dense1')(r)
    r = Dense(nvar, use_bias=False, name=f'output_r_dense2')(r)

    outr = r[:,:N0,]

    return Model(inputs = [X_in, E_in], outputs = outr)


def myloss(y_true, y_pred):
    inner_truth = tf.gather(y_true, elem_inner, axis=1)
    inner_pred = tf.gather(y_pred, elem_inner, axis=1)
    
    loss = tf.reduce_mean(tf.keras.losses.MSE(inner_truth, inner_pred))

    return loss

if run_train:
    model = get_model(width, depth)
    
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    print('Trainable params: {:,}'.format(trainable_count))
            
    if load_existing:
        model.load_weights(f'models/{filename}')
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=myloss)

    train_gen = NoisySequence(nf_train, ef_train, y_train, nstd=nstd, res_norm=res_norm, shuffle=True, batch_size=batch_size)
    
    with Timer('Training'):
        history = model.fit(
            train_gen,
            epochs=epochs,
            callbacks=[],
            verbose=2)


    model.save_weights(f'models/{filename}')
    pickle_save(history.history, f'models/history_{filename}.pkl')

model = get_model(width, depth)
model.load_weights(f'models/{filename}')
print(filename)

def recurrent_predict(nf0, n_iter, truth):
    # truth is only used for inlet/outlet ghost elements
    pred_recurr = np.zeros(truth.shape)
    ef0 = tf.convert_to_tensor(ef[[ntrain],])

    for i in range(n_iter):
        rf_pred = model.predict([tf.convert_to_tensor(nf0), ef0])
        rf_pred = rf_pred*res_norm

        nf0[0, elem_inout,] = truth[i, elem_inout, ] # truth used for inlet/outlet ghost elements
        nf0[0, elem_inner,] = nf0[0, elem_inner,] + rf_pred[0, elem_inner,] # other cells from prediction

        pred_recurr[[i], ] = nf0

    return pred_recurr

c = np.array([5e5, 200, 200, 2.5e3, 1, 1, 1, 1])

nf_pred = recurrent_predict(nf[[ntrain-1],], 1, nf[[ntrain],])
nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain,]), fs.denormalize(nf_pred[-1,]), multioutput='raw_values'))/c
print(f'NRMSE 1 @ 0: {nrmse}')

nf_pred = recurrent_predict(nf[[ntrain+10],], 1, nf[[ntrain+1+10],])
nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+1+10,]), fs.denormalize(nf_pred[-1,]), multioutput='raw_values'))/c
print(f'NRMSE 1 @ 11: {nrmse}')

nf_pred = recurrent_predict(nf[[ntrain-1+ntest],], 1, nf[[ntrain+ntest],])
nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+ntest,]), fs.denormalize(nf_pred[-1,]), multioutput='raw_values'))/c
print(f'NRMSE 1 @ {ntest}: {nrmse}')

nf_pred = recurrent_predict(nf[[ntrain],], 10, nf[ntrain+1:ntrain+1+10,])
nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+10,]), fs.denormalize(nf_pred[-1,]), multioutput='raw_values'))/c
print(f'NRMSE {10}: {nrmse}')

with Timer('Prediction'):
    nf_pred = recurrent_predict(nf[[ntrain],], ntest, nf[ntrain+1:ntrain+1+ntest,])
    if ntest>100:
        nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+100,]), fs.denormalize(nf_pred[-(ntest-100)-1,]), multioutput='raw_values'))/c
        print(f'NRMSE {100}: {nrmse}')
    if ntest>200:
        nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+200,]), fs.denormalize(nf_pred[-(ntest-200)-1,]), multioutput='raw_values'))/c
        print(f'NRMSE {200}: {nrmse}')
    nrmse = np.sqrt(mean_squared_error(fs.denormalize(nf[ntrain+ntest,]), fs.denormalize(nf_pred[-1,]), multioutput='raw_values'))/c
    print(f'NRMSE {ntest}: {nrmse}')
np.save(f'data/gnet_pred.npy', fs.denormalize(np.concatenate((nf[[ntrain],], nf_pred), axis=0)[::10,])) # includes the IC
