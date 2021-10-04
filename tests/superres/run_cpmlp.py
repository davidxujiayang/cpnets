import os
import sys
sys.path.append('../../src/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from scipy.io import loadmat
import scipy.io
from cond_dense import ConditionedDense
import matplotlib.pyplot as plt
from misc import pickle_save
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np
import h5py
from tensorflow.keras.layers import Input, Dense, ELU, Concatenate


seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

if not os.path.exists('models'):
    os.mkdir('models')
if not os.path.exists('images'):
    os.mkdir('images')

run_train = True
load_existing = False

z_train = [700, 800]
z_test = [650, 750, 850] + z_train
nx_list = [8, 16, 24, 32, 48, 64]

# input
dtrain = None
for z in z_train:
    for nx in nx_list:
        F = loadmat(f'data/dg{z}_{nx}.mat')
        data = F['dg_data']
        if dtrain is None:
            dtrain = data
        else:
            dtrain = np.concatenate((dtrain, data), axis=0)
print(dtrain.shape)
dtrain = np.reshape(dtrain, (-1, data.shape[-1]))

# split train/val/test data
x_train, x_val, p_train, p_val, y_train, y_val = train_test_split(
    dtrain[:, :36], dtrain[:, [36]], dtrain[:, 37:], train_size=0.9, random_state=0)

# Build network
width = 48


def network_model():
    input1 = Input(shape=36)
    input2 = Input(shape=1)

    x = Dense(width, )(input1)
    x = ELU(alpha=1.0)(x)
    p = input2
    x = ConditionedDense(width, use_bias=True)([x, p])
    x = ELU(alpha=1.0)(x)
    x = Dense(width, )(x)
    x = ELU(alpha=1.0)(x)
    x = Dense(32, )(x)
    x = ELU(alpha=1.0)(x)
    output = Dense(16)(x)

    return Model([input1, input2], output)


model = network_model()
model.summary()

if not run_train or load_existing:
    model = load_model(f'models/cpmlp')

if run_train:
    # compile the keras model
    optimizer = optimizers.Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)

    # fit the keras model on the dataset
    history = model.fit(
        [x_train, p_train],
        y_train,
        batch_size=128,
        epochs=100,
        validation_data=([x_val, p_val], y_val),
        shuffle=True,
        verbose=2)

    model.save(f'models/cpmlp')
    pickle_save(history.history, f'models/cpmlp_history.pkl')

# Prediction
for z in z_test:
    for nx in nx_list:
        F = loadmat(f'data/dg{z}_{nx}.mat')
        dtest = F['dg_data']
        x_test = dtest[:, :36]
        p_test = dtest[:, [36]]
        y_test = dtest[:, 37:]
        pred_test = model.predict([x_test, p_test])
        np.save(f'data/cpmlp_pred{z}_{nx}.npy', pred_test)
        scipy.io.savemat(
            f'data/cpmlp_pred{z}_{nx}.mat', dict(dg_data=pred_test))
