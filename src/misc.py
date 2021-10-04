import pickle
from copy import deepcopy
import time


class FeatureScaler():
    def __init__(self, shift=None, norm=None, separate_channel=True):
        self.shift = shift
        self.norm = norm
        self.separate_channel = separate_channel

    def normalize(self, feature):
        feature = deepcopy(feature)
        if self.separate_channel:
            for i in range(feature.shape[-1]):
                feature[..., i] = (feature[..., i] -
                                   self.shift[i])/self.norm[i]
        else:
            feature = (feature - self.shift)/self.norm

        return feature

    def denormalize(self, feature):
        feature = deepcopy(feature)
        if self.separate_channel:
            for i in range(feature.shape[-1]):
                feature[..., i] = feature[..., i]*self.norm[i] + self.shift[i]
        else:
            feature = feature * self.norm + self.shift

        return feature

def pickle_save(obj, filename, binary=True):
    if binary:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
    else:
        with open(filename, 'w') as f:
            pickle.dump(obj, f, protocol=4)


def pickle_load(filename, binary=True):
    if binary:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(filename, "r") as f:
            obj = pickle.load(f)
    return obj

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))