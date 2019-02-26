import numpy as np
from scipy.io import loadmat
import os
from mne.filter import resample
from LatentSVDD.params import PATH_TO_DATA

def mean_resample(x, windows=[]):
    new_x = []
    if len(windows):
        for i in range(len(windows) - 1):
            left = windows[i]
            right = windows[i + 1]
            new_x.append(x[:,left:right,:].mean(axis=1,keepdims=True))
        x = np.hstack(new_x)
    return x

def gradient_features(x, windows=[], step=1):
    x = mean_resample(x, windows)
    grad = (x[:,1:,:] - x[:,:-1,:]) / step
    return grad

def get_feature_vectors(x, winsize=15):
    windows = []
    for i in range(x.shape[1] // winsize):
        windows.append(i * winsize)
    windows.append(x.shape[1])
    x = gradient_features(x, windows, winsize)
    return x.reshape(-1, x.shape[1] * x.shape[2])

def standard_features(x):
    pass

def gram(x):
    return np.dot(x,x.T)

def kernel(x):
    pass
    return gram(x)

def phi(x): # Phi
    return x

def jointFeatureMap(x, phi, n_states): # Psy
    pass

class DataLoader:
    def __init__(self, path, start_epoch=-0.5, end_epoch=1, sample_rate=500):
        self.path = path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.sample_rate = sample_rate

    def _baseline(self,X,baseline_window=()):
        bl_start = int((baseline_window[0] - self.start_epoch) * self.sample_rate)
        bl_end = int((baseline_window[1] - self.start_epoch) * self.sample_rate)
        return X[:, bl_start:bl_end, :].mean(axis=1)

    def _resample(self, X, y, resample_to):
        self.sample_rate = resample_to
        duration = self.end_epoch - self.start_epoch
        downsample_factor = X.shape[1] / (resample_to * duration)
        return resample(X, up=1., down=downsample_factor, npad='auto', axis=1), y

    def data_shuffle(self, X, y, seed=None):
        inds = np.arange(y.shape[0])
        np.random.seed(seed)
        np.random.shuffle(inds)
        X = X[inds]
        y = y[inds]
        return X, y

    def get_data(self, subjects, plusminusone_target=True, window=None,
                 baseline_window=(),resample_to=None, shuffle=True, seed=None):
        res = {}
        for subject in subjects:
            eegT = loadmat(os.path.join(self.path, str(subject), 'eegT.mat'))['eegT']
            eegNT = loadmat(os.path.join(self.path, str(subject), 'eegNT.mat'))['eegNT']
            X = np.concatenate((eegT, eegNT), axis=-1).transpose(2, 0, 1)
            if len(baseline_window):
                baseline = self._baseline(X, baseline_window)
                baseline = np.expand_dims(baseline, axis=1)
                X = X - baseline
            if plusminusone_target:
                y = np.hstack((np.ones(eegT.shape[2]), -np.ones(eegNT.shape[2])))
            else:
                y = np.hstack((np.ones(eegT.shape[2]), np.zeros(eegNT.shape[2])))
            if (resample_to is not None) and (resample_to != self.sample_rate):
                X, y = self._resample(X, y, resample_to)

            if window is not None:
                start_window_ind = int((window[0] - self.start_epoch) * self.sample_rate)
                end_window_ind = int((window[1] - self.start_epoch) * self.sample_rate)
                X, y = X[:, start_window_ind:end_window_ind, :], y

            if shuffle:
                X, y = self.data_shuffle(X, y, seed)
            res[subject] = (X, y)
        return res


if __name__ == '__main__':
    data = DataLoader(PATH_TO_DATA).get_data([33,34], window=(0.2,0.5), baseline_window=(0.2,0.3))
    X, y = data[34][0], data[34][1]
    print(y)