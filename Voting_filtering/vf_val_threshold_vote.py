from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt

# Noise rate - float from 0 to 0.5 indicating proportion of data to be removed
noise_rate = 0.1
# Data import and making train, test and validation sets
sbjs = [33, 34]  # [25,26,27,28,29,30,32,33,34,35,36,37,38]
path_to_data = '/home/likan_blk/BCI/NewData/'  # os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3))
# Some files for logging
logdir = os.path.join(os.getcwd(), 'logs', 'vf_val_thres_vote')
if not os.path.isdir(logdir):
    os.makedirs(logdir)
fname = os.path.join(logdir, 'auc_scores.csv')
with open(fname, 'w') as fout:
    fout.write('subject,auc_noisy,auc_pure,samples_before,samples_after,epoch_number\n')


epochs = 50
dropouts = (0.2, 0.4, 0.6)
n_shufflings = 5  # number of different splits on train and val (in cross-validation)

# Iterate over subjects and clean label noise for all of them
for sbj in sbjs:
    print("Voting filtering for subject %s data"%(sbj))
    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=108)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]

    mistakes = []  # Array of indices of samples, for which prediction is wrong
    for i in range(n_shufflings):
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i*2+1)
        val_inds = []
        fold_pairs = []
        for tr_ind, val_ind in cv.split(X_train, y_train):
            X_tr, X_val = X_train[tr_ind], X_train[val_ind]
            y_tr, y_val = y_train[tr_ind], y_train[val_ind]
            fold_pairs.append((X_tr, y_tr, X_val, y_val))
            val_inds.append(train_ind[val_ind])  # indices of all the validation instances in the initial X array

        # Getting and training models with cross-validation
        i = 0
        pure_ind = []
        bestepochs = np.array([])
        time_samples_num = X_train.shape[1]
        channels_num = X_train.shape[2]

        for fold in fold_pairs:
            X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
            model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
            callback = LossMetricHistory(n_iter=epochs, validation_data=(X_val, y_val),
                                         verbose=1, fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(i)))
            hist = model.fit(X_tr, y_tr, epochs=epochs,
                             validation_data=(X_val, y_val), callbacks=[callback],
                             batch_size=64, shuffle=True)
            bestepochs = np.append(bestepochs, callback.bestepoch+1)

            # Validation and data cleaning
            model = load_model(os.path.join(logdir, "model%s.hdf5" % (i)))
            y_pred = model.predict(X_val)[:,1]
            # Indices of noisy samples
            mistakes += list(val_inds[i][np.abs(y_val[:,1] - y_pred) >= 0.5])
            i += 1
        bestepoch=int(round(bestepochs.mean()))

    # Removing instances with noisy labels
    pure_ind = []  # Indices of non-noisy samples (according to majority vote)
    mistakes_per_sample = []  # Array of numbers of mistakes per each sample (from 0 to n_shufflings)
    for i in train_ind:
        c = mistakes.count(i)
        mistakes_per_sample.append(c)
        if c < n_shufflings/2.:
            pure_ind.append(i)
    X_train_pure = X[pure_ind]
    y_train_pure = y[pure_ind]

    # Plotting histogram of number of mistakes per sample
    plt.title('Number of mistakes per sample histogram for %s subject'%(sbj))
    plt.xlabel('number of mistakes')
    plt.ylabel('number of samples')
    plt.hist(mistakes_per_sample, bins=n_shufflings+1, rwidth=0.8, color='indigo')
    plt.savefig(os.path.join(logdir, str(sbj)+'mistakes_hist.png'))

    # Testing and comparison of cleaned and noisy data
    samples_before = y_train.shape[0]
    samples_after = y_train_pure.shape[0]
    y_train = to_categorical(y_train)
    y_train_pure = to_categorical(y_train_pure)

    model_noisy, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_noisy.fit(X_train, y_train, epochs=bestepoch,
                    batch_size=64, shuffle=True)

    y_pred_noisy = model_noisy.predict(X_test)
    y_pred_noisy = y_pred_noisy[:, 1]
    auc_noisy = roc_auc_score(y_test, y_pred_noisy)

    model_pure, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_pure.fit(X_train, y_train, epochs=bestepoch,
                   batch_size=64, shuffle=True)
    y_pred_pure = model_pure.predict(X_test)
    y_pred_pure = y_pred_pure[:, 1]
    auc_pure = roc_auc_score(y_test, y_pred_pure)

    with open(fname, 'a') as fout:
        fout.write(','.join(map(str, [sbj, auc_noisy, auc_pure, samples_before, samples_after, bestepoch])))
        fout.write('\n')
