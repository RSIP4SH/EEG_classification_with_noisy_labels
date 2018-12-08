random_state = 108
import random
random.seed(random_state)
import numpy as np
np.random.seed(random_state)
import tensorflow as tf
tf.set_random_seed(random_state)

from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold

import sys
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import os
if len(sys.argv) < 4:
    print("Usage: \n"
          "python %s path_to_data path_to_logs filtration_rate[0...0.5)"%sys.argv[0],
          "[compute_noisy_ens (0 or 1, default 1)] [compute_noisy_naive (0 or 1, default 1)]\n"
          "For example, if you want to discard 10% of data in each class \n"
          "and then train the network again, use something like: \n"
          "%s ../Data ./logs/cf 0.1"%sys.argv[0])
    exit()

filt_rate = sys.argv[3]
logdir = sys.argv[2] #os.path.join(os.getcwd(),'logs', 'cf_ensemble_naive_fr%s'%filt_rate)
if not os.path.isdir(logdir):
    os.makedirs(logdir)
fname_ens = os.path.join(logdir, 'auc_scores_ens.csv')
with open(fname_ens, 'w') as fout:
    fout.write('subject,auc_noisy,auc_pure,samples_before,samples_after\n')

fname_nai = os.path.join(logdir, 'auc_scores_naive.csv')
with open(fname_nai, 'w') as fout:
    fout.write('subject,auc_noisy,auc_pure,samples_before,samples_after, best_epoch\n')

with open(os.path.join(logdir, 'err_ind_ens.csv'), 'w') as fout:
    fout.write('subject,class,indices\n')
with open(os.path.join(logdir, 'err_ind_naive.csv'), 'w') as fout:
    fout.write('subject,class,indices\n')

epochs = 150
dropouts = (0.72,0.32,0.05)
nfold = 4

path_to_data = sys.argv[1] # '/home/likan_blk/BCI/NewData/'
sbjs = [25,26,27,28,29,30,32,33,34,35,36,37,38]
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3), resample_to=323)
dropouts = (0.72,0.32,0.05)

mean_val_aucs=[]
test_aucs_naive = []
test_aucs_ensemble = []

for sbj in sbjs:
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=random_state)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
    val_inds = []
    fold_pairs = []
    cv = StratifiedKFold(n_splits=4, shuffle=False)
    for tr_ind, val_ind in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_ind], X_train[val_ind]
        y_tr, y_val = y_train[tr_ind], y_train[val_ind]
        fold_pairs.append((X_tr, y_tr, X_val, y_val))
        val_inds.append(train_ind[val_ind])  # indices of all the validation instances in the initial X array


    bestepochs = np.array([])
    time_samples_num = X_train.shape[1]
    channels_num = X_train.shape[2]
    y_pred = 0
    y_pred_test = 0
    for fold in fold_pairs:
        X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
        y_val_bin = fold[3]

        np.random.seed(random_state)
        tf.set_random_seed(random_state)

        model = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,verbose=1,
                                     fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(nfold)))
        hist = model.fit(X_tr, y_tr, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[callback],
                        batch_size=64, shuffle=False)
        bestepochs = np.append(bestepochs, callback.bestepoch+1)

        # Validation and data cleaning
        model = load_model(os.path.join(logdir, "model%s.hdf5"%(nfold)))
        y_pred += model.predict(X_train)[:,1]   #  ensemble predictions
        y_pred_test += model.predict(X_test)[:,1]

    bestepoch = int(round(bestepochs.mean()))

    # Data cleaning
    filt_rate = float(filt_rate)
    #ind = np.array(val_inds[i])  # indices of validation samples in the initial dataset
    n_err1 = int(np.round(y_train.sum() * filt_rate))  # Number of samples in target class to be thrown away
    n_err0 = int(np.round((len(y_train) - y_train.sum()) * filt_rate))  # Number of samples in nontarget class
                                                                        # to be thrown away

    #######################################
    ########### Ensemble model ############
    #######################################
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    y_pred /= nfold
    pure_ind = np.array([], dtype=np.int32)
    err_nontarg_ind = np.array([], dtype=np.int32)
    err_target_ind = np.array([], dtype=np.int32)

    argsort0 = np.argsort(y_pred[y_train == 0])[::-1]   # Descending sorting of predictions for nontarget class
                                                        # so that the most erroneous sample are at the
                                                        # beginning of the array
    argsort1 = np.argsort(y_pred[y_train == 1])     # Ascending Sorting of predictions for target class
                                                    # so that the most erroneous sample are at the
                                                    # beginning of the array
    target_ind = train_ind[y_train == 1][argsort1]
    nontarg_ind = train_ind[y_train == 0][argsort0]
    err_target_ind = np.append(err_target_ind, target_ind[:n_err1])
    err_nontarg_ind = np.append(err_nontarg_ind, nontarg_ind[:n_err0])  # Take demanded amount of error samples
    pure_ind = np.append(pure_ind, target_ind[n_err1:])
    pure_ind = np.append(pure_ind, nontarg_ind[n_err0:])

    # Removing instances with noisy labels
    np.random.shuffle(pure_ind)
    X_train_pure = X[pure_ind]
    y_train_pure = y[pure_ind]

    # Saving erroneous sample indices
    with open(os.path.join(logdir, 'err_ind_ens.csv'), 'a') as fout:
        fout.write(str(sbj))
        fout.write(',0,')
        fout.write(','.join(map(str, err_nontarg_ind)))
        fout.write('\n')
        fout.write(str(sbj))
        fout.write(',1,')
        fout.write(','.join(map(str, err_target_ind)))
        fout.write('\n')

    # Train ensemble model on pure data
    y_train_pure = to_categorical(y_train_pure)

    cv = StratifiedKFold(n_splits=nfold, shuffle=False)
    fold_pairs_pure = []
    y_pred_pure = 0
    for fold, (tr_ind, val_ind) in enumerate(cv.split(X_train_pure, y_train_pure[:,1])):
        X_tr, X_val = X_train_pure[tr_ind], X_train_pure[val_ind]
        y_tr, y_val = y_train_pure[tr_ind], y_train_pure[val_ind]
        fold_pairs_pure.append((X_tr, y_tr, X_val, y_val))
        callback = LossMetricHistory(n_iter=epochs, verbose=1,
                                     fname_bestmodel=os.path.join(logdir, "model%s_ens_pure.hdf5" % (nfold)))
        np.random.seed(random_state)
        tf.set_random_seed(random_state)
        model_pure = get_model(time_samples_num, channels_num, dropouts=dropouts)
        model_pure.fit(X_tr, y_tr, epochs=epochs,
                       validation_data=(X_val, y_val), callbacks=[callback],
                       batch_size=64, shuffle=False)
        model_pure = load_model(os.path.join(logdir, "model%s_ens_pure.hdf5" % (nfold)))
        y_pred_pure += model_pure.predict(X_test)[:, 1]

    y_pred_pure /= nfold

    # Compare to old (noisy) ensemble model
    samples_before = y_train.shape[0]
    samples_after= y_train_pure.shape[0]

    auc_noisy_ens = roc_auc_score(y_test, y_pred_test)
    auc_pure_ens = roc_auc_score(y_test, y_pred_pure)

    #######################################
    ###### Mean-epoch (naive) model #######
    #######################################
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    pure_ind = np.array([], dtype=np.int32)
    err_nontarg_ind = np.array([], dtype=np.int32)
    err_target_ind = np.array([], dtype=np.int32)

    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    model = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model.fit(X_train, to_categorical(y_train), epochs=bestepoch, batch_size=64)
    y_pred = model.predict(X_train)[:, 1]
    argsort0 = np.argsort(y_pred[y_train == 0])[::-1]   # Descending sorting of predictions for nontarget class
                                                        # so that the most erroneous sample are at the
                                                        # beginning of the array
    argsort1 = np.argsort(y_pred[y_train == 1])     # Ascending Sorting of predictions for target class
                                                    # so that the most erroneous sample are at th
                                                    # beginning of the array
    target_ind = train_ind[y_train == 1][argsort1]
    nontarg_ind = train_ind[y_train == 0][argsort0]
    err_target_ind = np.append(err_target_ind, target_ind[:n_err1])
    err_nontarg_ind = np.append(err_nontarg_ind, nontarg_ind[:n_err0])  # Take demanded amount of error samples
    pure_ind = np.append(pure_ind, target_ind[n_err1:])
    pure_ind = np.append(pure_ind, nontarg_ind[n_err0:])

    # Removing instances with noisy labels
    np.random.shuffle(pure_ind)
    X_train_pure = X[pure_ind]
    y_train_pure = y[pure_ind]

    # Saving erroneous sample indices
    with open(os.path.join(logdir, 'err_ind_naive.csv'), 'a') as fout:
        fout.write(str(sbj))
        fout.write(',0,')
        fout.write(','.join(map(str, err_nontarg_ind)))
        fout.write('\n')
        fout.write(str(sbj))
        fout.write(',1,')
        fout.write(','.join(map(str, err_target_ind)))
        fout.write('\n')

    # Train naive model on pure data
    y_train_pure = to_categorical(y_train_pure)

    cv = StratifiedKFold(n_splits=nfold, shuffle=False)
    fold_pairs_pure = []
    y_pred_pure = 0
    bestepochs = np.array([])

    for fold, (tr_ind, val_ind) in enumerate(cv.split(X_train_pure, y_train_pure[:,1])):
        X_tr, X_val = X_train_pure[tr_ind], X_train_pure[val_ind]
        y_tr, y_val = y_train_pure[tr_ind], y_train_pure[val_ind]
        callback = LossMetricHistory(n_iter=epochs, verbose=1)
                                     #fname_bestmodel=os.path.join(logdir, "model%s_naive_pure.hdf5" % (fold)))
        np.random.seed(random_state)
        tf.set_random_seed(random_state)
        model_pure = get_model(time_samples_num, channels_num, dropouts=dropouts)
        model_pure.fit(X_tr, y_tr, epochs=epochs,
                       validation_data=(X_val, y_val), callbacks=[callback],
                       batch_size=64)
        bestepochs = np.append(bestepochs, callback.bestepoch + 1)
    bestepoch = int(round(bestepochs.mean()))

    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    model_pure = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_pure.fit(X_train_pure, y_train_pure, epochs=bestepoch, batch_size=64)
    y_pred_pure = model_pure.predict(X_test)[:, 1]

    # Compare to old (noisy) ensemble model
    y_pred = model.predict(X_test)[:, 1]
    auc_noisy_naive = roc_auc_score(y_test, y_pred)
    auc_pure_naive = roc_auc_score(y_test, y_pred_pure)

    #######################################
    # Write results
    with open(fname_ens, 'a') as fout:
        fout.write(u"%s,%.04f,%.04f,%s,%s\n"%(sbj, auc_noisy_ens, auc_pure_ens,
                                                            samples_before, samples_after))
    with open(fname_nai, 'a') as fout:
        fout.write(u"%s,%.04f,%.04f,%s,%s,%s\n"%(sbj, auc_noisy_naive, auc_pure_naive, samples_before,
                                                             samples_after, bestepoch))
