random_state = 9
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
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve
import os, sys

#if len(sys.argv) < 3:
#    print("Usage: \n"
#          "python classification_filtering.py path_to_data path_to_logs filtration_rate[0...0.5) \n"
#          "For example, if you want to discard 10% of data in each class \n"
#          "and then train the network again, use something like: \n"
#          "./classification_filtering.py ../Data ./logs/cf 0.1")
#    exit()


# Data import and making train, test and validation sets
sbjs = [32] #[25,26,27,28,29,30,32,33,34,35,36,37,38]
frs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] # filter rates
path_to_data =  sys.argv[1] #'/home/likan_blk/BCI/NewData/' #os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3), resample_to=323)
# Some files for logging
logdir =  sys.argv[2] #os.path.join(os.getcwd(),'logs', 'cf_fr')
if not os.path.isdir(logdir):
    os.makedirs(logdir)
for fr in frs:
    fname_auc = os.path.join(logdir, 'auc_scores')
    with open(fname_auc+str(fr)+'.csv', 'w') as fout:
        fout.write('subject,auc_noisy,auc_pure,samples_before,samples_after\n')
    fname_err_ind = os.path.join(logdir, 'err_indices')
    with open(fname_err_ind+str(fr)+'.csv', 'w') as fout:
        fout.write('subject,class,indices\n')

epochs = 150
dropouts = (0.72,0.32,0.05)

#if len(sys.argv) > 3:
#    filt_rate = sys.argv[3]
#else:
#    filt_rate = "all"


# Iterate over subjects and clean label noise for all of them
for sbj in sbjs:
    print("Classification filtering for subject %s data"%(sbj))
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=108)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]

    time_samples_num = X_train.shape[1]
    channels_num = X_train.shape[2]

    cv = StratifiedKFold(n_splits=4, shuffle=False)
    val_inds = []
    fold_pairs = []
    for tr_ind, val_ind in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_ind], X_train[val_ind]
        y_tr, y_val = y_train[tr_ind], y_train[val_ind]
        fold_pairs.append((X_tr, y_tr, X_val, y_val))
        val_inds.append(train_ind[val_ind]) # indices of all the validation instances in the initial X array

    pure_ind = {}
    err_target_ind = {}
    err_nontarg_ind = {}
    for fr in frs:
        pure_ind[fr] = np.array([], dtype=np.int32)
        err_target_ind[fr] = np.array([], dtype=np.int32)
        err_nontarg_ind[fr] = np.array([], dtype=np.int32)
    bestepochs = np.array([])


    for fold,  (X_tr, y_tr, X_val, y_val_bin) in enumerate(fold_pairs):
        y_tr = to_categorical(y_tr)
        y_val = to_categorical(y_val_bin)
        callback = LossMetricHistory(n_iter=epochs,verbose=1,
                                     fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(fold)))

        np.random.seed(random_state)
        tf.set_random_seed(random_state)
        model = get_model(time_samples_num, channels_num, dropouts=dropouts)
        model.fit(X_tr, y_tr, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[callback],
                        batch_size=64, shuffle=True)
        bestepochs = np.append(bestepochs, callback.bestepoch+1)

        # Classification filtering of validation data
        model = load_model(os.path.join(logdir, "model%s.hdf5"%(fold)))
        y_pred = model.predict(X_val)[:,1]

        #if filt_rate != "all":
        for fr in frs:
            #filt_rate = float(filt_rate)
            ind = np.array(val_inds[fold]) # indices of validation samples in the initial dataset
            n_err1 = int(np.round(y_val_bin.sum()*fr))  # Number of samples in target class to be thrown away
            n_err0 = int(np.round((len(y_val_bin)-y_val_bin.sum())*fr))  # Number of samples in nontarget class
                                                                        # to be thrown away
            argsort0 = np.argsort(y_pred[y_val_bin==0])[::-1]   # Descending sorting of predictions for nontarget class
                                                            # so that the most erroneous sample are at the
                                                            #begining of the array
            argsort1 = np.argsort(y_pred[y_val_bin==1])     # Ascending Sorting of predictions for target class
                                                        # so that the most erroneous sample are at the
                                                        #begining of the array
            target_ind = ind[y_val_bin==1][argsort1]
            nontarg_ind = ind[y_val_bin==0][argsort0]
            err_target_ind[fr] = np.append(err_target_ind, target_ind[:n_err1])
            err_nontarg_ind[fr] = np.append(err_nontarg_ind, nontarg_ind[:n_err0]) # Take demanded amount of error samples
            pure_ind[fr] = np.append(pure_ind[fr], target_ind[n_err1:])
            pure_ind[fr] = np.append(pure_ind[fr], nontarg_ind[n_err0:])
    bestepoch = int(round(bestepochs.mean()))
    np.random.seed(random_state)
    tf.set_random_seed(random_state)
    model_noisy = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_noisy.fit(X_train, to_categorical(y_train),
                    epochs=bestepoch,
                    batch_size=64, shuffle=False)
    # Test noisy classifier
    y_pred_noisy = model_noisy.predict(X_test)
    y_pred_noisy = y_pred_noisy[:, 1]
    auc_noisy = roc_auc_score(y_test, y_pred_noisy)

    for fr in frs:
        np.random.shuffle(pure_ind[fr])
        X_train_pure = X[pure_ind[fr]]
        y_train_pure = y[pure_ind[fr]]

        # OPTIONALLY: saving erroneous sample indices
        with open(fname_err_ind+str(fr)+'.csv', 'a') as fout:
            fout.write(str(sbj))
            fout.write(',0,')
            fout.write(','.join(map(str,err_nontarg_ind[fr])))
            fout.write('\n')
            fout.write(str(sbj))
            fout.write(',1,')
            fout.write(','.join(map(str,err_target_ind[fr])))
            fout.write('\n')

        # Testing and comparison of cleaned and noisy data
        samples_before = y_train.shape[0]
        samples_after = y_train_pure.shape[0]
        y_train_pure = to_categorical(y_train_pure)

        callback = LossMetricHistory(n_iter=epochs, verbose=1,
                                     fname_bestmodel=os.path.join(logdir, "model_pure%s.hdf5" % str(fr)))
        np.random.seed(random_state)
        tf.set_random_seed(random_state)
        model_pure = get_model(time_samples_num, channels_num, dropouts=dropouts)
        model_pure.fit(X_train_pure, y_train_pure, epochs=bestepoch,
                       batch_size=64, shuffle=False)
        y_pred_pure = model_pure.predict(X_test)
        y_pred_pure = y_pred_pure[:,1]
        auc_pure = roc_auc_score(y_test,y_pred_pure)

        with open(fname_auc+str(fr)+'.csv', 'a') as fout:
            fout.write(','.join(map(str,[sbj,auc_noisy,auc_pure,samples_before,samples_after])))
            fout.write('\n')
