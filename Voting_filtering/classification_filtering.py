from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve
import os, sys

if len(sys.argv) < 3:
    print("Usage: \n"
          "python classification_filtering.py path_to_data path_to_logs filtration_rate[0...0.5) \n"
          "For example, if you want to discard 10% of data in each class \n"
          "and then train the network again, use something like: \n"
          "./classification_filtering.py ../Data ./logs/cf 0.1")
    exit()


# Data import and making train, test and validation sets
sbjs = [25,26,27,28,29,30,32,33,34,35,36,37,38]
path_to_data = sys.argv[1] #'/home/likan_blk/BCI/NewData/'  # os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3), resample_to=323)
# Some files for logging
logdir = sys.argv[2]#os.path.join(os.getcwd(),'logs', 'cf_threshold')
if not os.path.isdir(logdir):
    os.makedirs(logdir)
fname = os.path.join(logdir, 'auc_scores.csv')
with open(fname, 'w') as fout:
    fout.write('subject,auc_noisy,auc_pure,samples_before,samples_after,epoch_number\n')
fname_err_ind = os.path.join(logdir, 'err_indices.csv')
with open(fname_err_ind, 'w') as fout:
    fout.write('subject,class,indices\n')

epochs = 150
dropouts = (0.72,0.32,0.05)

if len(sys.argv) > 3:
    filt_rate = sys.argv[3]
else:
    filt_rate = "all"


# Iterate over subjects and clean label noise for all of them
for sbj in sbjs:
    print("Classification filtering for subject %s data"%(sbj))
    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=108)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
    cv = StratifiedKFold(n_splits=4, shuffle=False)

    val_inds = []
    fold_pairs = []
    for tr_ind, val_ind in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_ind], X_train[val_ind]
        y_tr, y_val = y_train[tr_ind], y_train[val_ind]
        fold_pairs.append((X_tr, y_tr, X_val, y_val))
        val_inds.append(train_ind[val_ind]) # indices of all the validation instances in the initial X array

    # Getting and training models with cross-validation
    i = 0 # Fold number iterator
    pure_ind = np.array([], dtype=np.int32)
    err_target_ind = np.array([], dtype=np.int32)
    err_nontarg_ind = np.array([], dtype=np.int32)
    bestepochs = np.array([])
    time_samples_num = X_train.shape[1]
    channels_num = X_train.shape[2]

    for fold in fold_pairs:
        X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
        y_val_bin = fold[3]

        model = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,verbose=1,
                                     fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(i)))
        hist = model.fit(X_tr, y_tr, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[callback],
                        batch_size=64, shuffle=True)
        bestepochs = np.append(bestepochs, callback.bestepoch+1)

        # Validation and data cleaning
        model = load_model(os.path.join(logdir, "model%s.hdf5"%(i)))
        y_pred = model.predict(X_val)[:,1]

        # Choosing threshold (specificity should be at least 0.9)
        FPR, TPR, thresholds = roc_curve(y_val_bin, y_pred)
        threshold = (thresholds[FPR <= 0.1]).min()

        if filt_rate != "all":
            filt_rate = float(filt_rate)
            ind = np.array(val_inds[i]) # indices of validation samples in the initial dataset
            n_err1 = int(np.round(y_val_bin.sum()*filt_rate))  # Number of samples in target class to be thrown away
            n_err0 = int(np.round((len(y_val_bin)-y_val_bin.sum())*filt_rate))  # Number of samples in nontarget class
                                                                        # to be thrown away
            argsort0 = np.argsort(y_pred[y_val_bin==0])[::-1]   # Descending sorting of predictions for nontarget class
                                                            # so that the most erroneous sample are at the
                                                            #begining of the array
            argsort1 = np.argsort(y_pred[y_val_bin==1])     # Ascending Sorting of predictions for target class
                                                        # so that the most erroneous sample are at the
                                                        #begining of the array
            target_ind = ind[y_val_bin==1][argsort1]
            nontarg_ind = ind[y_val_bin==0][argsort0]
            err_target_ind = np.append(err_target_ind, target_ind[:n_err1])
            err_nontarg_ind = np.append(err_nontarg_ind, nontarg_ind[:n_err0]) # Take demanded amount of error samples
            pure_ind = np.append(pure_ind, target_ind[n_err1:])
            pure_ind = np.append(pure_ind, nontarg_ind[n_err0:])
        else:
            # Indices of non-noisy samples
            for j, ind in enumerate(val_inds[i]):
                if y[ind] and y_pred[j] >= threshold or \
                    y[ind] == 0 and y_pred[j] < threshold:
                    pure_ind = np.append(pure_ind, ind)
                #if np.abs(y[ind] - y_pred[j]) < 0.5: # It is useful only if threshold = 0.5
                #    pure_ind.append(ind)
                # OPTIONALLY: let's save indices of erroneous samples for each class separately
                # in order to look at this data after all.
                elif y[ind] == 1:
                    err_target_ind = np.append(err_target_ind, ind)
                else:
                    err_nontarg_ind = np.append(err_nontarg_ind, ind)
            #pure_ind += list(val_inds[i][np.abs(y_val_bin - y_pred) < 0.5])
        i += 1 # Fold number


    bestepoch = int(round(bestepochs.mean()))

    # Removing instances with noisy labels
    np.random.shuffle(pure_ind)
    X_train_pure = X[pure_ind]
    y_train_pure = y[pure_ind]

    # OPTIONALLY: saving erroneous sample indices
    with open(fname_err_ind, 'a') as fout:
        fout.write(str(sbj))
        fout.write(',0,')
        fout.write(','.join(map(str,err_nontarg_ind)))
        fout.write('\n')
        fout.write(str(sbj))
        fout.write(',1,')
        fout.write(','.join(map(str,err_target_ind)))
        fout.write('\n')

    # Testing and comparison of cleaned and noisy data
    samples_before = y_train.shape[0]
    samples_after = y_train_pure.shape[0]
    y_train = to_categorical(y_train)
    y_train_pure = to_categorical(y_train_pure)

    model_noisy = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_noisy.fit(X_train, y_train, epochs=bestepoch,
                    batch_size=64, shuffle=False)

    y_pred_noisy = model_noisy.predict(X_test)
    y_pred_noisy = y_pred_noisy[:,1]
    auc_noisy = roc_auc_score(y_test,y_pred_noisy)

    model_pure = get_model(time_samples_num, channels_num, dropouts=dropouts)
    model_pure.fit(X_train_pure, y_train_pure, epochs=bestepoch,
                   batch_size=64, shuffle=False)
    y_pred_pure = model_pure.predict(X_test)
    y_pred_pure = y_pred_pure[:,1]
    auc_pure = roc_auc_score(y_test,y_pred_pure)

    with open(fname, 'a') as fout:
        fout.write(','.join(map(str,[sbj,auc_noisy,auc_pure,samples_before,samples_after,bestepoch])))
        fout.write('\n')
