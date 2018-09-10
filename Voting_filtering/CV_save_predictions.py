from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import os
import csv
import matplotlib.pyplot as plt
from src.utils import remove_commas


# Data import and making train, test and validation sets
sbjs = [25,26,27,28,29,30,32,33,34,35,36,37,38]
path_to_data = '/home/likan_blk/BCI/NewData/'  # os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3))
# Some files for logging
logdir = os.path.join(os.getcwd(),'logs', 'cf')
if not os.path.isdir(logdir):
    os.mkdir(logdir)

fname = os.path.join(logdir, 'predictions08.csv')

with open(fname, 'w') as fout:
    fout.write('subject,predictions\n')

epochs = 150
dropouts = (0.2, 0.4, 0.6)

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
    time_samples_num = X_train.shape[1]
    channels_num = X_train.shape[2]

    with open(fname, 'a') as fout:
        fout.write('%s,'%sbj)
    i = 0  # Fold number
    for fold in fold_pairs:
        X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
        model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,verbose=1,
                                     fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(i)))
        hist = model.fit(X_tr, y_tr, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[callback],
                        batch_size=64, shuffle=True)
        # Validation and saving prediction errors
        model = load_model(os.path.join(logdir, "model%s.hdf5"%(i)))
        y_pred = model.predict(X_val)[:,1]

        with open(fname, 'a') as fout:
            fout.write(','.join(map(str, list(y_val[:,1] - y_pred))))
            fout.write(',')
        i += 1  # Fold number
    with open(fname, 'a') as fout:
        fout.write('\n')

# Read predictions file and plot histograms
remove_commas(fname)

with open(fname, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    pred = dict()
    for row in csv_reader:
        if i != 0:
            pred[row[0]] = map(float, row[1:])
        i += 1
for sbj in pred.keys():
    pred[sbj] = np.array(pred[sbj])
    pred_T = pred[sbj][pred[sbj]>0]
    pred_NT = np.abs(pred[sbj][pred[sbj]<0])

    plt.title('Histogram of classifier deviation for %s subject'%(sbj))
    plt.xlabel('|True label - Predicted probability|')
    plt.ylabel('number of samples')
    plt.xlim(xmin=0, xmax=1)
    bins = np.arange(0,1.04,0.1)
    plt.xticks(bins)
    plt.hist(pred_T, bins=bins,
             rwidth=0.8, color='#191970', label='target class')
    plt.hist(pred_NT, bins=bins,
             rwidth=0.8, color='#FF1493', label='non-target class', alpha=0.4)
    plt.axvline(x=threshold, color='grey')
    plt.legend()
    plt.savefig(os.path.join(logdir, str(sbj)+'difference_hist.png'))
    plt.clf()
    plt.cla()
