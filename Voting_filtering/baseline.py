"""
Training baseline algorithm without data filtering with results logging:

auc scores (aucs.csv and test_aucs.csv),
predictions on test (test_predictions.csv),
aggregating CV predictions on all folds (predictions.csv),
true labels for all the validation sets (true_labels.csv),
true labels for the test set (test_true_labels.csv),
indices for all the validation sets (indices.csv),
indices for all the validation sets (test_indices.csv),
deviations of the test predictions from the true labels (test_deviations.csv),
deviations of the CV predictions from the true labels (deviations.csv),
aggregating roc plots (roc/roc*),
test roc plots (roc/test_roc*),
histograms of deviations (hist/*)

"""

from src.utils import *
from src.callbacks import LossMetricHistory
from src.data import DataBuildClassifier
from src.NN import get_model
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.models import load_model

# Data import and making train, test and validation sets
sbjs = [25,26,27,28,29,30,32,33,34,35,36,37,38]
path_to_data = os.path.join(os.pardir,'sample_data') #'/home/likan_blk/BCI/NewData/'  #
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3))
# Some files for logging
logdir = os.path.join(os.getcwd(),'logs', 'cf', 'baseline')
if not os.path.isdir(logdir):
    os.makedirs(logdir)

fname_preds = os.path.join(logdir, 'predictions.csv')
fname_true = os.path.join(logdir, 'true_labels.csv')
fname_dev = os.path.join(logdir, 'deviations.csv')
fname_ind = os.path.join(logdir, 'indices.csv')
fname_auc = os.path.join(logdir, 'aucs.csv')

fname_tpreds = os.path.join(logdir, 'test_predictions.csv')
fname_ttrue = os.path.join(logdir, 'test_true_labels.csv')
fname_tdev = os.path.join(logdir, 'test_deviations.csv')
fname_tind = os.path.join(logdir, 'test_indices.csv')
fname_tauc = os.path.join(logdir, 'test_aucs.csv')

with open(fname_preds, 'w') as fout:
    fout.write('subject,predictions\n')

with open(fname_true, 'w') as fout:
    fout.write('subject,labels\n')

with open(fname_dev, 'w') as fout:
    fout.write('subject,deviations\n')

with open(fname_ind, 'w') as fout:
    fout.write('subject,indices\n')

with open(fname_tpreds, 'w') as fout:
    fout.write('subject,predictions\n')

with open(fname_ttrue, 'w') as fout:
    fout.write('subject,labels\n')

with open(fname_tdev, 'w') as fout:
    fout.write('subject,deviations\n')

with open(fname_tind, 'w') as fout:
    fout.write('subject,indices\n')

if not os.path.isdir(os.path.join(logdir,'roc')):
    os.makedirs(os.path.join(logdir,'roc'))

if not os.path.isdir(os.path.join(logdir,'hist')):
    os.makedirs(os.path.join(logdir,'hist'))

epochs = 150
dropouts = (0.2, 0.4, 0.6)

# Iterate over subjects to train and test models separately
for sbj in sbjs:
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

    with open(fname_preds, 'a') as fout:
        fout.write('%s,'%sbj)

    with open(fname_true, 'a') as fout:
        fout.write('%s,'%sbj)

    with open(fname_dev, 'a') as fout:
        fout.write('%s,'%sbj)

    with open(fname_ind, 'a') as fout:
        fout.write('%s,'%sbj)

    i = 0  # Fold number
    bestepochs = np.array([])
    for fold in fold_pairs:
        X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
        model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,
                                     verbose=1, fname_bestmodel=os.path.join(logdir,"model%s_%s.hdf5"%(sbj,i)))
        hist = model.fit(X_tr, y_tr, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[callback],
                        batch_size=64, shuffle=True)
        bestepochs = np.append(bestepochs, callback.bestepoch + 1)

        # Validation and saving prediction errors
        model = load_model(os.path.join(logdir, "model%s_%s.hdf5"%(sbj,i)))
        y_pred = model.predict(X_val)[:,1]

        with open(fname_preds, 'a') as fout:
            fout.write(','.join(map(str, list(y_pred))))
            fout.write(',')

        with open(fname_true, 'a') as fout:
            fout.write(','.join(map(str, list(y_val[:,1]))))
            fout.write(',')

        with open(fname_dev, 'a') as fout:
            fout.write(','.join(map(str, list(y_val[:,1] - y_pred))))
            fout.write(',')

        with open(fname_ind, 'a') as fout:
            fout.write(','.join(map(str, val_inds[i])))
            fout.write(',')

        i += 1  # Fold number

    bestepoch = int(round(bestepochs.mean()))

    with open(fname_preds, 'a') as fout:
        fout.write('\n')
    with open(fname_true, 'a') as fout:
        fout.write('\n')
    with open(fname_dev, 'a') as fout:
        fout.write('\n')
    with open(fname_ind, 'a') as fout:
        fout.write('\n')



    # Training model on all folds together
    y_train = to_categorical(y_train)

    with open(fname_tpreds, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_ttrue, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tdev, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tind, 'a') as fout:
        fout.write('%s,' % sbj)

    model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
    callback = LossMetricHistory(n_iter=epochs,
                                 verbose=1, fname_lastmodel=os.path.join(logdir, "last_model%s.hdf5" % (sbj)))
    hist = model.fit(X_train, y_train, epochs=bestepoch,
                     batch_size=64, shuffle=True)

    # Testing on a hold-out set and saving prediction errors
    y_pred = model.predict(X_test)[:, 1]
    with open(fname_tpreds, 'a') as fout:
        fout.write(','.join(map(str, list(y_pred))))
        fout.write('\n')

    with open(fname_ttrue, 'a') as fout:
        fout.write(','.join(map(str, list(y_test))))
        fout.write('\n')

    with open(fname_tdev, 'a') as fout:
        fout.write(','.join(map(str, list(y_test - y_pred))))
        fout.write('\n')

    with open(fname_tind, 'a') as fout:
        fout.write(','.join(map(str, test_ind)))
        fout.write('\n')

remove_commas(fname_preds)
remove_commas(fname_true)
remove_commas(fname_dev)
remove_commas(fname_ind)
remove_commas(fname_tpreds)
remove_commas(fname_ttrue)
remove_commas(fname_tdev)
remove_commas(fname_tind)

# Read result files and plot histograms and roc curves
hist_deviations(fname_dev, os.path.join(logdir, 'hist'))
hist_deviations(fname_tdev, os.path.join(logdir, 'hist'), word='test')
roc_curve_and_auc(fname_true, fname_preds, logdir, os.path.join(logdir, 'roc'))
roc_curve_and_auc(fname_ttrue, fname_tpreds, logdir, os.path.join(logdir, 'roc'), word='test')