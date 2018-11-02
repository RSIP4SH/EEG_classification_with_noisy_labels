from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils import to_categorical
from keras.models import load_model
from src.utils import *
from sklearn.metrics import roc_auc_score


# Data import and making train, test and validation sets
sbjs = [25,26,27,28,29,30,32,33,34,35,36,37,38] #[33,34]
path_to_data = os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3), resample_to=323)
# Some files for logging
logdir = os.path.join(os.getcwd(),'logs', 'cf', 'CV_resampled', '323Hz')

fname_tpreds = [os.path.join(logdir, 'test', 'predictions_ens.csv'),
                os.path.join(logdir, 'test', 'predictions_mean_epoch.csv')]
fname_ttrue = os.path.join(logdir, 'test', 'labels.csv')
fname_tdev = [os.path.join(logdir, 'test', 'deviations_ens.csv'),
                os.path.join(logdir, 'test', 'deviations_mean_epoch.csv')]
fname_tauc = os.path.join(logdir, 'test', 'aucs.csv')
fname_cvauc = os.path.join(logdir, 'test', 'CV_taucs.csv')
if not os.path.isdir(os.path.join(logdir, 'test')):
    os.makedirs(os.path.join(logdir, 'test'))
print(os.path.join(logdir, 'test'))

nsplits = 4 # number of splits in cross-validation
fname_preds = []
fname_true = []
fname_dev = []
fname_ind = []
fname_loss = []
fname_vpreds = []
fname_vtrue = []
fname_vdev = []
fname_vind = []
fname_vauc = []
fname_vloss = []

for i in range(nsplits):
    if not os.path.isdir(os.path.join(logdir, str(i))):
        os.makedirs(os.path.join(logdir, str(i)))
    fname_preds.append(os.path.join(logdir, str(i), 'train_predictions.csv'))
    fname_true.append(os.path.join(logdir, str(i), 'train_true_labels.csv'))
    fname_dev.append(os.path.join(logdir, str(i), 'train_deviations.csv'))
    fname_ind.append(os.path.join(logdir, str(i), 'train_indices.csv'))
    fname_loss.append(os.path.join(logdir, str(i), 'train_loss.csv'))

    fname_vpreds.append(os.path.join(logdir, str(i), 'val_predictions.csv'))
    fname_vtrue.append(os.path.join(logdir, str(i), 'val_true_labels.csv'))
    fname_vdev.append(os.path.join(logdir, str(i), 'val_deviations.csv'))
    fname_vind.append(os.path.join(logdir, str(i), 'val_indices.csv'))
    fname_vauc.append(os.path.join(logdir, str(i), 'val_aucs_dynamics.csv'))
    fname_vloss.append(os.path.join(logdir, str(i), 'val_loss.csv'))

    with open(fname_preds[i], 'w') as fout:
        fout.write('subject,predictions\n')

    with open(fname_true[i], 'w') as fout:
        fout.write('subject,labels\n')

    with open(fname_dev[i], 'w') as fout:
        fout.write('subject,deviations\n')

    with open(fname_ind[i], 'w') as fout:
        fout.write('subject,indices\n')

    with open(fname_vpreds[i], 'w') as fout:
        fout.write('subject,predictions\n')

    with open(fname_vtrue[i], 'w') as fout:
        fout.write('subject,labels\n')

    with open(fname_vdev[i], 'w') as fout:
        fout.write('subject,deviations\n')

    with open(fname_vind[i], 'w') as fout:
        fout.write('subject,indices\n')

    with open(fname_vauc[i], 'w') as fout:
        fout.write('subject,aucs\n')

    with open(fname_loss[i], 'w') as fout:
        fout.write('subject,loss\n')

    with open(fname_vloss[i], 'w') as fout:
        fout.write('subject,loss\n')

with open(fname_tpreds[0], 'w') as fout:
    fout.write('subject,predictions\n')

with open(fname_tdev[0], 'w') as fout:
    fout.write('subject,deviations\n')

with open(fname_tpreds[1], 'w') as fout:
    fout.write('subject,predictions\n')

with open(fname_tdev[1], 'w') as fout:
    fout.write('subject,deviations\n')

with open(fname_tauc, 'w') as fout:
    fout.write('subject,auc_ensemble,auc_mean_epoch\n')

with open(fname_cvauc, 'w') as fout:
    fout.write('subject,aucs\n')

epochs = 150
dropouts = (0.718002897971255, 0.32013533319134346, 0.058501026070547524) #(0.2, 0.4, 0.6)

# Iterate over subjects
for sbj in sbjs:
    print("Classification for subject %s data"%(sbj))
    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=108)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
    cv = StratifiedKFold(n_splits=nsplits, shuffle=False)

    time_samples_num = X_train.shape[1]
    channels_num = X_train.shape[2]

    val_inds = []
    fold_pairs = []

    with open(fname_tauc, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_cvauc, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_ttrue, 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tpreds[0], 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tpreds[1], 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tdev[0], 'a') as fout:
        fout.write('%s,' % sbj)

    with open(fname_tdev[1], 'a') as fout:
        fout.write('%s,' % sbj)

    y_pred_list = []
    n = 0 # number of a split
    for tr_ind, val_ind in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_ind], X_train[val_ind]
        y_tr, y_val = y_train[tr_ind], y_train[val_ind]
        fold_pairs.append((X_tr, y_tr, X_val, y_val))
        val_inds.append(train_ind[val_ind]) # indices of all the validation instances in the initial X array

        # Getting and training models with cross-validation
        with open(fname_preds[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_true[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_dev[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_ind[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_loss[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vpreds[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vtrue[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vdev[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vind[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vloss[n], 'a') as fout:
            fout.write('%s,' % sbj)

        with open(fname_vauc[n], 'a') as fout:
            fout.write('%s,' % sbj)

        bestepochs = np.array([])
        model = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,
                                     verbose=1, fname_bestmodel=os.path.join(logdir, str(n), "model%s.hdf5" % (sbj)))
        model.fit(X_tr, to_categorical(y_tr), epochs=epochs,
                         validation_data=(X_val, to_categorical(y_val)), callbacks=[callback],
                         batch_size=64, shuffle=True)
        bestepochs = np.append(bestepochs, callback.bestepoch + 1)


        # Testing and saving predictions
        model = load_model(os.path.join(logdir, str(n), "model%s.hdf5" % (sbj)))
        y_pred_tr = model.predict(X_tr)[:, 0]
        y_pred_val = model.predict(X_val)[:, 0]
        y_pred_test = model.predict(X_test)
        y_pred_list.append(y_pred_test)
        y_pred_test = y_pred_test[:, 0]

        with open(fname_preds[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_pred_tr))))
            fout.write('\n')

        with open(fname_true[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_tr))))
            fout.write('\n')

        with open(fname_dev[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_tr - y_pred_tr))))
            fout.write('\n')

        with open(fname_ind[n], 'a') as fout:
            fout.write(','.join(map(str, tr_ind)))
            fout.write('\n')

        with open(fname_loss[n], 'a') as fout:
            fout.write(','.join(map(str, list(callback.losses))))
            fout.write('\n')

        with open(fname_vpreds[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_pred_val))))
            fout.write('\n')

        with open(fname_vtrue[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_val))))
            fout.write('\n')

        with open(fname_vdev[n], 'a') as fout:
            fout.write(','.join(map(str, list(y_val - y_pred_val))))
            fout.write('\n')

        with open(fname_vind[n], 'a') as fout:
            fout.write(','.join(map(str, val_ind)))
            fout.write('\n')

        with open(fname_vauc[n], 'a') as fout:
            fout.write(','.join(map(str, list(callback.scores['auc']))))
            fout.write('\n')

        with open(fname_vloss[n], 'a') as fout:
            fout.write(','.join(map(str, list(callback.val_losses))))
            fout.write('\n')

        auc = roc_auc_score(y_test, y_pred_test)
        with open(fname_cvauc, 'a') as fout:
            fout.write('%s,'%auc)

        n += 1
    with open(fname_cvauc, 'a') as fout:
        fout.write('\n')
    bestepoch = int(round(bestepochs.mean()))
    # Test the ensemble model and save predictions
    ensemble(y_pred_list, y_test, fname_tpreds[0], fname_tdev[0], fname_tauc)

    # Train and test the model with the mean number of epochs
    model = get_model(time_samples_num, channels_num, dropouts=dropouts)
    callback = LossMetricHistory(n_iter=epochs,
                                 verbose=1, fname_lastmodel=os.path.join(logdir, "test", "last_model%s.hdf5" % (sbj)))
    model.fit(X_train, to_categorical(y_train), epochs=bestepoch,
                     batch_size=64, shuffle=True)
    y_pred_test = model.predict(X_test)[:, 1] ### Why [:,0] was at first???

    with open(fname_tpreds[1], 'a') as fout:
        fout.write(','.join(map(str, list(y_pred_test))))
        fout.write('\n')

    with open(fname_ttrue, 'a') as fout:
        fout.write(','.join(map(str, list(y_test))))
        fout.write('\n')

    with open(fname_tdev[1], 'a') as fout:
        fout.write(','.join(map(str, list(y_test - y_pred_test))))
        fout.write('\n')

    auc = roc_auc_score(y_test, y_pred_test)
    with open(fname_tauc, 'a') as fout:
        fout.write(str(auc))
        fout.write('\n')
remove_commas(fname_tpreds[0])
remove_commas(fname_tpreds[1])
remove_commas(fname_tdev[0])
remove_commas(fname_tdev[1])
remove_commas(fname_cvauc)
for i in range(nsplits):
    remove_commas(fname_preds[i])
    remove_commas(fname_true[i])
    remove_commas(fname_dev[i])
    remove_commas(fname_ind[i])
    remove_commas(fname_vpreds[i])
    remove_commas(fname_vtrue[i])
    remove_commas(fname_vdev[i])
    remove_commas(fname_vind[i])
    remove_commas(fname_vauc[i])
    remove_commas(fname_loss[i])
    remove_commas(fname_vloss[i])

    # Read result files and plot histograms, roc curves, AUCs and losses
    hist_deviations(fname_dev[i], os.path.join(logdir, str(i), 'hist'))
    hist_deviations(fname_vdev[i], os.path.join(logdir, str(i), 'hist'), word='val')
    roc_curve_and_auc(fname_true[i], fname_preds[i], os.path.join(logdir, str(i)), os.path.join(logdir, str(i), 'roc'), word='train')
    roc_curve_and_auc(fname_vtrue[i], fname_vpreds[i], os.path.join(logdir, str(i)), os.path.join(logdir, str(i), 'roc'), word='val')
    plot_losses(fname_loss[i], fname_vloss[i], os.path.join(logdir, str(i), 'loss'))
    plot_auc(fname_vauc[i], os.path.join(logdir, str(i), 'aucs'))
hist_deviations(fname_tdev[0], os.path.join(logdir, 'test', 'hist_ens'))
hist_deviations(fname_tdev[1], os.path.join(logdir, 'test', 'hist_mean_epoch'))
