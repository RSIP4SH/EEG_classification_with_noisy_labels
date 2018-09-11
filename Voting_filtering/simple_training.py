"""
Training and testing baseline algorithm without data filtering with results logging:

auc scores (aucs.csv and test_aucs.csv),
predictions on test (test_predictions.csv),
aggregating CV predictions on all folds (predictions.csv),
#true labels for all the validation sets (true_labels.csv),
true labels for the test set (test_true_labels.csv),
indices for all the validation sets (indices.csv),
indices for all the validation sets (test_indices.csv),
deviations of the test predictions from the true labels (test_deviations.csv),
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
sbjs = [25,26]#,27,28,29,30,32,33,34,35,36,37,38]
path_to_data = os.path.join(os.pardir,'sample_data') #'/home/likan_blk/BCI/NewData/'  #
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3))
# Some files for logging
logdir = os.path.join(os.getcwd(),'logs', 'cf', 'simple_training')
if not os.path.isdir(logdir):
    os.makedirs(logdir)

fname_preds = os.path.join(logdir, 'train_predictions.csv')
fname_true = os.path.join(logdir, 'train_true_labels.csv')
fname_dev = os.path.join(logdir, 'train_deviations.csv')
fname_ind = os.path.join(logdir, 'train_indices.csv')
fname_loss = os.path.join(logdir, 'train_loss.csv')

fname_tpreds = os.path.join(logdir, 'test_predictions.csv')
fname_ttrue = os.path.join(logdir, 'test_true_labels.csv')
fname_tdev = os.path.join(logdir, 'test_deviations.csv')
fname_tind = os.path.join(logdir, 'test_indices.csv')
fname_tauc = os.path.join(logdir, 'test_aucs.csv')
fname_tloss = os.path.join(logdir, 'test_loss.csv')

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

with open(fname_tauc, 'w') as fout:
    fout.write('subject,aucs\n')

with open(fname_loss, 'w') as fout:
    fout.write('subject,loss\n')

with open(fname_tloss, 'w') as fout:
    fout.write('subject,loss\n')

epochs = 2#150
dropouts = (0.2, 0.4, 0.6)

# Iterate over subjects to train and test models separately
for sbj in sbjs:
    X, y = data[sbj][0], data[sbj][1]
    train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                           test_size=0.2, stratify=y,
                                           random_state=108)
    X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]

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

    with open(fname_loss, 'a') as fout:
        fout.write('%s,'%sbj)

    with open(fname_tloss, 'a') as fout:
        fout.write('%s,'%sbj)

    bestepochs = np.array([])
    model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
    callback = LossMetricHistory(n_iter=epochs,
                                 verbose=1, fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(sbj)))
    print(y_train.shape, y_test.shape, X_train.shape, X_test.shape)
    hist = model.fit(X_train, to_categorical(y_train), epochs=epochs,
                    validation_data=(X_test, to_categorical(y_test)), callbacks=[callback],
                    batch_size=64, shuffle=True)
    bestepoch = callback.bestepoch + 1

    # Testing and saving predictions
    model = load_model(os.path.join(logdir, "model%s.hdf5"%(sbj)))
    y_pred_test = model.predict(X_test)[:,1]
    y_pred_train = model.predict(X_train)[:,1]

    with open(fname_preds, 'a') as fout:
        fout.write(','.join(map(str, list(y_pred_train))))
        fout.write('\n')

    with open(fname_true, 'a') as fout:
        fout.write(','.join(map(str, list(y_train))))
        fout.write('\n')

    with open(fname_dev, 'a') as fout:
        fout.write(','.join(map(str, list(y_train - y_pred_train))))
        fout.write('\n')

    with open(fname_ind, 'a') as fout:
        fout.write(','.join(map(str, train_ind)))
        fout.write('\n')

    with open(fname_tpreds, 'a') as fout:
        fout.write(','.join(map(str, list(y_pred_test))))
        fout.write('\n')

    with open(fname_ttrue, 'a') as fout:
        fout.write(','.join(map(str, list(y_test))))
        fout.write('\n')

    with open(fname_tdev, 'a') as fout:
        fout.write(','.join(map(str, list(y_test - y_pred_test))))
        fout.write('\n')

    with open(fname_tind, 'a') as fout:
        fout.write(','.join(map(str, test_ind)))
        fout.write('\n')

    with open(fname_tauc, 'a') as fout:
        fout.write(','.join(map(str, list(callback.scores['auc']))))
        fout.write('\n')

    with open(fname_loss, 'a') as fout:
        fout.write(','.join(map(str, list(callback.losses))))
        fout.write('\n')

    with open(fname_tloss, 'a') as fout:
        fout.write(','.join(map(str, list(callback.val_losses))))
        fout.write('\n')


remove_commas(fname_preds)
remove_commas(fname_true)
remove_commas(fname_dev)
remove_commas(fname_ind)
remove_commas(fname_tpreds)
remove_commas(fname_ttrue)
remove_commas(fname_tdev)
remove_commas(fname_tind)
remove_commas(fname_tauc)
remove_commas(fname_loss)
remove_commas(fname_tloss)

# Read result files and plot histograms, roc curves, AUCs and losses
#hist_deviations(fname_dev, os.path.join(logdir, 'hist'))
#hist_deviations(fname_tdev, os.path.join(logdir, 'hist'), word='test')
roc_curve_and_auc(fname_true, fname_preds, logdir, os.path.join(logdir, 'roc'))
roc_curve_and_auc(fname_ttrue, fname_tpreds, logdir, os.path.join(logdir, 'roc'), word='test')
plot_losses(fname_loss, fname_tloss, os.path.join(logdir,'loss'))
plot_auc(fname_tauc, os.path.join(logdir, 'aucs'))