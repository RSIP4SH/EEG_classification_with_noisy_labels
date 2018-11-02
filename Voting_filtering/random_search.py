from __future__ import print_function
import numpy as np
from src.utils import mean_and_pvalue

from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import load_model
from src.utils import *
from sklearn.metrics import roc_auc_score

def reset_params(params):
    params['dropout'] = [np.random.rand(), np.random.rand(), np.random.rand()]
    params['l1_l2'] = np.random.rand()*2e-4
    params['sample_rate'] = np.random.choice([0,200,250])

params = dict()
N = 10#00 # number of parameter sets
nsplits = 4 # number of splits in cross-validation
epochs=3#150
path_to_data = os.path.join(os.pardir,'sample_data')
sbjs = [25,26]#,27,28,29,30,32,33,34,35,36,37,38]
for i in range(N):
    reset_params(params)

    logdir = os.path.join(os.getcwd(), 'logs', 'cf', 'random_search')
                          #u'%.03f_%.03f_%.03f_%.02f_%d'%(params['dropout'][0], params['dropout'][0], params['dropout'][0],
                          #                              params['l1_l2']*1e4, params['sample_rate']))

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    fname = os.path.join(logdir, 'params_pvalues.txt')
    with open(fname,'w') as fout:
        fout.write('id,\t\tdropout1,\t\tdropout2,\t\tdropout3,\t\tl1_l2,\t\tsample rate,\t\tp-value\n')
    logdir = os.path.join(logdir, str(i))
    os.mkdir(logdir)
    data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                      windows=[(0.2, 0.5)],
                                                      baseline_window=(0.2, 0.3), resample_to=params['sample_rate'])
    for sbj in sbjs:
        X, y = data[sbj][0], data[sbj][1]
        train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                               test_size=0.2, stratify=y,
                                               random_state=108)
        X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
        cv = StratifiedKFold(n_splits=nsplits, shuffle=False)

        time_samples_num = X_train.shape[1]
        channels_num = X_train.shape[2]

        #val_inds = []
        fold_pairs = []
        n = 0
        for tr_ind, val_ind in cv.split(X_train, y_train):
            X_tr, X_val = X_train[tr_ind], X_train[val_ind]
            y_tr, y_val = y_train[tr_ind], y_train[val_ind]
            fold_pairs.append((X_tr, y_tr, X_val, y_val))
            #val_inds.append(train_ind[val_ind])  # indices of all the validation instances in the initial X array
            
            bestepochs = np.array([])
            model, _ = get_model(time_samples_num, channels_num, dropouts=params['dropout'])
            callback = LossMetricHistory(n_iter=epochs,
                                         verbose=1,
                                         fname_bestmodel=os.path.join(logdir, str(n), "model%s.hdf5" % (sbj)))
            model.fit(X_tr, y_tr, epochs=epochs,
                      validation_data=(X_val, y_val), callbacks=[callback],
                      batch_size=64, shuffle=True)
            bestepochs = np.append(bestepochs, callback.bestepoch + 1)
            # Testing and saving predictions
            model = load_model(os.path.join(logdir, str(n), "model%s.hdf5" % (sbj)))
            y_pred_tr = model.predict(X_tr)[:, 0]
            y_pred_val = model.predict(X_val)[:, 0]
            y_pred_test = model.predict(X_test)
            #y_pred_list.append(y_pred_test)
            y_pred_test = y_pred_test[:, 0]
            auc = roc_auc_score(y_test, y_pred_test)
            with open(os.path.join(logdir,'testaucs.csv'), 'a') as fout:
                fout.write('%s,' % auc)
            n += 1
        with open(fname_cvauc, 'a') as fout:
            fout.write('\n')
        bestepoch = int(round(bestepochs.mean()))
        # Test the ensemble model and save predictions
        ensemble(y_pred_list, y_test, fname_tpreds[0], fname_tdev[0], fname_tauc)
        # Train and test the model with the mean number of epochs
        model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
        callback = LossMetricHistory(n_iter=epochs,
                                     verbose=1,
                                     fname_lastmodel=os.path.join(logdir, "test", "last_model%s.hdf5" % (sbj)))
        model.fit(X_train, y_train, epochs=bestepoch,
                  batch_size=64, shuffle=True)
        y_pred_test = model.predict(X_test)[:, 0]
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
'''
file1 = '/home/moskaleona/alenadir/GitHub/EEG_classification_with_noisy_labels/Voting_filtering/logs/cf/CV_resampled/200Hz/test/aucs.csv'
file2 = '/home/moskaleona/alenadir/GitHub/EEG_classification_with_noisy_labels/Voting_filtering/logs/cf/CV_resampled/250Hz/test/aucs.csv'
file3 = '/home/moskaleona/alenadir/GitHub/EEG_classification_with_noisy_labels/Voting_filtering/logs/cf/CV/test/aucs.csv'

res = mean_and_pvalue(file1, file2)
print(u"For naive \n Mean1 %0.4f +- %0.4f, mean2 %0.4f +- %0.4f \n"%(res[1][0],res[2][0],res[3][0],res[4][0]),
      u"p-value %0.4f"%(res[0][0]))
print(u"For ensemble \n Mean1 %0.4f +- %0.4f, mean2 %0.4f +- %0.4f \n"%(res[1][1],res[2][1],res[3][1],res[4][1]),
      u"p-value %0.4f\n"%(res[0][1]))

res = mean_and_pvalue(file2, file3)
print(u"For naive \n Mean1 %0.4f +- %0.4f, mean2 %0.4f +- %0.4f \n"%(res[1][0],res[2][0],res[3][0],res[4][0]),
      u"p-value %0.4f"%(res[0][0]))
print(u"For ensemble \n Mean1 %0.4f +- %0.4f, mean2 %0.4f +- %0.4f \n"%(res[1][1],res[2][1],res[3][1],res[4][1]),
      u"p-value %0.4f\n"%(res[0][1]))

'''





