# -*- coding: utf-8 -*-

#Test, where classifier trained and tested on same subject throw full CV procedure with train, val and test split
#Simple netwrok, data ebci


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import codecs
import shutil
import os
from src.data_bogdan import DataBuildClassifier
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from src.NN_bogdan import get_model
from src.utils_bogdan import single_auc_loging, clean_bad_auc_models
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
import numpy as np
import pickle

def cv_test(x,y,model,model_path):
    model.save_weights('tmp.h5') # Nasty hack. This weights will be used to reset model
    same_subj_auc = AucMetricHistory()

    folds = 4 #To preserve split as 0.6 0.2 0.2
    x_tr_ind, x_tst_ind, y_tr, y_tst = train_test_split(range(x.shape[0]), y, test_size = 0.2, stratify=y)
    x_tr, x_tst = x[x_tr_ind], x[x_tst_ind]

    cv = StratifiedKFold(n_splits=folds,shuffle=True)
    best_val_epochs = []
    best_val_aucs = []
    # for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr[:,1])):
    for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr)):
        fold_model_path = os.path.join(model_path,'%d' % fold)
        if not os.path.isdir(fold_model_path):
            os.makedirs(fold_model_path)
        make_checkpoint = ModelCheckpoint(os.path.join(fold_model_path, '{epoch:02d}.hdf5'),
                                          monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        model.load_weights('tmp.h5') # Rest model on each fold
        x_tr_fold,y_tr_fold = x_tr[train_idx],y_tr[train_idx]
        x_val_fold, y_val_fold = x_tr[val_idx], y_tr[val_idx]
        val_history = model.fit(x_tr_fold, y_tr_fold, epochs=150, validation_data=(x_val_fold, y_val_fold),
                            callbacks=[same_subj_auc,make_checkpoint], batch_size=64, shuffle=True)
        best_val_epochs.append(np.argmax(val_history.history['val_auc']) + 1) # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))
        clean_bad_auc_models(fold_model_path, val_history.history)


    #Test  performance (Naive, until best epoch
    model.load_weights('tmp.h5') # Rest model before traning on train+val

    test_history = model.fit(x_tr, y_tr, epochs=int(np.mean(best_val_epochs)),
                        validation_data=(x_tst, y_tst),callbacks=[same_subj_auc],batch_size=64, shuffle=True)
    model.save(os.path.join(model_path,'final_%d.hdf5' %int(np.mean(best_val_epochs))))

    with open(os.path.join(model_path,'tessting_data.pkl'), 'wb') as output:
        pickle.dump((x_tst, y_tst),output,pickle.HIGHEST_PROTOCOL)

    os.remove('tmp.h5')

    # Test  performance (ensemble)
    best_models = []
    predictions = np.zeros_like(y_tst)
    for fold_folder in os.listdir(model_path):
        fold_model_path = os.path.join(model_path,fold_folder)
        if os.path.isdir(fold_model_path):
            model_checkpoint = os.listdir(fold_model_path)[0]
            fold_model_path = os.path.join(fold_model_path,model_checkpoint)
            # best_models.append(load_model(fold_model_path))
            predictions+=load_model(fold_model_path).predict(x_tst)[:,0]

    predictions /= (folds)
    test_auc_ensemble = roc_auc_score(y_tst,predictions)#roc_auc_score(y_tst[:,1],predictions[:,1])


    return np.mean(best_val_aucs),np.std(best_val_aucs), test_history.history,test_auc_ensemble



if __name__=='__main__':
    random_state=42
    all_subjects_indeces = range(16)
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    all_subjects = [25,26,27,28,29,30,32,33,34,35,36,37,38]
    subjects = data.get_data(all_subjects,shuffle=False, windows=[(0.2,0.5)],baseline_window=(0.2, 0.3),resample_to=250)
    dropouts = (0.2,0.4,0.6)
    # subjects_sets = [(33, 34), (35, 36), (37, 38)]
    subjects_sets = [25,26,27,28,29,30,32,33,34,35,36,37,38]
    mean_val_aucs=[]
    test_aucs_naive = []
    test_aucs_ensemble = []
    for train_subject in subjects_sets:
        path_to_save = './res/cv_simple_ebci/%s' % train_subject
        model_path = os.path.join(path_to_save,'checkpoints')
        if not os.path.isdir(path_to_save):
            os.makedirs(model_path)
        x = subjects[train_subject][0]
        # y = to_categorical(subjects[train_subject][1],2)
        y = subjects[train_subject][1]
        model= get_model(x.shape[1], x.shape[2],dropouts=dropouts)
        mean_val_auc,std_val_auc, test_histpory,test_auc_ensemble = cv_test(x, y, model,model_path)
        path_to_save = './res/cv_simple_ebci/%s' % train_subject
        hyperparam_name = 'DO_%s' %('_'.join([str(dropout) for dropout in dropouts]))
        plot_name = '%s_%.02f_%d' %(hyperparam_name,test_histpory['val_auc'][-1],len(test_histpory['val_auc']))
        #if os.path.isdir(path_to_save):
        #    shutil.rmtree(path_to_save)
        single_auc_loging(test_histpory, plot_name, path_to_save=path_to_save)
        mean_val_aucs.append((mean_val_auc,std_val_auc))
        test_aucs_naive.append(test_histpory['val_auc'][-1])
        test_aucs_ensemble.append(test_auc_ensemble)
        with codecs.open('%s/res.txt' %path_to_save,'w', encoding='utf8') as f:
            f.write(u'Val auc %.04f±%.04f\n' %(mean_val_auc,std_val_auc))
            f.write('Test auc naive %.04f\n' % (test_histpory['val_auc'][-1]))
            f.write('Test auc ensemble %.04f\n' % test_auc_ensemble)

    general_res_path = './res/cv_simple_ebci/'
    with codecs.open('%s/res.txt' %general_res_path,'w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs,test_aucs_naive,test_aucs_ensemble\n')
        for tr_subj_idx, tr_subj in enumerate(subjects_sets):
            f.write(u'%d, %.04f±%.04f, %.04f, %.04f\n' \
                    % (tr_subj,mean_val_aucs[tr_subj_idx][0],mean_val_aucs[tr_subj_idx][1],test_aucs_naive[tr_subj_idx],
                       test_aucs_ensemble[tr_subj_idx]))

        final_val_auc = np.mean(zip(*mean_val_aucs)[0])
        final_auc_naive = np.mean(test_aucs_naive)
        final_auc_ensemble = np.mean(test_aucs_ensemble)
        final_val_auc_std = np.std(zip(*mean_val_aucs)[0])
        final_auc_naive_std = np.std(test_aucs_naive)
        final_auc_ensemble_std = np.std(test_aucs_ensemble)
        f.write(u'MEAN, %.04f±%.04f, %.04f±%.04f, %.04f±%.04f\n' \
                % (final_val_auc,final_val_auc_std,final_auc_naive,final_auc_naive_std,final_auc_ensemble,
                   final_auc_ensemble_std))


###############################################################################################################