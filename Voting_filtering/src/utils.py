import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import csv
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import wilcoxon

# def loging(history,title):
#     fig = plt.figure()
#     plt.plot(history['loss'], figure=fig)
#     plt.plot(history['val_loss'], figure=fig)
#     plt.plot(history['val_auc'], figure=fig)
#     min_index, min_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
#     plt.title('%s_min_%.3f_epoch%d' % (title, min_value, min_index))
#     plt.savefig('./res/%s.png' % (title), figure=fig)
#     with open('./res/%s.pkl'%title,'wb') as output:
#         pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)


def clean_bad_auc_models(model_path,history):
    '''
    Function clear all,but best (by auc) models in specified dir. Shoud be used with ModelCheckpoint and
    AucMetricHistory callbacks
    :param model_path: Dir, where placed all saved models, their names should be written in {epoch:02d}.hdf5 pattern
    :param history: dict, history of model's learning, which used to find best epoch by auc history
    :return:
    '''
    best_auc_history = np.argmax(history['val_auc']) + 1 #because ModelCheckpoint counts from 1
    best_checpoint = '%02d.hdf5' %best_auc_history
    for checkpoint_name in os.listdir(model_path):
        if checkpoint_name != best_checpoint:
            os.remove(os.path.join(model_path,checkpoint_name))


def single_auc_loging(history,title,path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key],label='cl train loss')
    ax1.plot(history['val_%s' %loss_key],label='cl val loss')
    ax1.legend()
    ax2.plot(history['val_auc'])
    max_index, max_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    plt.title('%s_max_auc_%.3f_epoch%d' % (title, max_value, max_index),loc='right')
    plt.savefig('%s/%s.png' % (path_to_save,title), figure=f)
    plt.close()
    with open('%s/%s.pkl' % (path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)

def multi_auc_loging(history,title,val_subject_numbers,path_to_save):
    """
    Function for ploting nn-classifier performance on different subjects. It makes N poctures,where N is number of
    different val sets (i.e. subjects) Each picture has two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plots as a pictures and as single pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param subject_numbers: list of subject's numbers, used for val sets
    :param path_to_save: Path to save file
    :return:
    """
    aucs = {}
    for subj in val_subject_numbers:
        f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))
        if 'loss' in history.keys():
            loss_key = 'loss'                       #for simple NN
        elif 'class_out_loss' in history.keys():
            loss_key = 'class_out_loss'             #for DAL NN
        else:
            raise ValueError('Not found correct key for loss information in history')
        ax1.plot(history[loss_key],label='train loss')
        ax1.plot(history['val_loss_%d' %subj],label='val loss')
        ax1.legend()
        ax2.plot(history['val_auc_%d' %subj])
        min_loss_index, min_loss_value = min(enumerate(history['val_loss_%d' %subj]), key=operator.itemgetter(1))
        max_auc_index, max_auc_value = max(enumerate(history['val_auc_%d' %subj]), key=operator.itemgetter(1))
        aucs[subj] =  max_auc_value
        plt.title('min_val_loss:%.3f_epoch%d;_max_auc:%.3f_epoch:%d' % (min_loss_value, min_loss_index,
                                                                        max_auc_value, max_auc_index),loc='right')
        plt.savefig('%s/%d.png' % (path_to_save, subj), figure=f)
        plt.close()
    with open('%s/%s.pkl' %(path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)

    with open('%s/res.txt' %(path_to_save),'wb') as f:
        mean_auc = reduce(lambda x,y: x+y,aucs.values())/float(len(aucs))
        for subj in sorted(aucs.keys()):
            f.write('%d %f\n' %(subj,aucs[subj]))
        f.write('Mean AUC %f\n' %mean_auc)

def multisubj_val_split(subjects,val_split=0.2,random_state=42):
    """
    This function make joins data from input subjects and splits them on train and test sets
    :param subjects: dict {subj_number:(x,y)}
    :return: x_train, x_val, y_train, y_val
    """
    tmp = []
    for subj in subjects.keys():
        tmp.append(train_test_split(subjects[subj][0], subjects[subj][1], test_size=val_split,
                                    stratify=subjects[subj][1], random_state=random_state))
    tmp = zip(*tmp)

    x_train = np.concatenate(tmp[0], axis=0)
    x_val = np.concatenate(tmp[1], axis=0)
    y_train = np.concatenate(tmp[2])
    y_val = np.concatenate(tmp[3])
    #save domain info about validation data
    return x_train, x_val, y_train, y_val

def remove_commas(fname):
    ''' Rewriting .csv file to get rid of ',' at the ends of some lines (if necessary) '''
    newlines = []
    with open(fname, 'r') as f:
        newlines.append(f.readline())
        for line in f.readlines():
            if line[-2] == ',':
                newlines.append(line[:-2] + '\n')
            else:
                newlines.append(line)
    with open(fname, 'w') as f:
        f.writelines(newlines)

def plot_EEG(data, logdir, ind, timewin = (0.2,0.5)):
    for sbj in data.keys():
        X, y = data[int(sbj)][0], data[int(sbj)][1]
        ind_T = np.arange(len(y))[y == 1]  # Indices of target class instances
        ind_NT = np.arange(len(y))[y == 0]  # Indices of non-target class instances
        ind_NT_ok = list(
            set(ind_NT) - set(ind[str(sbj)]['0']))  # Indices of non-target class instances without an error
        ind_T_ok = list(set(ind_T) - set(ind[str(sbj)]['1']))  # Indices of target class instances without an error
        for channel in range(X.shape[2]):
            plt.title("Averaged epochs")
            if ind[str(sbj)]['0'] != []:
                plt.plot(np.arange(X.shape[1]) * timewin[1] * 1000 / X.shape[1],
                         X[ind[str(sbj)]['0'], :, channel].mean(0),
                         label='FP', color='y')
            if ind[str(sbj)]['1'] != []:
                plt.plot(np.arange(X.shape[1]) * timewin[1] * 1000 / X.shape[1],
                         X[ind[str(sbj)]['1'], :, channel].mean(0),
                         label='FN', color='g')
            if ind_NT_ok != []:
                plt.plot(np.arange(X.shape[1]) * timewin[1] * 1000 / X.shape[1],
                         X[ind_NT_ok, :, channel].mean(0),
                         label='TN', color='b')
            if ind_T_ok != []:
                plt.plot(np.arange(X.shape[1]) * timewin[1] * 1000 / X.shape[1],
                         X[ind_T_ok, :, channel].mean(0),
                         label='TP', color='r')
            plt.axvline(x=200, color='grey')
            plt.axvline(x=500, color='grey')
            plt.legend()
            plt.savefig(os.path.join(logdir, '%sch%ssbj.png' % (channel, sbj)))
            plt.clf()
            plt.cla()

def hist_deviations(fname, dir_hist, word='',threshold=None):
    '''
    Plotting histograms of deviations of classifier predictions
     from the true labels for each of the two classes for each subjec
    :param fname: name of the .csv file of deviations
    :param dir_hist: name of directory where to save histograms
    :param word: optional, str, additional word to name the resulting histograms. It will be added to the beginning of file name
    :param threshold: optional, if not None - plotting a vertical line x = threshold
    :return: None
    '''
    if not os.path.isdir(dir_hist):
        os.makedirs(dir_hist)
    if word != '':
        word += '_'

    with open(fname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        deviations = dict()
        for row in csv_reader:
            if i != 0:
                deviations[row[0]] = map(float, row[1:])
            i += 1
    for sbj in deviations.keys():
        deviations[sbj] = np.array(deviations[sbj])
        pred_T = deviations[sbj][deviations[sbj] > 0]
        pred_NT = np.abs(deviations[sbj][deviations[sbj] < 0])
        plt.title('Histogram of classifier deviation for %s subject' % (sbj))
        plt.xlabel('|True label - Predicted probability|')
        plt.ylabel('number of samples')
        plt.xlim(xmin=0, xmax=1)
        bins = np.arange(0, 1.04, 0.1)
        plt.xticks(bins)
        plt.hist(pred_T, bins=bins,
                 rwidth=0.8, color='#191970', label='target class')
        plt.hist(pred_NT, bins=bins,
                 rwidth=0.8, color='#FF1493', label='non-target class', alpha=0.4)
        if threshold is not None:
            plt.axvline(x=threshold, color='grey')
        plt.legend()
        plt.savefig(os.path.join(dir_hist, word+"deviations%s.png"%sbj))
        plt.clf()
        plt.cla()

def roc_curve_and_auc(fname_true, fname_pred, dir_auc, dir_roc, word=''):
    '''
    Plot ROC curves and count ROC AUCs
    :param fname_true: str, name of file with true labels for each subject dataset
    :param fname_pred: str, name of file with predictions for each subject dataset
    :param dir_auc: str, name of directory where to put AUC scores
    :param dir_roc: str, name of directory where to put ROC curve plots
    :param word: optional, str, additional word to name the resulting files. It will be added to the beginning of file name
    :return: None
    '''
    if not os.path.isdir(dir_auc):
        os.makedirs(dir_auc)
    if not os.path.isdir(dir_roc):
        os.makedirs(dir_roc)
    if word != '':
        word += '_'

    with open(fname_true, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        y_true = dict()
        for row in csv_reader:
            if i != 0:
                y_true[row[0]] = map(float, row[1:])
            i += 1
    with open(fname_pred, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        y_pred = dict()
        for row in csv_reader:
            if i != 0:
                y_pred[row[0]] = map(float, row[1:])
            i += 1
    with open(os.path.join(dir_auc, word+'aucs.csv'), 'w') as fout:
        fout.write('subject,aucs\n')
    aucs = {}

    for sbj in y_true.keys():
        y_true[sbj] = np.array(y_true[sbj])
        y_pred[sbj] = np.array(y_pred[sbj])

        aucs[sbj] = roc_auc_score(y_true[sbj], y_pred[sbj])
        with open(os.path.join(dir_auc, word + 'aucs.csv'), 'a') as fout:
            fout.write('%s,%s\n'%(sbj,aucs[sbj]))

        FPR, TPR, _ = roc_curve(y_true[sbj], y_pred[sbj])
        plt.title('Roc curve for %s subject' % (sbj))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim(xmin=0, xmax=1)
        plt.plot(FPR, TPR)
        plt.plot(np.array([0,1]), np.array([0,1]), color='grey')
        plt.savefig(os.path.join(dir_roc, '%sroc%s.png'%(word,sbj)))
        plt.clf()
        plt.cla()


def plot_auc(fname, dir_plots, word=''):
    if not os.path.isdir(dir_plots):
        os.makedirs(dir_plots)
    if word != '':
        word += '_'

    with open(fname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        aucs = dict()
        for row in csv_reader:
            if i != 0:
               aucs[row[0]] = map(float, row[1:])
            i += 1

    for sbj in aucs.keys():
        aucs[sbj] = np.array(aucs[sbj])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('ROC AUC dynamics for %s subject' % (sbj))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.plot(np.arange(1,len(aucs[sbj])+1), aucs[sbj])
        xmax = np.argmax(aucs[sbj])
        ymax = np.round(aucs[sbj][xmax],2)
        xmax += 1 # Epoch numbers begin with 1
        ax.plot(xmax, ymax, 'ro')
        ax.annotate('(%s,%s)'%(xmax,ymax), xy=(xmax, ymax))
        ax.plot([1, xmax], [ymax, ymax], '--', color='grey')
        ax.plot([xmax, xmax], [0, ymax], '--', color='grey')
        fig.savefig(os.path.join(dir_plots, '%sauc%s.png'%(word,sbj)))
        ax.cla()
        fig.clf()


def plot_losses(fname1, fname2, dir_plots):
    if not os.path.isdir(dir_plots):
        os.makedirs(dir_plots)

    with open(fname1, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        trloss = dict()
        for row in csv_reader:
            if i != 0:
               trloss[row[0]] = map(float, row[1:])
            i += 1
    with open(fname2, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        tsloss = dict()
        for row in csv_reader:
            if i != 0:
               tsloss[row[0]] = map(float, row[1:])
            i += 1

    for sbj in trloss.keys():
        trloss[sbj] = np.array(trloss[sbj])
        tsloss[sbj] = np.array(tsloss[sbj])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Loss function dynamics for %s subject'%sbj)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.plot(np.arange(1,len(trloss[sbj])+1), trloss[sbj], label='train loss')
        ax.plot(np.arange(1,len(tsloss[sbj])+1), tsloss[sbj], label='test loss')
        plt.legend()
        fig.savefig(os.path.join(dir_plots, 'loss%s.png'%sbj))
        ax.cla()
        fig.clf()


def plot_loss_auc(fname_auc, fname_loss, fname_tloss, dir_plots):
    if not os.path.isdir(dir_plots):
        os.makedirs(dir_plots)

    with open(fname_auc, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        aucs = dict()
        for row in csv_reader:
            if i != 0:
                aucs[row[0]] = map(float, row[1:])
            i += 1
    with open(fname_loss, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        trloss = dict()
        for row in csv_reader:
            if i != 0:
                trloss[row[0]] = map(float, row[1:])
            i += 1
    with open(fname_tloss, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        tsloss = dict()
        for row in csv_reader:
            if i != 0:
                tsloss[row[0]] = map(float, row[1:])
            i += 1
    for sbj in aucs.keys():
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
        ax1.set_title('AUC for %s subject' % sbj)
        ax2.set_title('Loss for %s subject' % sbj)

        aucs[sbj] = np.array(aucs[sbj])
        trloss[sbj] = np.array(trloss[sbj])
        tsloss[sbj] = np.array(tsloss[sbj])

        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('AUC')
        ax2.set_ylabel('Loss')

        ax1.plot(np.arange(1, len(aucs[sbj]) + 1), aucs[sbj])
        xmax = np.argmax(aucs[sbj])
        ymax = aucs[sbj][xmax]
        xmax += 1  # Epoch numbers begin with 1
        ax1.plot(xmax, ymax, 'ro')
        ax1.annotate('(%s,%s)' % (xmax, np.round(ymax, 2)), xy=(xmax, ymax))
        ax1.plot([1, xmax], [ymax, ymax], '--', color='grey')
        ax1.plot([xmax, xmax], [0, ymax], '--', color='grey')

        ax2.plot(np.arange(1, len(trloss[sbj]) + 1), trloss[sbj], label='train loss')
        ax2.plot(np.arange(1, len(tsloss[sbj]) + 1), tsloss[sbj], label='test loss')
        xmin = np.argmin(tsloss[sbj])
        ymin = tsloss[sbj][xmin]
        xmin += 1  # Epoch numbers begin with 1
        ax2.plot(xmin, ymin, 'ro')
        ax2.annotate('(%s,%s)' % (xmin, np.round(ymin, 2)), xy=(xmin, ymin))
        ax2.plot([1, xmin], [ymin, ymin], '--', color='grey')
        ax2.plot([xmin, xmin], [0, ymin], '--', color='grey')
        plt.legend()

        fig.savefig(os.path.join(dir_plots, 'loss_auc%s.png' % sbj))
        ax1.cla()
        ax2.cla()
        fig.clf()

def ensemble(predictions_list, y_true, fname_preds, fname_dev, fname_auc):
    y_pred = reduce(lambda a, b: np.hstack((a,b)), predictions_list)
    y_pred = np.mean(y_pred, axis=1)
    with open(fname_preds, 'a') as fout:
        fout.write(','.join(map(str, list(y_pred))))
        fout.write('\n')
    with open(fname_dev, 'a') as fout:
        fout.write(','.join(map(str, list(y_true - y_pred))))
        fout.write('\n')
    auc = roc_auc_score(y_true, y_pred)
    with open(fname_auc, 'a') as fout:
        fout.write('%s,'%auc)

def mean_and_pvalue(file1, file2=None):
    """
    For auc files compute mean, std of aucs (over subjects) for both files and do the Wilcoxon signed-rank test.
    :param file1: csv file with auc scores of the algorithm of interest in the following format:
                                header
                                sbj1, auc1_1, auc1_2, ..., auc1_n
                                sbj2, auc2_1, auc2_2, ..., auc2_n
                                ...
                                sbjm, aucm_1, aucm_2, ..., aucm_n
    :param file2: csv file with auc scores of an algorithm to compare with (in the same format as file1)
                if None than just mean, std offile1 columns and p-values of Wilcoxon signed-rank test of all
                the columns of file1 comparing to auc*_1 column
    :return: tuple of 3 numpy.ndarray pvalue[n], mean1[n], std1[n], mean2[n], std2[n]
            or mean[n], std[n] if file2 is None
    """
    aucs1 = np.loadtxt(file1, delimiter=',', skiprows=1)[:,1:]

    if not file2:
        pval = []
        for i in range(1,aucs1.shape[1]):
            pval.append(wilcoxon(aucs1[:,0], aucs1[:,i])[1])
        return aucs1.mean(0), aucs1.std(0), pval

    aucs2 = np.loadtxt(file2, delimiter=',', skiprows=1)[:,1:]

    assert aucs1.shape[1] == aucs2.shape[1],\
            "Auc files are incompatible"

    pval = []
    for i in range(aucs1.shape[1]):
        pval.append(wilcoxon(aucs1[:,i], aucs2[:,i])[1])

    return aucs1.mean(0), aucs1.std(0), aucs2.mean(0), aucs2.std(0), np.array(pval)




