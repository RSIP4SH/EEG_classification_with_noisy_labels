import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from src.data import DataBuildClassifier

fname = os.path.join(os.getcwd(), 'logs', 'cf', 'err_indices.csv')

# Rewrite file to get rid of ',' at the ends of some lines
newlines = []
with open(fname, 'r') as f:
    newlines.append(f.readline())
    for line in f.readlines():
        if line[-2] == ',':
            newlines.append(line[:-2]+'\n')
        else:
            newlines.append(line)
with open(fname, 'w') as f:
    f.writelines(newlines)


with open(fname, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    ind = dict()
    for row in csv_reader:
        if i != 0:
            if row[0] not in ind.keys():
                ind[row[0]] = {}
            ind[row[0]][row[1]] = map(int, row[2:])
        i += 1

datadir = '/home/likan_blk/BCI/NewData/'
timewin = (0.0, 0.8)
data = DataBuildClassifier(datadir).get_data(map(int, ind.keys()), shuffle=False,
                                             windows=[timewin],
                                             baseline_window=(0.2, 0.3))
logdir = os.path.join(os.getcwd(), 'logs', 'cf', 'plots')
if not os.path.isdir(logdir):
    os.makedirs(logdir)

# Plot averaged epochs of erroneous and normal samples
for sbj in data.keys():
    X, y = data[int(sbj)][0], data[int(sbj)][1]
    ind_T = np.arange(len(y))[y == 1]  # Indices of target class instances
    ind_NT = np.arange(len(y))[y == 0]  # Indices of non-target class instances
    ind_NT_ok = list(set(ind_NT) - set(ind[str(sbj)]['0'])) # Indices of non-target class instances without an error
    ind_T_ok = list(set(ind_T) - set(ind[str(sbj)]['1']))  # Indices of target class instances without an error
    for channel in range(X.shape[2]):
        plt.title("Averaged epochs")
        if ind[str(sbj)]['0'] != []:
            plt.plot(np.arange(X.shape[1])*timewin[1]*1000/X.shape[1],
                     X[ind[str(sbj)]['0'],:,channel].mean(0),
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
        plt.savefig(os.path.join(logdir, '%sch%ssbj.png'%(channel, sbj)))
        plt.clf()
        plt.cla()
