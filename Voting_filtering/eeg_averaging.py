import os
import csv
from src.data import DataBuildClassifier
from src.utils import remove_commas, plot_EEG

fname = os.path.join(os.getcwd(), 'logs', 'cf', 'err_indices.csv')

remove_commas(fname) # Rewrite file to get rid of ',' at the ends of some lines

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
logdir = os.path.join(os.getcwd(), 'logs', 'cf', 'plots', 'test')
if not os.path.isdir(logdir):
    os.makedirs(logdir)

# Plot averaged epochs of erroneous and normal samples
plot_EEG(data, logdir, ind, timewin)
