from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import os

# Data import and making train, test and validation sets
sbj = 33
path_to_data = os.path.join(os.pardir,'sample_data') #'/home/likan_blk/BCI/NewData/'
data = DataBuildClassifier(path_to_data).get_data([sbj],
                                                    shuffle=False,
                                                    windows=[(0.2, 0.5)],
                                                    baseline_window=(0.2, 0.3))

X, y = data[sbj][0], data[sbj][1]
train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                    test_size=0.2, stratify=y,
                                    random_state=8)
X_train, y_train, X_test, y_test = X[train_ind], y[train_ind], X[test_ind], y[test_ind]
cv = StratifiedKFold(n_splits=4)


val_inds = [] 
fold_pairs = []
for tr_ind, val_ind in cv.split(X_train, y_train):
    X_tr, X_val = X_train[tr_ind], X_train[val_ind]
    y_tr, y_val = y_train[tr_ind], y_train[val_ind]
    fold_pairs.append((X_tr, y_tr, X_val, y_val))
    val_inds.append(train_ind[val_ind])# indices of all the validation instances in the initial X array 

# Getting and training models with cross-validation
time_samples_num = X_train.shape[1]
channels_num = X_train.shape[2]
epochs=1
dropouts = (0.2, 0.4, 0.6)
logdir = os.path.join(os.pardir,'logs')
if not os.path.isdir(logdir):
    os.mkdir(logdir)
i = 0
pure_ind = []
bestepochs = np.array([])
for fold in fold_pairs:
    X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
    model, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
    callback = LossMetricHistory(n_iter=epochs, validation_data=(X_val, y_val),
                                 verbose=1, fname_bestmodel=os.path.join(logdir,"model%s.hdf5"%(i)))
    hist = model.fit(X_tr, y_tr, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=[callback],
                    batch_size=64, shuffle=True)
    bestepochs = np.append(bestepochs, callback.bestepoch)

    # Validation and data cleaning
    model = load_model(os.path.join(logdir, "model%s.hdf5" % (i)))
    y_pred = model.predict(X_val)[:,1]
    # Indices of non-noisy samples
    pure_ind += list(val_inds[i][np.abs(y_val[:,1] - y_pred) < 0.5])
    i += 1
bestepoch=int(round(bestepochs.mean()))

# Removing instances with noisy labels
X_train_pure = X[pure_ind]
y_train_pure = y[pure_ind]

# Testing and comparison of cleaned and noisy data
y_train = to_categorical(y_train)
y_train_pure = to_categorical(y_train)

model_noisy, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
model_noisy.fit(X_train, y_train, epochs=bestepoch,
                            batch_size=64, shuffle=True)

y_pred_noisy = model_noisy.predict(X_test)
y_pred_noisy = y_pred_noisy[:,1]
auc_noisy = roc_auc_score(y_test,y_pred_noisy)

model_pure, _ = get_model(time_samples_num, channels_num, dropouts=dropouts)
model_noisy.fit(X_train, y_train, epochs=bestepoch,
                            batch_size=64, shuffle=True)
y_pred_pure = model_pure.predict(X_test)
y_pred_pure = y_pred_pure[:,1]
auc_pure = roc_auc_score(y_test,y_pred_pure)

print (auc_noisy, auc_pure)