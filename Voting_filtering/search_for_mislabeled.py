from src.data import DataBuildClassifier
from src.NN import get_model
from src.callbacks import LossMetricHistory
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from keras.utils import to_categorical

# Data import and making train, test and validation sets
sbj = 33
data = DataBuildClassifier('/home/likan_blk/BCI/NewData/').get_data([sbj],
                                                                    shuffle=False,
                                                                    windows=[(0.2, 0.5)],
                                                                    baseline_window=(0.2, 0.3))

X, y = data[sbj][0], data[sbj][1]
train_ind, test_ind = train_test_split(np.arange(len(y)), shuffle=True,
                                    test_size=0.2, stratify=y,
                                    random_state=108)
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
epochs=2
dropouts = (0.2, 0.4, 0.6)
for fold in fold_pairs:
    X_tr, y_tr, X_val, y_val = fold[0], to_categorical(fold[1]), fold[2], to_categorical(fold[3])
    model, _ = get_model(X_tr.shape[1], X_tr.shape[2], dropouts=dropouts)
    callback = LossMetricHistory(n_iter=epochs, validation_data=(X_val, y_val), verbose=1)
    test_history = model.fit(X_tr, y_tr, epochs=epochs,
                            validation_data=(X_val, y_val), callbacks=[callback],
                            batch_size=64, shuffle=True)


# Noise cleaning
pass
# Testing and comparison of cleaned and noisy data
pass
