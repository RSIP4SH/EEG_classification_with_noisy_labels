import numpy as np
from LatentSVDD.evaluation import KTA_score
from LatentSVDD.preproc import kernel

def majVoteRelabling(labels, latent_states):
    assert labels.shape[0] == latent_states.shape[0], \
        "Shapes of labels and latent states do not match"
    assert set(labels) == {-1, 1}, \
        "Labels should contain only 1 and -1. Got %s instead" % (set(labels))
    states = set(latent_states)
    for state in states:
        maj_label = 1 if labels[latent_states == state].sum() > 0 else -1
        labels[latent_states == state] = maj_label
    return labels

def KTA(data, labels, latent_states):
    assert labels.shape[0] == latent_states.shape[0], \
        "Shapes of labels and latent states do not match"
    assert set(labels) == {-1, 1}, \
        "Labels should contain only 1 and -1. Got %s instead" % (set(labels))
    old_labels = labels.copy()
    K = kernel(data)
    n_states = len(set(latent_states))
    maxKTA = 0
    for i in range(2**n_states):
        binary = ("{:0%db}"%n_states).format(i)
        #print(binary)
        for state in range(n_states):
            label = 1 if binary[state] != '0' else -1
            labels[latent_states == state] = label
        kta = KTA_score(labels, K)
        if kta > maxKTA:
            maxKTA = kta
            best_labels = labels.copy()
    # Inverse labels if more than a half was changed:
    if np.abs(old_labels - best_labels).mean() > 1:
        np.place(best_labels, best_labels == -1, 0)
        np.place(best_labels, best_labels == 1, -1)
        np.place(best_labels, best_labels == 0, 1)
    return maxKTA, best_labels

if __name__ == '__main__':
    y = np.array([1,-1,2,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1])
    z = np.array([0,1,1,0,2,1,0,2,2,0,1,1,2,0,1,2])
    y = np.array([1,-1,-1,1,1,1,1,1])
    z = np.array([0,1,1,0,2,0,2,2])
    kta, ynew = KTA(y,y,z)
    print(kta, ynew)
