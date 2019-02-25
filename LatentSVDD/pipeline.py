# Step 0: Data loading and preprocessing
# Step 1: LatentSVDD
# Step 2: Outlier removal
# Step 3: Relabeling
# Step 4: Evaluation

from LatentSVDD.preproc import *
from LatentSVDD.params import *


if __name__=='__main__':
    data = DataLoader(PATH_TO_DATA).get_data(SUBJECTS, window=(0.2,0.5), baseline_window=(0.2,0.3))
    for sbj in SUBJECTS:
        X, y = data[sbj][0], data[sbj][1]
        features = get_feature_vectors(X)
        print(features.shape)
