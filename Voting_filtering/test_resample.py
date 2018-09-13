""" Testing resample function by plotting the data before and after the resampling """
import os
import numpy as np
from src.data import DataBuildClassifier
import matplotlib.pyplot as plt

sbjs=[33]
path_to_data = os.path.join(os.pardir,'sample_data')
data = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3))
data_res = DataBuildClassifier(path_to_data).get_data(sbjs, shuffle=False,
                                                  windows=[(0.2, 0.5)],
                                                  baseline_window=(0.2, 0.3),resample_to=250)
data = data[33][0]
data_res = data_res[33][0]
print("Data shape before resample: %s \n Data shape after resample %s"%(data.shape, data_res.shape))
sample = np.random.randint(0, data.shape[0])
channel = np.random.randint(0, data.shape[2])
plt.title('Resample testing')
plt.plot(np.arange(data.shape[1]), data[sample,:,channel], color='b', label='before resample')
plt.plot(np.arange(data_res.shape[1])*2, data_res[sample,:,channel], color='orange', label = 'after resample')
plt.legend()
plt.show()

