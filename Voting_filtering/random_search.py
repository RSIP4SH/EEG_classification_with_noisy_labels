from __future__ import print_function
from src.utils import mean_and_pvalue

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