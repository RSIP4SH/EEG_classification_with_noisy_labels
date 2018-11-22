from src.utils import mean_and_pvalue
import sys

fname = sys.argv[1] #'/home/moskaleona/alenadir/GitHub/EEG_classification_with_noisy_labels/Voting_filtering/logs/cf_threshold/auc_scores.csv'
res = mean_and_pvalue(fname)
with open(fname, 'a') as f:
    f.write(u"Mean %0.4f+-%0.4f %0.4f+-%0.4f \n"%(res[0][0],res[1][0], res[0][1], res[1][1])+
            u"P-value: %0.4f"%(res[2][1]))
