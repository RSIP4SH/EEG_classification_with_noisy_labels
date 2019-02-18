import numpy as np

def KTA_score(y, gram=None):
    yyT = np.outer(y,y)
    kta = (gram * yyT).sum() / (y.shape[0] * np.linalg.norm(gram))  # <yyT, yyT>F = len(y)
    return kta
