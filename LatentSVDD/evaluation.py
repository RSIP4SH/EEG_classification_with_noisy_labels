import numpy as np

def KTA_score(y, gram):
    """
    Computes kernel target alignment score.
    :param y: target
    :param gram: Gram matrix
    :return: float
    """
    yyT = np.outer(y,y)
    kta = (gram * yyT).sum() / (y.shape[0] * np.linalg.norm(gram))  # <yyT, yyT>F = len(y)
    return kta

if __name__ == '__main__':
    x = np.array([[1,-2,3],[0,-1,1],[2,1,0]])
    y = np.array([1,-1, 1])
    gram = np.dot(x.T, x)
    print(gram.shape)
    print(gram)
    kta = KTA_score(y, gram)
    print(kta)