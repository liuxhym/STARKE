import numpy as np


class KRR:
    def __init__(self, C=1):
        self.C = C
        self.w = None
        self.components_ = None

    def fit(self, X, y):
        m = X.shape[0]
        K = np.dot(X, X.T) + self.C * np.identity(m)
        if np.linalg.det(K) == 0:
            inv_mat = np.linalg.pinv(K)
        else:
            inv_mat = np.linalg.inv(K)
        self.w = np.dot(np.dot(X.T, inv_mat), y)

    def predict(self, x):
        val = np.dot(x, self.w)
        return val

    def err(self, xlist, ylist):
        error = 0
        assert len(xlist) == len(ylist)
        for i in range(len(xlist)):
            val = self.predict(xlist[i])
            error += (val - ylist[i]) ** 2
        return error / len(xlist)
