import copy
import numpy as np
import cvxpy as cp


class SVM:
    def __init__(self, kernel, C=1):
        self.C = C
        self.Kernel = kernel
        self.alpha = None
        self.dual_coef = None
        self.basis = None
        self.w = None

    def fit(self, x, label):
        y = np.array(label)
        assert len(x) == len(y)
        m = len(y)
        d = x.shape[1]
        a = cp.Variable(shape=(m), pos=True)
        C = self.C
        G = self.Kernel.Gram_mat(x, x)

        objective = cp.Maximize(cp.sum(a) - (1 / 2) * cp.quad_form(cp.multiply(a, y), G))

        constraints = [a <= C]
        prob = cp.Problem(objective, constraints)
        _ = prob.solve(solver=cp.ECOS)  
        self.alpha = copy.deepcopy(a.value)

        self.w = (label * self.alpha).flatten()
        self.dual_coef = (label * self.alpha).reshape(m, 1).repeat(d, axis=1)
        self.basis = x

    def predict_val(self, x_test):
        K = self.Kernel.Gram_mat(self.basis, x_test)
        val_pred = np.dot(self.w, K)
        return val_pred

    def predict(self, x_test):
        val_pred = self.predict_val(x_test)
        label_pred = copy.deepcopy(val_pred)
        label_pred[label_pred > 0] = 1
        label_pred[label_pred <= 0] = -1
        return label_pred

    def score(self, x_test, y_test):
        assert len(x_test) == len(y_test)
        y_pred = self.predict(x_test)
        tot = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                tot += 1
        return tot * 1.0 / len(x_test)

    def Lambda(self, x_test, y_test):
        assert len(x_test) == len(y_test)
        y_pred = self.predict_val(x_test)
        tot = 0
        for i in range(len(y_pred)):
            result = 1 - y_test[i] * y_pred[i]
            tot += max(0, result)
        return tot
