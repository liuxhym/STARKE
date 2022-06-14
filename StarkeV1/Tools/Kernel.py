import abc
import copy

import numpy as np
from sklearn.metrics import pairwise_kernels


class BaseKernel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def kernel(self, x1, x2) -> float:
        """

        :param x1:
        :param x2:
        :return:
        """

    @abc.abstractmethod
    def Gram_mat(self, X1, X2):
        """
        :param X1:
        :param X2:
        """


class LinearKernel(BaseKernel):
    def __init__(self):
        self.kernel_name = 'linear'

    def kernel(self, x1, x2):
        return self.Gram_mat([x1], [x2]).flatten()[0]

    def Gram_mat(self, X1, X2):
        return pairwise_kernels(X1, X2, metric='linear')


class PolynominalKernel(BaseKernel):
    def __init__(self, degree=3):
        self.kernel_name = 'poly'
        self.degree = degree

    def kernel(self, x1, x2):
        return self.Gram_mat([x1], [x2]).flatten()[0]

    def Gram_mat(self, X1, X2):
        return pairwise_kernels(X1, X2, metric='poly', degree=self.degree)


class RBFKernel(BaseKernel):
    def __init__(self, gamma=0.2):
        self.kernel_name = 'rbf'
        self.gamma = gamma

    def kernel(self, x1, x2):
        return self.Gram_mat([x1], [x2]).flatten()[0]

    def Gram_mat(self, X1, X2):
        return pairwise_kernels(X1, X2, metric='rbf', gamma=self.gamma)


class LaplacianKernel(BaseKernel):
    def __init__(self):
        self.kernel_name = 'laplacian'

    def kernel(self, x1, x2):
        return self.Gram_mat([x1], [x2]).flatten()[0]

    def Gram_mat(self, X1, X2):
        return pairwise_kernels(X1, X2, metric='laplacian')


class ExponentialKernel(BaseKernel):
    def __init__(self, sigma=0.9):
        self.kernel_name = 'exponential'
        self.sigma = sigma

    def kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) / (2 * (self.sigma ** 2)))

    def Gram_mat(self, X1, X2):
        if X2 is None:
            X2 = copy.deepcopy(X1)
        L = [[0 for i in range(len(X2))] for j in range(len(X1))]
        for i in range(len(X1)):
            for j in range(len(X2)):
                L[i][j] = self.kernel(X1[i], X2[j])
        return np.array(L)
