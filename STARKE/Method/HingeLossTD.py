from sklearn.kernel_approximation import Nystroem

from Tools.SVM import SVM
from Tools.Kernel import *
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class HingeLossTD:
    def __init__(self, kernel, theta_star_C=100, learner_C=1e6, n_components=6, random_state=0):
        self.theta_star_C = theta_star_C
        self.learner_C = learner_C
        self.n_components = n_components
        self.kernel = kernel
        self.x_pool = None
        self.y_pool = None
        self.random_state = random_state

    def calc_teaching_set(self, x, y):
        self.x_pool = copy.deepcopy(x)
        self.y_pool = copy.deepcopy(y)
        clf = SVM(C=self.theta_star_C, kernel=LinearKernel())
        feature_map_nystroem = None
        if self.kernel.kernel_name == 'rbf':
            feature_map_nystroem = Nystroem(gamma=self.kernel.gamma,
                                            random_state=self.random_state,
                                            n_components=self.n_components)
        if self.kernel.kernel_name == 'linear':
            feature_map_nystroem = Nystroem(kernel='linear',
                                            random_state=self.random_state,
                                            n_components=self.n_components)
        if self.kernel.kernel_name == 'poly':
            feature_map_nystroem = Nystroem(kernel='poly',
                                            degree=self.kernel.degree,
                                            random_state=self.random_state,
                                            n_components=self.n_components)
        if self.kernel.kernel_name == 'laplacian':
            feature_map_nystroem = Nystroem(kernel='laplacian',
                                            random_state=self.random_state,
                                            n_components=self.n_components)
        if self.kernel.kernel_name == 'exponential':
            feature_map_nystroem = Nystroem(kernel=self.kernel.kernel,
                                            random_state=self.random_state,
                                            n_components=self.n_components)
        if feature_map_nystroem is None:
            print("No such kernel!")
            return None
        data_transformed = feature_map_nystroem.fit_transform(x)
        clf.fit(data_transformed, y)

        # calculate teaching set
        x_ = feature_map_nystroem.components_
        mat = np.dot(clf.basis, feature_map_nystroem.normalization_)
        alpha = np.sum(clf.dual_coef * mat, axis=0)
        x__ = []
        y_ = []
        flag = []
        for i in range(len(x_)):
            omega = None
            if self.kernel.kernel_name == 'rbf':
                omega = np.dot(alpha, pairwise_kernels([x_[i]], x_, metric='rbf', gamma=self.kernel.gamma)[0])
            if self.kernel.kernel_name == 'linear':
                omega = np.dot(alpha, pairwise_kernels([x_[i]], x_, metric='linear')[0])
            if self.kernel.kernel_name == 'poly':
                omega = np.dot(alpha, pairwise_kernels([x_[i]], x_, metric='poly', degree=self.kernel.degree)[0])
            if self.kernel.kernel_name == 'laplacian':
                omega = np.dot(alpha, pairwise_kernels([x_[i]], x_, metric='laplacian')[0])
            if self.kernel.kernel_name == 'exponential':
                omega = np.dot(alpha, pairwise_kernels([x_[i]], x_, metric=self.kernel.kernel)[0])
            if omega is None:
                print("No such kernel!")
                return None
            if alpha[i] * omega / self.learner_C > 1:
                y_value = 1 / omega
                num = int(alpha[i] / (self.learner_C * y_value))
                for j in range(num):
                    x__.append(x_[i])
                    y_.append(y_value)
                x__.append(x_[i])
                y_.append(alpha[i] / self.learner_C - num * y_value)
                flag.append(num + 1)
            else:
                x__.append(x_[i])
                y_.append(alpha[i] / self.learner_C)
                flag.append(1)
        return np.array(x__), np.array(y_)

    def fit_model_with_teaching_set(self, teaching_set_x, teaching_set_y):
        new_clf = SVM(C=self.learner_C, kernel=self.kernel)
        new_clf.fit(teaching_set_x, teaching_set_y)
        new_clf_Lambda = new_clf.Lambda(self.x_pool, self.y_pool)
        return new_clf, new_clf_Lambda
