from sklearn import metrics
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge

from Tools.KRR import KRR
from Tools.Kernel import *
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class SquareLossTD:
    def __init__(self, kernel, theta_star_C=1, learner_C=1, n_components=10, random_state=0):
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
        model = KRR(C=self.theta_star_C)
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
        model.fit(data_transformed, y)

        # calculate teaching set
        x_ = feature_map_nystroem.components_
        alpha = np.dot(feature_map_nystroem.normalization_, model.w)
        y_ = []
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
            y_.append(alpha[i] + omega)
        return np.array(x_), np.array(y_)

    def fit_model_with_teaching_set(self, teaching_set_x, teaching_set_y):
        new_model = None
        if self.kernel.kernel_name == 'rbf':
            new_model = KernelRidge(kernel='rbf', gamma=self.kernel.gamma)
        if self.kernel.kernel_name == 'linear':
            new_model = KernelRidge(kernel='linear')
        if self.kernel.kernel_name == 'poly':
            new_model = KernelRidge(kernel='poly', degree=self.kernel.degree)
        if self.kernel.kernel_name == 'laplacian':
            new_model = KernelRidge(kernel='laplacian')
        if self.kernel.kernel_name == 'exponential':
            new_model = KernelRidge(kernel=self.kernel.kernel)
        if new_model is None:
            print("No such kernel!")
            return None

        new_model.fit(teaching_set_x, teaching_set_y)
        y_pred = new_model.predict(self.x_pool)
        mse = metrics.mean_squared_error(y_pred, self.y_pool)
        return new_model, mse
