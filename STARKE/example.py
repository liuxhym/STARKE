from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge

from Method.SquareLossTD import SquareLossTD
from Tools.DataLoader import DataLoader
from Method.HingeLossTD import HingeLossTD
from Tools.Kernel import *

# This is a demo to show how to use Starke.

# 1. we need a dataset

# 2. we can calculate the teaching set with Starke.

# Additionally, we provide 5 kinds of kernels,
# they are: Linear Kernel, Polynomial Kernel, RBF Kernel,
# Laplacian Kernel, Exponential Kernel.

# And the common Gaussian Kernel is achieved by RBF Kernel with some corresponding gamma.

# 3. we can use teaching set to train Learner.

# For hingeloss, we give the following example:
dataloader = DataLoader(DataName="make_moon", n_samples=250)
x, y = dataloader.getData()
# n_components is a parameter to be settled, it will decide the size of teaching set
hingelossTD = HingeLossTD(kernel=RBFKernel(gamma=0.617), n_components=6)
teaching_set_x, teaching_set_y = hingelossTD.calc_teaching_set(x, y)
model, _ = hingelossTD.fit_model_with_teaching_set(teaching_set_x, teaching_set_y)
print("teaching set size:", len(teaching_set_x))
print("accuracy:", model.score(x, y))

# For squareloss, we give the following example:
dataloader = DataLoader(DataName="sin")
x, y = dataloader.getData()
# n_components is a parameter to be settled, it will decide the size of teaching set
squarelossTD = SquareLossTD(kernel=RBFKernel(gamma=0.617), n_components=10)
teaching_set_x, teaching_set_y = squarelossTD.calc_teaching_set(x, y)
model, _ = squarelossTD.fit_model_with_teaching_set(teaching_set_x, teaching_set_y)
print("teaching set size:", len(teaching_set_x))
new_clf = KernelRidge(kernel='rbf', gamma=0.617)
new_clf.fit(teaching_set_x, teaching_set_y)
y_pred = new_clf.predict(x)
mse = metrics.mean_squared_error(y_pred, y)
print("mse:", mse)
