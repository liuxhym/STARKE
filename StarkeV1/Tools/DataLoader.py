import copy
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from Tools.FileIO import read_file


def libsvm_transformer(f, dim):
    x = []
    y = []
    while f:
        str = f.readline().split()
        if len(str) == 0:
            break
        y.append(float(str[0]))
        xelement = [0 for i in range(dim)]
        for i in range(1, len(str)):
            str1 = str[i].split(":")
            feature = int(str1[0])
            num = float(str1[1])
            xelement[feature - 1] = num
        x.append(xelement)
    return np.array(x), np.array(y)


class DataLoader:
    def __init__(self, DataName, n_samples=100, dim=1):
        self.DataName = DataName
        self.n_samples = n_samples
        self.dim = dim

    def getData(self):
        y = []
        x = []
        if self.DataName == "adult":
            f = open('Data/a1a.txt', 'r')
            x, y = libsvm_transformer(f, 123)
            self.n_samples = len(x)
            return x, y
        if self.DataName == "make_moon":
            X, Y = datasets.make_moons(n_samples=self.n_samples, shuffle=True, noise=0.05, random_state=6)
            X_train = copy.deepcopy(X)
            Y_train = copy.deepcopy(Y)
            Y_train[Y_train == 0] = -1
            return X_train, Y_train  
        if self.DataName == "make_circle":
            X, Y = datasets.make_circles(n_samples=self.n_samples, shuffle=True, noise=0.06, factor=0.5, random_state=6)
            X_train = copy.deepcopy(X)
            Y_train = copy.deepcopy(Y)
            Y_train[Y_train == 0] = -1
            return X_train, Y_train
        if self.DataName == "mpg":
            f = open('Data/mpg_scale.txt', 'r')
            x, y = libsvm_transformer(f, 7)
            self.n_samples = len(x)
            return x, y
        if self.DataName == "sonar":
            f = open('Data/sonar_scale.txt', 'r')
            x, y = libsvm_transformer(f, 60)
            self.n_samples = len(x)
            return x, y
        if self.DataName == "eunite":
            f = open('Data/eunite2001.txt', 'r')
            x, y = libsvm_transformer(f, 16)
            self.n_samples = len(x)
            return x, y
        if self.DataName == "make_regression":
            return datasets.make_regression(n_samples=self.n_samples, n_features=self.dim, random_state=0)
        if self.DataName == "sin":
            x = read_file("Data/sin150_x1.txt")
            y = read_file("Data/sin150_y1.txt")
            self.n_samples = len(x)
            return np.array(x), np.array(y)
