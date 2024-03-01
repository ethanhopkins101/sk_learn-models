import numpy as np


class PolynomialFeatures:
    def __init__(degree=2):
        self.degree = degree

    def fit(self, x):
        x = np.array(x)
        intercept = np.ones((x.shape[0], 1))
        table = np.zeros((x.shape[0], (self.degree - 1) * x.shape[1]))
        if x[:, 0] != intercept:
            x = np.concat(intercept, x, axis=1)
        j=1
        for i in range(table.shape[1]):
            table[:, i] = x[:,j]**
