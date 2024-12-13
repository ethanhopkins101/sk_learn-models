import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
"""
Module Name: preprociessing.py

Description
-----------
    This module provides various tools/classes to preprocess most data

    
Attributes
-----------
    - List any important constants or variables defined in the module.

Classes
-------
    - PolynomialFeatures: transforms input features into polynomials,
        with specified degree

    Provide a simple example of how to use the module.

"""
class PolynomialFeatures:
    def __init__(self, degree: int, fit_intercept: bool= False):

        self.degree: int= degree

    def fit(self, x_train: MatrixLike | ArrayLike, y_train: MatrixLike | ArrayLike):
        
        pass

    def transform(self, x_test) -> np.ndarray:
        pass


class StandardScalar:
    def __init__(self):
        pass

    def fit(self):
        pass
    def transform(self):
        pass