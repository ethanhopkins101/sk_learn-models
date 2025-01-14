import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
from typing import Optional
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


class StandardScaler:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray]= None
        self.std: Optional[np.ndarray]= None

    def fit(self, x_train: MatrixLike | ArrayLike) -> None:
        
        self.mean= x_train.mean(axis= 0)
        self.std= x_train.std(axis= 0)

    def transform(self, x_transform: MatrixLike | ArrayLike) -> np.ndarray:

        if (self.mean | self.std) is None:
            raise TypeError('Fit method was not called, parameters are not initialized ')
        return (x_transform - self.mean) / self.std

    def fit_transform(self, x_train: MatrixLike | ArrayLike) -> np.ndarray:

        self.mean= x_train.mean(axis= 0)
        self.std= x_train.std(axis= 0)

        return (x_train - self.mean) / self.std
    
    def inverse_transform(self, x_inverse: MatrixLike | ArrayLike) -> np.ndarray:
        return (x_inverse * self.std) + self.mean
    
    def get_coef(self) -> np.ndarray:
        
        return np.array((self.mean,self.std))

class MinMaxScaler:

    def __init__(self) -> None:
        self.min: Optional[np.ndarray]= None
        self.max: Optional[np.ndarray]= None

    def fit(self, x_train: MatrixLike | ArrayLike) -> None:

        self.min= x_train.min(axis= 0)
        self.max= x_train.max(axis= 0)
    
    def transform(self, x_transform: MatrixLike | ArrayLike) -> np.ndarray:

        return (x_transform - self.min) / (self.max - self.min)

    def fit_transform(self, x_train: MatrixLike | ArrayLike) -> np.ndarray:
        
         self.min= x_train.min(axis= 0)
         self.max= x_train.max(axis= 0)
         return (x_train-self.min) / (self.max - self.min)
    
    def inverse_transform(self, x_inverse: MatrixLike | ArrayLike) -> np.ndarray:

        return x_inverse * (self.max - self.min) + self.min
    
    def get_coef(self) -> np.ndarray:
        return np.array((self.min,self.max))