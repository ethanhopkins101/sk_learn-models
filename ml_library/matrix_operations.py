"""
    matrix_operation module : focuses on methods that are relevant for data manipulation
"""
import pandas as pd
import numpy as np
from typing import Union,List
#from scypy.sparse import csr_matrix,spmatrix
# TODO add csr_matrix,spmatrix to MatrixLike

# Defining the MatrixLike alias
MatrixLike=Union[np.ndarray,pd.DataFrame]

#function that changes the learning algorithm inputs to array-like
def change_type(x,y=None):
    if y is None:
        return np.array(x)
    else:
        return np.array(x),np.array(y)
    
#designing intercept_fitting function:
def fit_intercept(x_train):
    intercept=np.ones((x_train.shape[0],1)) #column vector of ones
    return np.hstack((intercept,x_train)) #returning the appended vector with x_train

#arbitrary parameter initializer function:
def init_parameters(x_train):
    return np.zeros((x_train.shape[1],1))#return a column vector of shape x_train[1]


def validate_inputs(inputs: MatrixLike | ArrayLike,
                    target: MatrixLike | ArrayLike) -> None:
    
    if not isinstance(inputs, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if not isinstance(target, np.ndarray):
        raise TypeError("Target must be a numpy array.")
    if (input.ndim > 2) | (target.ndim > 2):
        raise TypeError("Inputs are not of valid dimension.")
    if (target.shape[1] != 1):
        raise TypeError("Target is not of valid shape.")
    if inputs.shape[0] != target.shape[0]:
        raise TypeError("Inputs must be of same shape.")