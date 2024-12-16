import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
"""
Module Name: model_selection.py

Description
-----------

    This module provides various tools for model selection such as k-fold,
     cross validation, hyper parameter tunning etc ...

Usage
-----
    Instructions on how to use the module or examples of typical use cases.

Attributes:
    - List any important constants or variables defined in the module.

Classes
-------
    - KFold
    - RepeatedKFold
    - LeavePOut

Functions
---------

    - train_test_split: splits the input data into fragments of kind (x_train, y_train),
        and (x_test, y_test)

"""

def train_test_split(inputs: MatrixLike | ArrayLike,
                     target: MatrixLike | ArrayLike,
                     test_size: float,
                     random_state: int | None = None,
                     stratify: MatrixLike | ArrayLike | None = None,
                     ) -> tuple[np.ndarray]:
    
    validate_inputs(inputs,target)
    holder_matrix: MatrixLike | ArrayLike= np.hstack((inputs,target), axis= 1)
    # shuffling the input data
    np.random.default_rng(seed= random_state).shuffle(holder_matrix, axis= 1)