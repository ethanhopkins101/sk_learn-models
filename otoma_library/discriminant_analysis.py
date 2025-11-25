from typing import Union, Sequence, Optional, Literal, Tuple
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]

def check_inputs(x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    #check validity of x and y (try/except)
    return np.array(x), np.array(y)



class LinearDiscriminantAnalysis:
    """
    LinearDiscriminantAnalysis an unsupervised model that trains the data by
    attempting to fit a gaussian-like model to the data
    then using the probabilities specifically the naive bayes it determines the predictions
    of some y_pred

    Parameters
    ----------
    Attributes
    ----------
    Methods
    -------
    """
    def __init__(self) -> None:
        
        self.theta: float= None
        self.mean0: np.ndarray= None
        self.mean1: np.ndarray= None
        self.sigma: ArrayLike= None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        
        x, y= check_inputs(x, y)

        self.theta= np.sum(y)/len(y)
        self.mean0= np.sum(x[y == 0, :], axis= 0) / len(x[y== 0,:])
        self.mean1= np.sum(x[y == 1, :], axis= 0) / len(x[y== 1,:])
        self.sigma= 