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
        temp= x.copy()
        for i in range(len(y)):
            if y[i] == 0:
                temp[i, :]= temp[i, :] - self.mean0
            else:
                temp[i, :]= temp[i, :] - self.mean1

        self.sigma= (temp.T @ temp) / len(x)

    def predict(self, xp: ArrayLike) -> np.ndarray :
        
        res0= []
        res1= []
        denominator0= 1/ (np.pow(2*np.pi,xp.shape[1]/2) * np.pow(np.linalg.det(self.sigma),1/2))
        denominator1= 1/ (np.pow(2*np.pi,xp.shape[1]/2) * np.pow(np.linalg.det(self.sigma),1/2))
        
        for i in range(len(xp)):
            numerator0= np.exp(- (1/2) * (xp[i, :] - self.mean0) @ np.linalg.inv(self.sigma) @ (xp[i, :] - self.mean0).T )
            numerator1= np.exp(- (1/2) * (xp[i, :] - self.mean1) @ np.linalg.inv(self.sigma) @ (xp[i, :] - self.mean1).T )

            res0.append(numerator0)
            res1.append(numerator1)

        px_y0= denominator0 * np.array(res0)
        px_y1= denominator1 * np.array(res1)
        py0= self.theta
        py1= 1 - self.theta

        py_x0= px_y0 * py0
        py_x1= px_y1 * py1

        result= [1 if py_x1[i] > py_x0[i] else 0 for i in range(len(py_x0))]

        return result