from typing import Union, Sequence, Optional
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]
def check_inputs(x: ArrayLike, y: ArrayLike, intercept: Optional[bool]= True) -> ArrayLike:
    return np.array(x), np.array(y)


def gradient_ascent(x: ArrayLike, y: ArrayLike, params, step, iterations) -> ArrayLike:
    
    parameters= params
    iterations= iterations
    hypothesis= 1/ (1 + np.exp(- x @ params))
    for i in range(iterations):
        parameters= parameters + step * (x @ (y - hypothesis))

class LinearRegression:
    """
    Ordinary least square Linear Regression

    fits a linear model with coefficients (w0,w1,...wn) to the data
    learns them using the sum squared error loss function
    error: between the true y values and the initialized ones .

    Parameters
    ----------


    Methods
    -------
    fit(x_train, y_train):

    Attributes
    ------
    """

    def __init__(self, fit_intercept: Optional[bool]= True) -> None:
        self.params: np.ndarray= None
        self.coef_: ArrayLike= None
        self.intercept_:float= None
        self.fit_intercept= fit_intercept

    def fit(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        check_inputs(x_train, y_train, self.fit_intercept)
        self.params= np.linalg.solve(x_train.T @ x_train , x_train.T @ y_train)
        self.coef_= self.params[1:]
        self.intercept_= self.params[0]

    def predict(self, x: ArrayLike) -> ArrayLike:
        check_inputs(x)
        return x @ self.params
    
    def __str__(self):
        return f'LinearRegressor trained on SSE Loss function with params {self.params}'
    
class LogisticRegression:
    """
    Logistic regression for binary classification
    This model will use the logistic loss to train the parameters (w0,w1,...wn)
    on the given training data, by initializing the parameters randomly,
    unless provided otherwise

    Parameters
    ----------

    Attributes
    ----------
    
    Methods
    -------

    """

    def __init__(self, fit_intercept:Optional[bool]= True, step:float= 0.01
                 , iterations: int= 10) -> None:
        
        self.params: np.ndarray= None
        self.coef_: ArrayLike= None
        self.intercept_: float= None
        self.fit_intercept= fit_intercept
        self.step= step
        self.iterations= iterations

    def fit(self, x_train:ArrayLike, y_train:ArrayLike) -> None:
        check_inputs(x_train, y_train, self.fit_intercept)
        self.params= np.zeros((x_train.shape[1],1))
        self.params= gradient_ascent(x_train, y_train, self.params, self.step, self.iterations)
        self.coef_= self.params[1:]
        self.intercept_= self.params[0]
    
    def predict(self, x:ArrayLike, probabilities: Optional[bool]= False) -> ArrayLike:
        
        predictions: ArrayLike= x @ self.params
        if probabilities:
            return predictions
        else:
            return (predictions >= 0.5).astype(int)
