from typing import Union, Sequence, Optional, Literal
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]
def check_inputs(x: ArrayLike, y: ArrayLike, intercept: Optional[bool]= True) -> ArrayLike:
    #check validity of x and y (try/except)
    intercept= np.ones((len(x),1))
    return np.array(np.hstack(intercept, x)), np.array(y)


def gradient_ascent(x: ArrayLike, y: ArrayLike, param, step
                    , hypo: Literal['log_reg','poisson']= 'log_reg') -> np.ndarray:
    
    params= param
    if hypo == 'log_reg':
        h= 1/ (1 + np.exp(- x @ params))
        gradient= x @ (y - h)
        while np.linalg.norm(gradient) > 1e-4:
            h= 1/ (1 + np.exp(- x @ params))
            gradient= x.T @ (y - h)
            params= params + step * gradient
    elif hypo =='poisson':
        h= np.exp(x @ params)
        gradient= x.T @ (y - h)
        while np.linalg.norm(gradient) > 1e-4:
            h= np.exp(x @ params)
            gradient= x.T @ (y - h)
            params= params + step * gradient
    return params


def newtons(x: ArrayLike, y:ArrayLike, params: ArrayLike) -> np.ndarray:
    h= 1 / (1 + np.exp(- x.T @ params))
    gradient= x.T @ (y - h)
    hessian= x.T @ np.diag( h * (1-h) ) @ x
    params= params - np.linalg.solve(hessian, gradient)
    return params


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
                 , method: Literal['newtons','gradient_ascent']= 'newtons') -> None:
        
        self.params: np.ndarray= None
        self.coef_: ArrayLike= None
        self.intercept_: float= None
        self.fit_intercept= fit_intercept
        self.step= step
        self.method= method

    def fit(self, x:ArrayLike, y:ArrayLike) -> None:
        check_inputs(x, y, self.fit_intercept)
        self.params= np.zeros((x.shape[1],1))
        if self.method == 'newtons':
            self.params= newtons(x, y, self.params)
        elif self.method == 'gradient_ascent':
            self.params= gradient_ascent(x, y, self.params, self.step, 'log_reg')
        self.coef_= self.params[1:]
        self.intercept_= self.params[0]
    
    def predict(self, x:ArrayLike, probabilities: Optional[bool]= False) -> ArrayLike:
        
        predictions: ArrayLike= x @ self.params
        if probabilities:
            return predictions
        else:
            return (predictions >= 0.5).astype(int)

class LocallyWeightedRegression:

    def __init__(self, t: float= 1) -> None:
        
        self.params: np.ndarray= None
        self.coef_: ArrayLike= None
        self.intercept_: float= None
        self.t= t
        self.weights= None
        self.x= None
        self.y= None
    
    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        
        self.x, self.y= check_inputs(x, y)

    def predict(self, xp: ArrayLike) -> ArrayLike:
        
        xp= np.hstack(np.ones(len(xp),1))
        predictions= []
        for i in range(xp.shape[1]):
            self.weights= np.diag( np.exp(- (np.sum((self.x - xp[i,:])**2, axis=1))/(2*np.pow(self.t,2))) )

            self.params= np.linalg.solve(self.x.T @ self.weights @ self.x
                                        , self.x.T @ self.weights @ self.y)
            predictions.append(xp[i, :] @ self.params)
        return np.array(predictions)
    
class PoissonRegression:
    
    def __init__(self, step: float= 0.01) -> None:

        self.params: np.ndarray= None
        self.coef_: ArrayLike= None
        self.intercept_: float= None
        self.step= step
    
    def fit(self, x: ArrayLike, y: ArrayLike) -> None:

        x, y= check_inputs(x, y)
        self.params= np.zeros((x.shape[1],1))
        self.params= gradient_ascent(x, y, self.params, self.step, 'poisson')
        self.coef_= self.params[1:]
        self.intercept_= self.params[0]

    def predict(self, x) -> None:
        
        return x @ self.params
