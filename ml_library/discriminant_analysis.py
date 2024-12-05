import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
class QuadraticDiscriminantAnalysis:
    """
    Gaussian Discriminant Analysis : Algorithm that uses the naive bayes rule,
    to determine the likelihood of the data
    """
    #initializer
    def __init__(self,sigma:bool=True):
        self.sigma=True #Equal covariances
        self.theta=self.mu0=self.mu1=None
    # defining the fit method
    def fit(self,x_train:ArrayLike,y_train):
        x_train,y_train=change_type(x_train,y_train)
        count_positive=sum(y_train) #Nb of positive examples
        count_negative=y_train.shape[0]-count_positive
        self.theta=count_positive/y_train.shape[0]
        self.mu0=np.sum(x_train[np.where(y_train==0)])/count_negative
        self.mu1=np.sum(x_train[np.where(y_train==1)])/count_positive
        
        pass
