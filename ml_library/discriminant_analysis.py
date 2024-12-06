import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
import pandas as pd

class QuadraticDiscriminantAnalysis:
    """
    Gaussian Discriminant Analysis : Algorithm that uses the naive bayes rule,
    to determine the likelihood of the data
    """
    #initializer
    def __init__(self):
        self.theta=self.mu0=self.mu1=self.sigma=None
    # defining the fit method
    def fit(self,x_train: MatrixLike | ArrayLike,y_train: MatrixLike | ArrayLike):
        x_train,y_train=change_type(x_train,y_train)
        count_positive=sum(y_train) #Nb of positive examples
        count_negative=y_train.shape[0]-count_positive
        self.theta=count_positive/y_train.shape[0]
        self.mu0=np.sum(x_train[np.where(y_train==0)],axis=0)/count_negative
        self.mu1=np.sum(x_train[np.where(y_train==1)],axis=0)/count_positive
        self.sigma=(np.dot((x_train[np.where(y_train==0)-self.mu0]).T,x_train[np.where(y_train==0)]-self.mu0),
                           +np.dot((x_train[np.where(y_train==1)]-self.mu1).T,(x_train[np.where(y_train==1)]-self.mu1)))/x_train.shape[0]
    
    # defining the predict method
    def predict(self,x_test:MatrixLike | ArrayLike)->np.ndarray:

        #probability of x|y which follow multivariate Gaussian distribution
        p_x_given_y0:MatrixLike=np.exp((-1/2)*np.dot(np.dot(x_test-self.mu0,(np.linalg.inv(self.sigma))),(x_test-self.mu0).T))
        p_x_give_y1:MatrixLike=np.exp((-1/2)*np.dot(np.dot(x_test-self.mu1,(np.linalg.inv(self.sigma))),(x_test-self.mu1).T))
        
        #probability of y
        column_vector = np.array((x_test.shape[0],1))
        p_y0 : MatrixLike = column_vector*(1-self.theta)
        p_y1 : MatrixLike = column_vector*self.theta

        #Naive bayes prediction
        probability_y0 : float = np.dot(p_x_given_y0,p_y0)
        probability_y1:float=np.dot(p_x_give_y1,p_y1)
        
        #Defining a probability check method
        def check_probability(probability_matrix : MatrixLike )->np.ndarray:
            predictions : List[int] = []
            for i in probability_matrix.shape[0]:
                if probability_matrix[i,0] > probability_matrix[i,1] :
                    predictions.append(0)
                else : 
                    predictions.append(1)
            return np.array(predictions)
        
        combined : MatrixLike = np.hstack((probability_y0,probability_y1))
        return check_probability(combined)

