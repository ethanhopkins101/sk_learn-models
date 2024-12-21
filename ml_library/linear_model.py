import numpy as np
from matrix_operations import *
from typing import Optional
from numpy.typing import ArrayLike
#Linear_Regression
class LinearRegression:
    """
    Least Mean Square LinearRegression
    """
    def __init__(self,fit_intercept=False):
        self.fit_intercept: bool= fit_intercept
        self.parameters: Optional[MatrixLike | ArrayLike]= None

    def fit(self,x_train: MatrixLike | ArrayLike,
            y_train: MatrixLike | ArrayLike) -> None:
        """
        Parameters
         ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
    
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
     
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        x_train,y_train= change_type(x_train,y_train)
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train= fit_intercept(x_train)
        #Designing the learning algorithm (normal equations: o=(x.T*x).inv*x.T*y)
        self.parameters: MatirxLike | ArrayLike= np.dot(np.linalg.inv(np.dot(x_train.T,x_train)),np.dot(x_train.T,y_train))
    
    def predict(self, x_test: MatrixLike | ArrayLike) -> np.ndarray:

        if self.parameters is None :
            raise ValueError('Fit method was not called'
                             'Training data was not provided !')

        x_test= change_type(x_test)
        if self.fit_intercept:
            x_test= fit_intercept(x_test)
        return np.dot(x_test, self.parameters) # (mx1) column vector
    
    @property
    def coef_(self) -> np.ndarray:
        return self.parameters[1:] # returns the 'coef' except for the intercept
    
    @property
    def intercept_(self) -> float:
        return self.parameters[0] # returns the 'intercept'



class LocallyWeightedRegression:

    def __init__(self,tunning_parameter,fit_intercept=False):
        self.tuning_parameter: float= tunning_parameter
        self.fit_intercept: bool= fit_intercept
        self.x_train: Optional[MatrixLike]= None
        self.y_train: Optional[MatrixLike]= None

    def fit(self, x_train: MatrixLike | ArrayLike,
            y_train: MatrixLike | ArrayLike) -> None:
        
        validate_inputs(x_train, y_train)

        x_train,y_train=change_type(x_train,y_train)
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=fit_intercept(x_train)

        self.x_train, self.y_train= x_train, y_train

    def predict(self, x_test: MatrixLike | ArrayLike) -> np.ndarray:

        #Checking if user provided training-sets already
        if self.x_train is None:
            raise ValueError('Fit method was not called'
                'Training data was not provided !')
        
        x_test= change_type(x_test)
        #Adding intercept term if specified by the user
        if self.fit_intercept :
            x_test=fit_intercept(x_test)

        t_square=np.square(self.tuning_parameter)
        predictions: List[float]= [] # empty prediction list

        #designing the algorithm using normal equations
        for i in range(x_test.shape[0]): #Looping through the prediction points 
            weights=[]
            for j in range(self.x_train.shape[0]): #Looping through our training points to formulate the weights
                distance=np.sum(np.square((self.x_train[j,:]-x_test[i,:])))
                weight=np.exp(distance/(2*t_square))
                weights.append(weight)
            weights=np.multiply(weights,np.identity(len(weights))) # Weight matrix (mxm)
            parameters=np.dot(np.linalg.inv(np.dot(self.x_train.T,np.dot(weights,self.x_train))),np.dot(self.x_train.T,np.dot(weights,self.y_train)))
            predictions.append(np.dot(x_test[i,:],parameters))

        return np.array(predictions)

class LogisticRegression:

    def __init__(self, max_iteration: Optional[int]= 1000,
                 fit_intercept: bool= False,
                 learn_rate: Optional[float]= 0.05) -> None:
        
        self.fit_intercept: bool= fit_intercept
        self.parameters: Optional[MatrixLike | ArrayLike]= None
        self.learn_rate: float= learn_rate
        self.max_iteration: int= max_iteration

    def fit(self, x_train: MatrixLike | ArrayLike,
            y_train: MatrixLike | ArrayLike) -> None:
        
        validate_inputs(x_train, y_train)
        x_train,y_train=change_type(x_train,y_train)
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=fit_intercept(x_train)

        self.parameters=init_parameters(x_train) # params = vector of zeros
        iteration: int= 0
        #train
        while(True):
            hypothesis: MatrixLike= 1/(1+np.exp(-np.dot(x_train,self.parameters)))
            gradient: MatrixLike= np.dot(x_train.T,(y_train-hypothesis))
            #gradient ascent
            self.parameters: MatrixLike= self.parameters + self.learn_rate * gradient
            #checking for convergence to end the loop
            if np.linalg.norm(gradient) < 1e-7:
                break

            iteration+= 1
            if iteration > self.max_iteration:
                print('Reached max iteration {} without convergence'.format(self.max_iteration))
                break

    def predict(self, x_test: MatrixLike | ArrayLike) -> np.ndarray:

        if self.parameters is None:
            raise ValueError('Fit method was not called, train-data was not provided')
        x_test= np.array(x_test)
        if self.fit_intercept:
            x_test= fit_intercept(x_test)
        return np.array(np.dot(x_test, self.parameters)) #(mx1) prediction vector

    @property
    def coef_(self) -> np.ndarray:
        return self.parameters[1:]
    
    @property
    def intercept_(self) -> float:
        return self.parameters[0]
    
class PoissonRegression:
    
    """
    PoissonRegression using gradient ascent to maximize the log-likelihood function

    Parameters:
        fit_intercept (bool) : wether to include intercept
        learning_rate (float) : step-size in the gradient ascent

    Methods:
        fit(x_train,y_train): training the parameters on the given data-set
        predict(x_test): predicting outputs for given inputs
    
    """

    #Class initializer
    def __init__(self,fit_intercept=False,learning_rate=0.005,exit_point=np.power(10.0,-5)):
        self.learning_rate=learning_rate
        self.parameters=None
        self.fit_intercept=fit_intercept #Intercept vector check
        self.exit_point=exit_point

    # Defining the fit method
    def fit(self,x_train,y_train):
        x_train,y_train=change_type(x_train,y_train)
        if (x_train.shape[0]!=y_train.shape[0]):
            raise ValueError('x_train shape does not match y_train'
                             f'{x_train.shape[0]} and {y_train.shape[0]}')
        if self.fit_intercept:
            x_train=fit_intercept(x_train)
        self.parameters=init_parameters(x_train) # initializing parameters to vector of zeros

        # defining the likelihood ascent :
        self.exit_point=np.power(10.0,-5) #defining exit_point for the loop
        gradient=np.inf

        while(gradient>self.exit_point):
            hypothesis=np.exp(np.dot(x_train,self.parameters)) # our poisson hypothesis
            gradient=np.dot(x_train.T,y_train-hypothesis)
            self.parameters+=self.learning_rate*gradient #gradient ascent

    # Defining the predict method
    def predict(self,x_test):
        if self.parameters is None:
            raise ValueError('Fit method was not called '
                             'Training data was not provided !')
        
        x_test=change_type(x_test)
        if self.fit_intercept:
            x_test=fit_intercept(x_test)
        return np.dot(x_test,self.parameters)
