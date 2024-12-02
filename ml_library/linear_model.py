import numpy as np
from matrix_operations import *
#Linear_Regression
class LinearRegression:
    #Class initializer
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.parameters=None #initializing parameters

    #defining the fit method
    def fit(self,x_train,y_train):
        x_train,y_train=change_type(x_train,y_train) #changing the inputs type to arrays
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=fit_intercept(x_train)
        #Designing the learning algorithm (normal equations: o=(x.T*x).inv*x.T*y)
        self.parameters=np.dot(np.linalg.inv(np.dot(x_train.T,x_train)),np.dot(x_train.T,y_train))
    
    #defining the predict method
    def predict(self,x_test):
        if self.parameters is None :
            raise ValueError('Fit method was not called'
                             'Training data was not provided !')

        x_test=change_type(x_test) #making sure x_test is of array type 
        if self.fit_intercept:
            x_test=fit_intercept(x_test) # adding intercept term to x_test
        return np.dot(x_test,self.parameters) # return m by 1 column vector of predictions
    @property
    def coef_(self):
        return self.parameters[1:] # returns the 'coef' except for the intercept
    @property
    def intercept_(self):
        return self.parameters[0] # returns the 'intercept'


#Locally_Weighted_Regression
class LocallyWeightedRegression:
    #class initializer
    def __init__(self,t,fit_intercept=False):
        self.t=t
        self.fit_intercept=fit_intercept
        self.x_train=None
        self.y_train=None
    #defining the fit method
    def fit(self,x_train,y_train,x_test):
        if (x_train.shape[0]!=y_train.shape[0]):
            raise ValueError('x_train shape does not match y_train'
                             f'{x_train.shape[0]} and {y_train.shape[0]}')
        #making sure the inputs are of array type
        x_train,y_train=change_type(x_train,y_train)

        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=fit_intercept(x_train)
        self.x_train,self.y_train=x_train,y_train
         #defining the predict method

    def predict(self,x_test):
        #Checking if user provided training-sets already
        if self.x_train is None:
            raise ValueError('Fit method was not called'
                'Training data was not provided !')
        x_test=change_type(x_test)
        #Adding intercept term if specified by the user
        if self.fit_intercept :
            x_test=fit_intercept(x_test)

        t_square=np.square(self.t)
        predictions=[] # empty prediction list

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
    #class initializer
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.parameters=None
        self.learn_rate=0.05
    # defining the fit method
    def fit(self,x_train,y_train):
        if (x_train.shape[0]!=y_train.shape[0]):
            raise ValueError('x_train shape does not match y_train'
                             f'{x_train.shape[0]} and {y_train.shape[0]}')
        #making sure inputs are in array type
        x_train,y_train=change_type(x_train,y_train)
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=fit_intercept(x_train)
        #initializing parameters
        self.parameters=init_parameters(x_train)
        likelihood_old='inf'
        #training the model
        while(True):
            hypothesis=1/(1+np.exp(-np.dot(x_train,self.parameters)))
            likelihood=np.dot(x_train.T,(y_train-hypothesis))
            #gradient ascent
            self.parameters=self.parameters+self.learn_rate*likelihood
            #checking for convergence to end the loop
            if likelihood_old>likelihood:
                break
            likelihood_old=likelihood
    #defining predict method
    def predict(self):
        pass

    @property
    def coef_(self):
        return self.parameters[1:]
    @property
    def intercept_(self):
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
