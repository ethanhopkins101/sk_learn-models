import numpy as np
#function that changes the learning algorithm inputs to array-like
def __change_type(x_train,y_train):
    return (np.array(x_train),np.array(y_train))#returning a tuple
#designing intercept_fitting function:
def __fit_intercept(x_train):
    intercept=np.ones((x_train.shape[0],1)) #column vector of ones
    return np.hstack((intercept,x_train)) #returning the appended vector with x_train
#arbitrary parameter initializer function:
def __init_parameters(x_train):
    return np.zeros((x_train.shape[1],1))#return a column vector of shape x_train[1]
class LinearRegression:
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.theta=None

#Locally_Weighted_Regression
class LocallyWeightedRegression:
    #class initializer
    def __init__(self,t,fit_intercept=False):
        self.t=t
        self.fit_intercept=fit_intercept
        self.parameters=None
    #defining the fit method
    def fit(self,x_train,y_train):
        #making sure the inputs are of array type
        x_train,y_train=__change_type(x_train,y_train)
        #initializing the parameters
        self.parameters
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=__fit_intercept(x_train)
        
    #defining the predict method
    def predict(self,x_test):
        pass


class LogisticRegression:
    #class initializer
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.parameters=None
        self.learn_rate=0.05
    # defining the fit method
    def fit(self,x_train,y_train):
        #making sure inputs are in array type
        x_train,y_train=__change_type(x_train,y_train)
        #initializing parameters
        self.parameters=np.zeros((x_train.shape[1],1))
        #adding intercept term if specified by user
        if self.fit_intercept:
            x_train=__fit_intercept(x_train)
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
    def predict():
        pass

    @property
    def coef_(self):
        return self.parameters[1:]
    @property
    def intercept_(self):
        return self.parameters[0]