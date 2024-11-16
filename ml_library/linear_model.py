import numpy as np
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
    #defining the fit function
    def fit(self,x_train,y_train):
        #making sure the inputs are of array type
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        #adding intercept term if specified by user
        if self.fit_intercept:
            intercept=np.ones((x_train.shape[0],1))
            x_train=np.hstack((intercept,x_train))
    #defining the predict function
    def predict(self,x_test):
        pass


class LogisticRegression:
    #class initializer
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.parameters=None
        self.learn_rate=0.05
    # defining the fit function
    def fit(self,x_train,y_train):
        #making sure inputs are in array type
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        #initializing parameters
        self.parameters=np.zeros((x_train.shape[1],1))
        #adding intercept term if specified by user
        if self.fit_intercept:
            intercept=np.ones((x_train.shape[0],1))
            x_train=np.hstack((intercept,x_train))
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
        
    def predict():
        pass

    @property
    def coef_(self):
        return self.parameters[1:]
    @property
    def intercept_(self):
        return self.parameters[0]