import numpy as np
class LinearRegression:
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.theta=None


class LogisticRegression:
    def __init__(self,fit_intercept=False):
        self.fit_intercept=fit_intercept
        self.parameters=None
        self.learn_rate=0.05
    def fit(self,x_train,y_train):
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        self.parameters=np.zeros((x_train.shape[1],1))
        if self.fit_intercept:
            intercept=np.ones((x_train.shape[0],1))
            x_train=np.hstack((intercept,x_train))
            likelihood_old='inf'
        while(True):
            hypothesis=1/(1+np.exp(-np.dot(x_train,self.parameters)))
            likelihood=np.dot((y_train-hypothesis).T,x_train).T
            self.parameters=self.parameters+self.learn_rate*likelihood
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