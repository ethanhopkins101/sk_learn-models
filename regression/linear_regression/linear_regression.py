import pandas as pd
import numpy as np

class LinearRegression():
    def __init__():
        self.theta=0

    def fit(x_train,y_train,step=0.05):
        x_train=np.array(x_train)
        intercept=np.ones((x_train.shape[1],1))
        x_train=np.concatenate((intercept,x_train),axis=1)
        y_train=np.array(y_train)
        self.theta=np.zeros((x_train.shape[1],1))
        hypothesis=np.dot(x_train,self.theta)
        dj=np.dot((hypothesis-y_train).T,x_train).T
        while(True):
            j_prev=(hypothesis-y_train)**2
            self.theta=selfa.theta+step*dj
            if ((hypothesis-y_train)**2) > j_prev :
                break
    def predict():
        pass
    def coef_():
        pass
    def intercept_():
