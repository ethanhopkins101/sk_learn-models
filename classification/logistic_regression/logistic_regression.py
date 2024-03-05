import numpy as np


class LogisticRegression:
    def __init__(self):
        self.theta = 0

    def fit(self, x_train, y_train, method="logistic_loss"):
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(len(y_train), 1)
        match method:
            case "logistic_loss":
                pass
            case "newton":
                pass

    def predict(x_test):
        x_test = np.array(x_test)
