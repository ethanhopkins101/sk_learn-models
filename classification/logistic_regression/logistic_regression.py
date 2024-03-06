import numpy as np


class LogisticRegression:
    def __init__(self, iteration=10, step=0.05, method="logistic_loss"):
        self.theta = 0
        self.iteration = iteration
        self.method = method
        self.step = step

    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(len(y_train), 1)
        self.theta = np.zeros((x_train.shape[1], 1))
        intercept = np.ones((x_train.shape[0], 1))
        match self.method:
            case "logistic_loss":
                for i in range(self.iteration):
                    z = np.dot(x_train, self.theta)
                    h = 1 / (1 + np.exp(-z))
                    dj = np.dot((h - y_train).T, x_train).T
                    loglikelihood_prev = np.sum(
                        np.multiply(y_train, np.log(h))
                        + np.multiply((1 - y_train), np.log(1 - h))
                    )
                    self.theta = self.theta - np.dot(self.step, dj)
                    loglikelihood = np.sum(
                        np.multiply(y_train, np.log(h))
                        + np.multiply((1 - y_train), np.log(1 - h))
                    )
                    if loglikelihood <= loglikelihood_prev:
                        break

            case "newton":
                pass

    def predict(x_test):
        x_test = np.array(x_test)
        return np.dot(x_test, self.theta)
