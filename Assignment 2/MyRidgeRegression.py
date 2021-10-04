import numpy as np


# reference: https://www.geeksforgeeks.org/implementation-of-ridge-regression-from-scratch-using-python/
class MyRidgeRegression:
    learning_rate = 0.01
    iterations = 1000

    def __init__(self, l2_lambda):
        self.l2_lambda = l2_lambda
        self.X = None
        self.y = None
        self.n_samples = None
        self.feature_weights = None
        self.intercept_weight = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]                     # number of samples
        self.feature_weights = np.zeros(X.shape[1])     # initialize feature weights (Beta_1 to Beta_n)
        self.intercept_weight = 0                       # initialize intercept weight (Beta_0)

        # update weights in gradient descent learning, this is where RSS + penalizing factor is minimized
        for i in range(self.iterations):
            self._update_weights()

    def _update_weights(self):
        y_pred = self.predict(self.X)

        # calculate derivates used to descend gradient
        dFeature_weights = (-(2 * self.X.T.dot(self.y - y_pred)) + (2 * self.l2_lambda * self.feature_weights)) / self.n_samples
        dIntercept_weight = -(2 * np.sum(self.y - y_pred) / self.m)

        # update weights
        self.feature_weights = self.feature_weights - self.learning_rate * dFeature_weights
        self.intercept_weight = self.intercept_weight - self.learning_rate * dIntercept_weight

    def predict(self, X):
        # prediction corresponding with formula Beta_0 + Beta_1 * X1 + Beta_2 * X2 ... + Beta_n * XN
        return self.intercept_weight + X.dot(self.feature_weights)
