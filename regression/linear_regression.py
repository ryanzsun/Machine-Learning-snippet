import numpy as np
from sklearn.datasets import load_boston

class LinearRegression():
    def __init__(self, learning_rate, convergence_diff):
        self.lr = learning_rate
        self.eps = convergence_diff
        pass

    def fit(self, data, label):
        pass

    def predict(self, data):
        pass

    def stochastic_gradient_descent(self):
        pass

    def loss(self, prediction, target):
        pass