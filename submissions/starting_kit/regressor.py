# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.reg = LinearRegression()
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)