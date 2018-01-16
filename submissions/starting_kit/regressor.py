# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.reg = RandomForestRegressor(n_estimators=500, n_jobs=-1)
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)