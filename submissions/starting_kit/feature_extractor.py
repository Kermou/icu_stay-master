# -*- coding: utf-8 -*-
import pandas as pd
import os

def fill_mean(feat):
    filled = feat.fillna(feat.mean())
    return filled

class FeatureExtractor(object):
    def __init__(self):
        pass
    
    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        features = ['heartrate_mean', 'sysbp_mean', 'diasbp_mean', 'resprate_mean', 'tempc_mean']
        X = X_df[features]
        heart = fill_mean(X.heartrate_mean)
        sbp = fill_mean(X.sysbp_mean)
        dbp = fill_mean(X.diasbp_mean)
        resp = fill_mean(X.resprate_mean)
        temp = fill_mean(X.tempc_mean)
        X = pd.concat([heart, sbp, dbp, resp, temp], axis=1)
        return X