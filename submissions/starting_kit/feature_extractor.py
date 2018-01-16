# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
def fill_mean(feat):
    filled = feat.fillna(feat.mean())
    return filled

class FeatureExtractor(object):
    def __init__(self):
        # create strings to replace nan and values that are in testing but not in training set
        # the strings are made such that they should not interfere with actual values
        self.na_string = "__na"
        self.other_string = "__other"
    
    def rename_other(self, X_df):
        # rename strings that do not appear in training and that do in testing
        return X_df.map(lambda s: self.other_string if s not in self.encoder.classes_ else s)
    
    def fit(self, X_df, y_array):
        # fit the LabelBinarizer on the dataset + the "other" string
        X = pd.concat([X_df['admission_type'], pd.Series(list(self.other_string))])
        self.encoder = LabelBinarizer()
        self.encoder.fit(X.fillna(self.na_string))        

        return self

    def transform(self, X_df):
        # preprocess features to be used
        features = ['heartrate_mean', 'sysbp_mean', 'diasbp_mean', 'resprate_mean', 'tempc_mean', 'admission_type']
        X = X_df[features].reset_index()
        
        # fill unknown values with mean
        heart = fill_mean(X.heartrate_mean)
        sbp = fill_mean(X.sysbp_mean)
        dbp = fill_mean(X.diasbp_mean)
        resp = fill_mean(X.resprate_mean)
        temp = fill_mean(X.tempc_mean)
        
        # binarize a string feature
        admit = pd.DataFrame(self.encoder.transform(
            self.rename_other(X['admission_type'].fillna(self.na_string))))
        
        X = pd.concat([heart, sbp, dbp, resp, temp, admit], axis=1)
        
        return X
    
    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)