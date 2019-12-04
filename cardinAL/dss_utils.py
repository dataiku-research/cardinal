from sklearn.base import BaseEstimator
import numpy as np


class DatasetWrapper(BaseEstimator):

    def __init__(self, column_names):
        self._column_names = np.asarray(column_names)
        self._class_names = np.asarray([s[6:] for s in column_names])
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        best = np.argmax(X[self._column_names], axis=1)
        return self._class_names[best]
    
    def predict_proba(self, X):
        return X[self._column_names]