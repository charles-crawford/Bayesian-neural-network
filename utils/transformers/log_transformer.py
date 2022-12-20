from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        :param X: data to transform
        :param y: no y needed
        :return:
        """
        data = X.copy()
        data = np.log(data)
        return data
