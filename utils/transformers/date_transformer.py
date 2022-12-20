from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class DateTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        :param X: data to transform
        :param y: no y needed
        :return:
        """
        year = 2011
        data = X.copy()
        data = year - data
        data = data.astype(int)
        return data
