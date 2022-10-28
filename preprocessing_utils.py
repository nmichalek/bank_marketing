from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Mapper(BaseEstimator, TransformerMixin):
    """Map categories to defined mapping"""

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X


class Missing_Adding(BaseEstimator, TransformerMixin):
    """Replacie 999 witn np.Nan"""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].replace({999: np.NaN})

        return X


class NonZero(BaseEstimator, TransformerMixin):
    """Add to feature 0.01 for avoid 0 value
     and enable logarithm calculation"""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature] + 0.01

        return X
