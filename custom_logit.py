import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomLogit(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=100, use_regularization=True):
        self.alpha = alpha
        self.use_regularization = use_regularization
        self.model = None

    def fit(self, X, y):
        try:
            if self.use_regularization:
                self.model = sm.Logit(y, X).fit_regularized(method='l1', maxiter=10000, alpha=self.alpha, trim_mode="auto")
            else:
                self.model = sm.Logit(y, X).fit()
        except Exception as e:
            logging.error(f"An error occurred during fitting: {e}")
            raise
        return self

    def predict(self, X):
        if self.model is None:
            logging.error("Model has not been fitted yet.")
            raise ValueError("Model is not fitted.")
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        if self.model is None:
            logging.error("Model has not been fitted yet.")
            raise ValueError("Model is not fitted.")
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(y, preds)
