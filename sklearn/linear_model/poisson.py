"""
Poisson Regression
"""

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse

from .base import LinearModel, RegressorMixin


def _log_likelihood():
    '''
    describes the log likelihood of the poisson regression

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Logistic loss.
    
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    '''
    n_samples, n_features = X.shape

    grad = np.empty_like(w)
    # sum_terms is a list that is as long as the number of samples points
    sum_terms = np.array(np.dot(X[i], [np.subtract(y[i], np.exp(np.dot(np.transpose(w), X[i]))) for i in range(n_samples)]))

    return np.sum(sum_terms)

def _grad_log_likelihood():
    """
    Gradient of the log likelihood used for finding the MLE
    """

def _hessian_log_likelihood():
    """
    Hessian used in the optimisation of the MLE
    """

class PoissonRegression(LinearModel, RegressorMixin):
    """    
    Poisson Regression:
    TODO: Description here of the actual poisson module

    Parameters:
    -----------
    TODO: optimisers to use, penalty to use, penalty param, max_iter

    Attributes:
    ----------
    
    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.
        `coef_` is of shape (1, n_features) when the given problem
        is binary.
    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.
    n_iter_ : array, shape (n_classes,) or (1, )

    See also:
    ---------

    Notes:
    ---------

    References:
    ----------
    
    """

    def __init__(self, tol=1e-4, alpha=0., fit_intercept=False,
                 solver='sag', max_iter=1000, verbose=0):
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y, exposure=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        """
    
    def predict(self, X):
        """Probability estimates.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")

        assert False