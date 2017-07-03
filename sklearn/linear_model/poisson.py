"""
Poisson Regression
"""

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse

from .base import LinearModel, RegressorMixin
from ..utils.extmath import safe_sparse_dot


def _log_likelihood(w, X, y, **kwargs):
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

    Returns
    -------
    out : float
        Poisson likelihood
    '''

    return safe_sparse_dot(safe_sparse_dot(y, X), w)

def _grad_log_likelihood(w, X, y, **kwargs):
    """
    Computes the jacobian of the Poisson log_likelihood objective function 

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    TODO: alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """

    return -safe_sparse_dot(X.T, y - np.exp(safe_sparse_dot(X, w)))

def _hessian_log_likelihood():
    """
    Computes the gradient and the Hessian, in the case of a poisson loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.

    Returns
    -------
    Hs : a matrix Hessian of the log_likelihood for Poisson Regression
    """

    diagonal_weight_matrix = sparse( np.exp(safe_sparse_dot(X, w)))

    return safe_sparse_dot(safe_sparse_dot(X.T, diagonal_weight_matrix), X)

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
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : returns an instance of self.
        """

        n_samples, n_features = X.shape
        w0 = np.zeros((n_features, 1))

        func = lambda w: _log_likelihood(w, X, y)
        f_prime = lambda w: _grad_log_likelihood(w, X, y)
        hess = lambda w: _hessian_log_likelihood(w, X, y)

        self.w = optimize.minimize(
                    func,
                    w0,
                    f_prime,
                    hess,
                    method='Newton-CG')
                )
        return self
    
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