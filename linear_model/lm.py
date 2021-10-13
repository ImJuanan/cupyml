# -*- coding: utf-8 -*-
"""
    cupyml
    ~~~~~~~~~~~~~~
    Cupy for Machine Learning: Overwriting classical machine learning algorithms using CuPy.

    :author: Juan An <yangxk196@163.com>
    :copyright: (c) 2021 by Juan An.
    :license: MIT, see LICENSE for more details.
"""
import cupy as cp


class LinearRegression:
    def __init__(self, fit_bias=True):
        """An ordinary least squares(OLS) regression model.

        :param fit_bias: Whether to fit an additional bias term. Default is True.
        """
        self.weight = None
        self.fit_bias = fit_bias

    def train(self, X, y):
        """Fit the model weights.

        :param X: :py:class:`cupy.ndarray` of shape `(N, K)` where `N` means sample size and `K` means number of features.
        :param y: :py:class:`cupy.ndarray` of shape `(N, 1)`, which means the targets for each of the `N` examples in `X`.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        pseudo_inverse = cp.linalg.inv(X.T @ X) @ X.T
        self.weight = cp.matmul(pseudo_inverse, y)

    def predict(self, X):
        """Generate predictions on a new dataset.

        :param X: :py:class:`cupy.ndarray` of shape `(Z, K)` where `Z` means the sample size of new dataset.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return cp.matmul(X, self.weight)


class RidgeRegression:
    def __init__(self, lambda_=1, fit_bias=True):
        """A ridge regression model.

        :param lambda_: L2 regularization coefficient. Default is 1.
        :param fit_bias: Whether to fit an additional bias term. Default is True.
        """
        self.weight = None
        self.lambda_ = lambda_
        self.fit_bias = fit_bias

    def train(self, X, y):
        """Fit the model weights.

        :param X: :py:class:`cupy.ndarray` of shape `(N, K)` where `N` means sample size and `K` means number of features.
        :param y: :py:class:`cupy.ndarray` of shape `(N, 1)`, which means the targets for each of the `N` examples in `X`.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        A = self.lambda_ * cp.eye(X.shape[1])
        pseudo_inverse = cp.linalg.inv(X.T @ X + A) @ X.T
        self.weight = pseudo_inverse @ y

    def predict(self, X):
        """Generate predictions on a new dataset.

        :param X: :py:class:`cupy.ndarray` of shape `(Z, K)` where `Z` means the sample size of new dataset.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return cp.matmul(X, self.weight)


class LogisticRegression:
    def __init__(self, fit_bias=True):
        """A logistic regression model.

        :param fit_bias: Whether to fit an additional bias term. Default is True.
        """
        self.weight = None
        self.fit_bias = fit_bias

    def train(self, X, y, lr=0.01, threshold=1e-7, max_iter=1e7):
        """Fit the model weights.

        :param X: :py:class:`cupy.ndarray` of shape `(N, K)` where `N` means sample size and `K` means number of features.
        :param y: :py:class:`cupy.ndarray` of shape `(N, 1)`, which means the binary targets for each of the `N` examples in `X`.
        :param lr: The learning rate. Default is 0.01.
        :param threshold: The threshold to stop the running of gradient descent. Default is 1e-7.
        :param max_iter: The maximum number of iterations. Default is 1e7.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]

        current_loss = cp.inf
        self.weight = cp.random.rand(X.shape[1])
        for i in range(int(max_iter)):
            y_pred = 1 / (1 + cp.exp(-cp.dot(X, self.weight)))
            loss = self._negative_log_likelihood(X, y, y_pred)
            if current_loss - loss < threshold:
                return
            current_loss = loss
            self.weight -= lr * -(cp.dot(y - y_pred, X)) / X.shape[0]

    def _negative_log_likelihood(self, X, y, y_pred):
        """Negative log likelihood(NLL) of the targets under the current model.
        """
        nll = -cp.log(y_pred[y == 1]).sum() - cp.log(1 - y_pred[y == 0]).sum()
        return nll / X.shape[0]

    def predict(self, X):
        """Generate prediction probabilities on a new dataset.

        :param X: :py:class:`cupy.ndarray` of shape `(Z, K)` where `Z` means the sample size of new dataset.
        """
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return 1 / (1 + cp.exp(cp.dot(X, self.weight)))
