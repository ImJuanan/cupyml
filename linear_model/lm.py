# -*- coding: utf-8 -*-
import cupy as cp


class LinearRegression:
    def __init__(self, fit_bias=True):
        self.weight = None
        self.fit_bias = fit_bias
        
    def train(self, X, y):
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        pseudo_inverse = cp.linalg.inv(X.T @ X) @ X.T
        self.weight = cp.matmul(pseudo_inverse, y)
        
    def predict(self, X):
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return cp.matmul(X, self.weight)


class RidgeRegression:
    def __init__(self, lambda_=1, fit_bias=False):
        self.weight = None
        self.lambda_ = lambda_
        self.fit_bias = fit_bias
        
    def train(self, X, y):
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        A = self.lambda_ * cp.eye(X.shape[1])
        pseudo_inverse = cp.linalg.inv(X.T @ X + A) @ X.T
        self.weight = pseudo_inverse @ y
        
    def predict(self, X):
        if self.fit_bias:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return cp.matmul(X, self.weight)