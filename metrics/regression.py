# -*- coding: utf-8 -*-
import cupy as cp

def MAE(y_true, y_pred):
    return cp.average(cp.abs(y_pred - y_true))


def MSE(y_true, y_pred, *, sqrt=False):
    mse = cp.average(cp.abs(y_pred - y_true)**2)
    if not sqrt:
        return mse
    return cp.sqrt(mse)


def MSLE(y_true, y_pred):
    error_msg = 'Negative values are not allowed.'
    condition = cp.any(y_true < 0) or cp.any(y_pred < 0)
    assert not condition, error_msg
    return MSE(cp.log1p(y_true), cp.log1p(y_pred))


def R2(y_true, y_pred):
    numerator = (1 * (y_true - y_pred)**2).sum(axis=0)
    denominator = (1 * (y_true - cp.average(y_true, axis=0))**2).sum(axis=0)
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    scores = cp.ones([y_true.shape[1]])
    scores[valid_score] = 1 - (numerator[valid_score] /
                               denominator[valid_score])
    scores[nonzero_numerator & ~nonzero_denominator] = 0
    return cp.average(scores)