import numpy as np


def sigmoid(a, derivative=False):
    f = 1 / (1 + np.exp(-a))
    df = f * (1 - f)
    if derivative:
        return df
    return f


def identity(a, derivative=False):
    f = a
    df = 1
    if derivative:
        return df
    return f


def relu(a, derivative=False):
    f = a if a > 0 else 0
    df = 1 if a > 0 else 0 if a < 0 else None
    if derivative:
        return df
    return f

