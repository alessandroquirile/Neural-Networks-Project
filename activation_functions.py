import numpy as np


def sigmoid(a, derivative=False):
    f_a = 1 / (1 + np.exp(-a))
    df_a = np.multiply(f_a, (1 - f_a))  # element-wise
    if derivative:
        return df_a
    return f_a


def identity(a, derivative=False):
    f = a
    df = np.ones(np.shape(a))
    if derivative:
        return df
    return f


def relu(a, derivative=False):
    f = a if a > 0 else 0
    df = 1 if a > 0 else 0 if a < 0 else None
    if derivative:
        return df
    return f
