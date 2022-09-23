import numpy as np


def sigmoid(a, derivative=False):
    f_a = 1 / (1 + np.exp(-a))
    df_a = np.multiply(f_a, (1 - f_a))  # element-wise
    if derivative:
        return df_a
    return f_a


def identity(a, derivative=False):
    f_a = a
    df_a = np.ones(np.shape(a))
    if derivative:
        return df_a
    return f_a


def relu(a, derivative=False):
    f_a = np.maximum(0, a)
    df_a = (a > 0) * 1
    if derivative:
        return df_a
    return f_a
