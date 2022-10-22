import numpy as np
from scipy.special import expit


def sigmoid(a, derivative=False):
    # f_a = 1 / (1 + np.exp(-a))  # potrebbe causare overflow
    f_a = expit(a)
    if derivative:
        return np.multiply(f_a, (1 - f_a))
    return f_a


def identity(a, derivative=False):
    f_a = a
    if derivative:
        return np.ones(np.shape(a))
    return f_a


def relu(a, derivative=False):
    f_a = np.maximum(0, a)
    if derivative:
        return (a > 0) * 1
    return f_a
