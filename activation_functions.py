import numpy as np


def sigmoid(a, derivate=False):
    f = 1 / (1 + np.exp(-a))
    df = f * (1 - f)
    if derivate:
        return df
    return f


def identity(a, derivate=False):
    f = a
    df = 1
    if derivate:
        return df
    return f


def relu(a, derivate=False):
    f = a if a > 0 else 0
    df = 1 if a > 0 else 0 if a < 0 else None
    if derivate:
        return df
    return f

