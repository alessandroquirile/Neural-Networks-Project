import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def identity(a):
    return a

def relu(a):
    return a if a > 0 else 0