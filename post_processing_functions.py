import numpy as np


def soft_max(y):
    return np.exp(y) / np.sum(np.exp(y))
