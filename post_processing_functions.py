import numpy as np


def softmax(y):
    epsilon = 10 ** -308
    e_y = np.exp(y - np.max(y, axis=0))
    sm = e_y / np.sum(e_y, axis=0)
    return np.clip(sm, epsilon, 1 - epsilon)  # more numerically stable
