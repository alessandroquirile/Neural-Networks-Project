import numpy as np


def softmax(y):
    e_y = np.exp(y - np.max(y, axis=0))
    return e_y / np.sum(e_y, axis=0)
    # return np.exp(y) / np.sum(np.exp(y))  # may cause over-under flow
