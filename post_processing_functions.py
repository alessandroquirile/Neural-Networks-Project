import numpy as np


def softmax(y):
    e_y = np.exp(y - np.max(y))
    return e_y / e_y.sum()
    # return np.exp(y) / np.sum(np.exp(y))  # may cause over-under flow
