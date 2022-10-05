import numpy as np


def softmax(y):
    e_x = np.exp(y - np.max(y))
    return e_x / e_x.sum()
    # return np.exp(y) / np.sum(np.exp(y))  # may cause over-under flow
