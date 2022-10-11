from post_processing_functions import *
from scipy.special import xlog1py


def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


def cross_entropy(y, t, derivative=False, post_process=True):
    epsilon = 10 ** -308
    if post_process:
        if derivative:
            return y - t
        sm = softmax(y)
        sm = np.clip(sm, epsilon, 1 - epsilon)  # avoid log(0)
        # return -np.sum(np.sum(np.multiply(t, np.log(sm)), axis=0))
        return -np.sum(np.sum(xlog1py(t, softmax(y)), axis=0))  # supposed to be more numerically stable
    else:
        if derivative:
            return -np.sum(np.divide(t, y), axis=1)  # axis=1 somma di elementi per righe
        return -np.sum(np.sum(xlog1py(t, y), axis=0))  # axis=0 somma di elementi per colonne
