from post_processing_functions import *
from scipy.special import xlog1py


def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


def cross_entropy(y, t, derivative=False, post_process=False):
    if post_process:
        if derivative:
            return y - t
        return -np.sum(np.sum(xlog1py(t, softmax(y)), axis=0))  # supposed to be more numerically stable
    else:
        if derivative:
            return -np.sum(np.divide(t, y), axis=1)  # axis=1 somma di elementi per righe
        return -np.sum(np.sum(xlog1py(t, y), axis=0))

    # epsilon = 10 ** -5  # for avodiing log(0)
    # if post_process:
    #     if derivative:
    #         return y - t
    #     return -np.sum(np.sum(np.multiply(t, np.log(epsilon + softmax(y))), axis=0))  # axis=0 somma di elementi per colonne
    # else:
    #     if derivative:
    #         return -np.sum(np.divide(t, y), axis=1)  # axis=1 somma di elementi per righe
    #     return -np.sum(np.sum(np.multiply(t, np.log(y)), axis=0))
