from post_processing_functions import *


def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


def cross_entropy(y, t, derivative=False, post_process=False):
    if post_process:
        if derivative:
            return y - t
        return -np.sum(np.sum(np.multiply(t, np.log(softmax(y))), axis=0))  # axis=0 somma di elementi per colonne
    else:
        if derivative:
            return -np.sum(np.divide(t, y), axis=1)  # axis=1 somma di elementi per righe
        return -np.sum(np.sum(np.multiply(t, np.log(y)), axis=0))
