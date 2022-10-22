from post_processing_functions import *


def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


def cross_entropy(y, t, derivative=False, post_process=True):
    if post_process:
        if derivative:
            return y - t
        sm = softmax(y)
        return -np.mean(np.sum(np.multiply(t, np.log(sm)), axis=0))
    else:
        if derivative:
            return -np.sum(np.divide(t, y), axis=1)
        return -np.mean(np.sum(np.multiply(t, y), axis=0))