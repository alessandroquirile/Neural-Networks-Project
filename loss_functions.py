from post_processing import *


def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


def cross_entropy(y, t, derivative=False, post_process=False):
    if post_process:
        if derivative:
            return y - t
        return -np.sum(np.sum(np.multiply(t, np.log(soft_max(y))), axis=0))
    else:
        if derivative:
            return -np.sum(np.divide(t, y), axis=1)
        return -np.sum(np.sum(np.multiply(t, np.log(y)), axis=0))
