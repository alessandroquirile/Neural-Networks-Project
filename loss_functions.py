# Dare la possibilità di usare almeno la sum-of-squares oppure la cross-entropy con e senza soft-max
import numpy as np


# sum-of-squares locale all'istanza n-esima
def local_sum_of_squares(c, t, y, derivative=False):
    ret = 0
    for k in range(c):
        ret += np.square(y - t)
    ret = 1 / 2 * ret
    if derivative:
        return y - t
    return ret


def sum_of_squares(N, c, t, y):
    ret = 0
    for n in range(N):
        ret += local_sum_of_squares(c, t, y)
    return ret


# Da aggiungere la versione con soft-max e la derivata
def cross_entropy(c, t, y):
    ret = 0
    for k in range(c):
        ret += t * np.log(y)
    return -ret
