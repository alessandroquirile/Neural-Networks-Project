# Dare la possibilità di usare almeno la sum-of-squares oppure la cross-entropy con e senza soft-max
import numpy as np


# sum-of-squares locale all'istanza n-esima
def sum_of_squares(y, t, derivative=False):
    if derivative:
        return y - t
    return 0.5 * np.sum(np.sum(np.square((y - t))))


# aggiungere la versione con soft-max
def cross_entropy(y, t, derivative=False):
    if derivative:
        return -np.sum(np.divide(t, y), axis=1)
    return -np.sum(np.sum(np.multiply(t, np.log(y)), axis=0))


# Da aggiungere la versione con soft-max e la derivata
"""def cross_entropy(c, t, y, derivative=False):
    ret = 0
    for k in range(c):
        ret += np.multiply(t, np.log(y))
    if derivative:
        pass
    return -ret"""
