# Dare la possibilità di usare almeno la sum-of-squares oppure la cross-entropy con e senza soft-max
import numpy as np


def local_sum_of_squares(c, t, y):
    ret = 0
    for k in range(c):
        ret += (y - t) ** 2
    return 1 / 2 * ret


def sum_of_squares(N, c, t, y):
    """
    Funzione di errore per un problema di regressione

    :param N: Dimensione del dataset
    :param c: Numero di neuroni di uscita
    :param t: Gold label
    :param y: Predizione
    :return: funzione di errore sull'intero dataset
    """
    ret = 0
    for n in range(N):
        ret += local_sum_of_squares(c, t, y)
    return ret


# Da aggiungere la versione con soft-max
def cross_entropy(c, t, y):
    ret = 0
    for k in range(c):
        ret += t * np.log(y)
    return -ret
