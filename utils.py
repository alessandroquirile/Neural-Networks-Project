import math
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


# Stampa per intero un array numpy
def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def accuracy_score(targets, predictions):
    targets = one_hot_to_label(targets)
    predictions = one_hot_to_label(predictions)  # no softmax needed
    return np.mean(predictions == targets)


def one_hot(targets):
    return np.asmatrix(np.eye(10)[targets]).T


def one_hot_to_label(targets):
    return np.asarray(np.argmax(targets, axis=0))


def plot(epochs, loss_train, loss_val):
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_val, color="orange")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def balanced(targets_train, targets_val, targets_test, tolerance=2.5):
    rules = [is_balanced(targets_train, tolerance),
             is_balanced(targets_val, tolerance),
             is_balanced(targets_test, tolerance)]
    return all(rules)


# https://www.researchgate.net/figure/Class-percentages-in-MNIST-dataset_fig2_320761896
def is_balanced(targets, tolerance):
    c = Counter(targets)
    percentages = [v / c.total() * 100 for v in c.values()]
    return math.isclose(min(percentages), max(percentages), abs_tol=tolerance)
