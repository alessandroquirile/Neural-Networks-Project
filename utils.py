import math

import numpy as np
from matplotlib import pyplot as plt

from post_processing_functions import softmax


# Stampa per intero un array numpy
def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def accuracy_score(targets, predictions):
    predictions = softmax(predictions)
    correct_predictions = 0
    for item in range(np.shape(predictions)[1]):
        # print(predictions[:, item])
        argmax_idx = np.argmax(predictions[:, item])
        # print("argmax idx", argmax_idx)
        # print(targets[:, item])
        if targets[argmax_idx, item] == 1:
            correct_predictions += 1
    return correct_predictions / np.shape(predictions)[1]


def one_hot(targets):
    return np.asmatrix(np.eye(10)[targets]).T  # vettore colonna


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


def balanced(targets_train, targets_val, targets_test):
    return is_balanced(targets_train) and is_balanced(targets_val) and is_balanced(targets_test)


# https://www.researchgate.net/figure/Class-percentages-in-MNIST-dataset_fig2_320761896
def is_balanced(targets):
    zeros = ones = twos = threes = fours = fives = sixes = sevens = eights = nines = 0
    total_targets = np.shape(targets)[0]
    for target in np.nditer(targets):
        if target == 0:
            zeros += 1
        elif target == 1:
            ones += 1
        elif target == 2:
            twos += 1
        elif target == 3:
            threes += 1
        elif target == 4:
            fours += 1
        elif target == 5:
            fives += 1
        elif target == 6:
            sixes += 1
        elif target == 7:
            sevens += 1
        elif target == 8:
            eights += 1
        elif target == 9:
            nines += 1
    """print("Class percentage:")
    print(f"Zeros {zeros / total_targets * 100}, ones {ones / total_targets * 100}, "
          f"twos {twos / total_targets * 100}, threes {threes / total_targets * 100}, "
          f"fours {fours / total_targets * 100}, fives {fives / total_targets * 100}, "
          f"sixes {sixes / total_targets * 100}, sevens {sevens / total_targets * 100}, "
          f"eights {eights / total_targets * 100}, nines {nines / total_targets * 100}")"""

    abs_tol = 2.0
    if math.isclose(zeros / total_targets * 100, ones / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(ones / total_targets * 100, twos / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(twos / total_targets * 100, threes / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(threes / total_targets * 100, fours / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(fours / total_targets * 100, fives / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(fives / total_targets * 100, sixes / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(sevens / total_targets * 100, eights / total_targets * 100, abs_tol=abs_tol) and \
            math.isclose(eights / total_targets * 100, nines / total_targets * 100, abs_tol=abs_tol):
        return True
    return False
