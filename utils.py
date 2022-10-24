import math
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def one_hot_to_label(targets):
    return np.asarray(np.argmax(targets, axis=0))


def plot_losses(epochs, train_losses, val_losses):
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses, color="orange")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_accuracies(epochs, train_accuracies, val_accuracies):
    plt.plot(epochs, train_accuracies)
    plt.plot(epochs, val_accuracies, color="orange")
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def balanced(targets_train, targets_val, targets_test, tolerance=3):
    rules = [is_balanced(targets_train, tolerance),
             is_balanced(targets_val, tolerance),
             is_balanced(targets_test, tolerance)]
    return all(rules)


# https://www.researchgate.net/figure/Class-percentages-in-MNIST-dataset_fig2_320761896
def is_balanced(targets, tolerance):
    c = Counter(targets)
    percentages = [v / c.total() * 100 for v in c.values()]
    return math.isclose(min(percentages), max(percentages), abs_tol=tolerance)
