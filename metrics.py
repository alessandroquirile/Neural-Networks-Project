import numpy as np

from utils import one_hot_to_label


def accuracy_score(targets, predictions):
    targets = one_hot_to_label(targets)
    predictions = one_hot_to_label(predictions)  # no softmax needed
    return np.mean(predictions == targets)
