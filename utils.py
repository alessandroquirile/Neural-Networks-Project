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


def generate_data(n_items, n_features, n_classes):
    X = np.asmatrix(np.random.normal(size=(n_items, n_features)))
    targets = np.asarray(np.random.randint(n_classes, size=n_items))
    targets = one_hot(targets)
    return X, targets


# La 1-hot encoding per un'etichetta y restituisce un vettore colonna 1-hot
def one_hot(targets):
    return np.asmatrix(np.eye(10)[targets]).T  # vettore colonna

# Converte la codifica one-hot in quella numerica
def one_hot_to_label(targets):
    return np.argmax(targets, axis=0)


def plot(epochs, loss_train, loss_val):
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_val, color="orange")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
