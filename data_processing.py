import numpy as np
import sklearn


def normalize(X, targets, shuffle):
    if shuffle:
        X, targets = sklearn.utils.shuffle(X, targets.T)
    X = X / np.max(X)
    X = X.T
    return X, targets


def split(X, targets, val_size):
    X_val, X_train = np.hsplit(X, [val_size])
    targets_val, targets_train = np.hsplit(targets, [val_size])
    return X_val, targets_val


def one_hot(targets):
    return np.asmatrix(np.eye(targets.max() + 1)[targets]).T
