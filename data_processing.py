import numpy as np
import sklearn


def normalize(X, targets, shuffle):
    if shuffle:
        X, targets = sklearn.utils.shuffle(X, targets.T)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X, targets


def split(X_train, targets_train, val_size):
    X_val, X_train = np.hsplit(X_train, [val_size])
    targets_val, targets_train = np.hsplit(targets_train, [val_size])
    return X_val, targets_val, X_train, targets_train


def one_hot(targets):
    return np.asmatrix(np.eye(targets.max() + 1)[targets]).T
