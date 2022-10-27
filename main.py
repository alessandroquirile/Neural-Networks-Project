import numpy as np
from mnist.loader import MNIST

from data_processing import normalize, split, one_hot
from metrics import accuracy_score
from models import NeuralNetwork, Layer
from utils import balanced

if __name__ == '__main__':
    mndata = MNIST(path="data", return_type="numpy")
    X_train, targets_train = mndata.load_training()  # 60.000 images, 28*28 features
    X_test, targets_test = mndata.load_testing()  # 10.000 images, 28*28 features

    X_train, targets_train = normalize(X_train, targets_train, shuffle=True)
    X_test, targets_test = normalize(X_test, targets_test, shuffle=False)  # todo: devo normalizzare?

    X_val, targets_val, X_train, targets_train = split(X_train, targets_train, val_size=10000)

    if not balanced(targets_train, targets_val, targets_test):
        raise Exception("Classes are not balanced")

    targets_train = one_hot(targets_train)
    targets_val = one_hot(targets_val)
    targets_test = one_hot(targets_test)

    net = NeuralNetwork()
    d = np.shape(X_train)[0]  # number of features, 28x28
    c = np.shape(targets_train)[0]  # number of classes, 10

    # Net creation
    for neurons in (d, 100, c):
        net.add(Layer(neurons))

    net.compile()

    best_net = net.fit(X_train=X_train, targets_train=targets_train,
                       X_val=X_val, targets_val=targets_val,
                       max_epochs=50, l_rate=0.000005, momentum=0.9)  # optimal tuning: 5x10**-6

    # Model testing
    predictions_test = best_net.predict(X_test)
    test_acc = accuracy_score(targets_test, predictions_test)
    print(f"Accuracy score on test set is: {test_acc * 100 :.2f} %")
