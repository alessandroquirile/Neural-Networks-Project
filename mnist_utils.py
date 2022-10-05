"""import numpy as np
from mnist import MNIST

from main import NeuralNetwork

mndata = MNIST('data')
images, labels = mndata.load_training()  # 60.000 immagini

    # index = 4
    # print(mndata.display(images[index]), labels[index])

    images = np.asmatrix(images)  # sono 60.000 immagini, ciascuna con 28*28 colonne
    labels = np.asarray(labels)

    net = NeuralNetwork()
    d = 28*28  # dimensione dell'input
    c = 10  # dimensione dell'output

    for m in (d, 4, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.create()

    net.train(images, labels)"""