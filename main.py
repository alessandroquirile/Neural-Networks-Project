import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def identity(x):
    return x


class NeuralNetwork:
    """Una rete neurale è composta da uno o più layer"""

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def create(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.is_input = True
            else:
                layer.init(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)
            print("neurons:", layer.neurons)
            print("is_input:", layer.is_input)
            print("act:", layer.activation)
            print("weight:", np.shape(layer.weight))
            print(layer.weight)
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")


class Layer:
    """Un layer è composto da uno o più neuroni, una funzione di attivazione unica, matrice di pesi in ingresso
    e bias"""

    def __init__(self, neurons, is_input=False, activation=None):
        self.weight = None
        self.bias = None
        self.neurons = neurons
        self.is_input = is_input
        self.activation = activation

    def init(self, prev_layer_neurons):
        self.weight = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T
        if self.activation is None:
            self.activation = sigmoid


if __name__ == '__main__':
    net = NeuralNetwork()  # costruisco la rete
    d = 2  # dimensione dell'input
    c = 3  # classi in output

    for m in (d, 4, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        # print("layer", m, "has got", layer.neurons, "neurons") # dbg
        net.add_layer(layer)

    net.create()

    net.summary()