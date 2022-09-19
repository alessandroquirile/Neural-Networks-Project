from activation_functions import *
from loss_functions import *


# dbg purpose
def show(v):
    print("Shape is", np.shape(v))
    print(v)
    print("")


class NeuralNetwork:
    """Una rete neurale è composta da uno o più layer"""

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def create(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.type = "input"
            else:
                if i == len(self.layers) - 1:
                    layer.type = "output"
                else:
                    layer.type = "hidden"
                layer.configure(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)  # layer id
            print("neurons:", layer.neurons)
            print("type:", layer.type)
            print("act:", layer.activation)
            print("weight:", np.shape(layer.weight))
            print(layer.weight)  # input weight matrix
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")

    def train(self, x, t):
        z = x.T
        for layer in self.layers:
            z = layer.forward_prop(z)

        # Back-propagation
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        for i, layer in enumerate(self.layers[::-1]):
            next_layer = self.layers[-i]
            print("Layer[%d] type:" % i, layer.type)
            layer.back_prop(next_layer, t, local_sum_of_squares)
            print("----")


class Layer:
    """Un layer è composto da uno o più neuroni, una funzione di attivazione unica, matrice di pesi in ingresso
    e bias"""

    def __init__(self, neurons, type=None, activation=None):
        self.dact_a = None
        self.out = None
        self.weight = None
        self.bias = None
        self.w_sum = None
        self.neurons = neurons
        self.type = type
        self.activation = activation
        self.deltas = None

    # Configura parametri e iperparametri del layer:
    # Matrice dei pesi in ingresso, bias e funzione di attivazione
    def configure(self, prev_layer_neurons):
        self.weight = np.asmatrix(np.ones((self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.ones(self.neurons)).T
        """self.weight = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T  # vettore colonna"""
        if self.activation is None:
            # th approx universale
            if self.type == "hidden":
                self.activation = sigmoid
            elif self.type == "output":
                self.activation = identity

    def forward_prop(self, x):
        # Se il layer è di input, si fa un un mirroring del vettore in ingresso
        if self.type == "input":
            self.out = x
        else:
            self.w_sum = np.dot(self.weight, x) + self.bias
            self.out = self.activation(self.w_sum)
        return self.out

    def back_prop(self, next_layer, t, local_loss):
        if self.type == "input":
            pass
        elif self.type == "output":
            # BP1
            self.dact_a = self.activation(self.w_sum, derivative=True)  # la g'(a) nella formula, per ogni k nel layer
            self.deltas = np.multiply(self.dact_a, local_loss(c, t, self.out, derivative=True))
        else:
            # BP2
            self.dact_a = self.activation(self.w_sum, derivative=True)
            self.deltas = np.multiply(self.dact_a, np.dot(next_layer.weight.T, next_layer.deltas))

        print("deltas:")
        print(self.deltas)
        print("")

        # Una volta calcolati i delta dei nodi di output e quelli interni, occorre calcolare
        # La derivata della funzione E rispetto al generico peso wij [formula 1.5]
        # Sull'istanza n-esima


if __name__ == '__main__':
    net = NeuralNetwork()
    d = 2  # dimensione dell'input
    c = 1  # dimensione dell'output

    for m in (d, 4, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.create()

    # net.summary()  # dbg

    # Training set
    X = np.asmatrix([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    t = np.asarray([0, 0, 1, 0])

    net.train(X[2], t[2])  # [1,0], 1
