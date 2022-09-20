from activation_functions import *
from loss_functions import *


class NeuralNetwork:

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

    def train(self, x, target):
        self.forward_prop(x.T)
        self.back_prop(target, local_sum_of_squares)

    def forward_prop(self, z):
        for layer in self.layers:
            z = layer.forward_prop_step(z)

    def back_prop(self, target, local_loss):
        # Back-propagation
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        # E di quello precedente, per il calcolo dE/dW
        for i, layer in enumerate(self.layers[:0:-1]):
            next_layer = self.layers[-i]  # quello sulla destra
            prev_layer = self.layers[-i - 2]  # quello sulla sinistra
            print("Layer[%d] type:" % i, layer.type)
            layer.back_prop_step(next_layer, prev_layer, target, local_loss)
            print("----")


class Layer:

    def __init__(self, neurons, type=None, activation=None):
        self.dE_db = None  # matrice di derivate dE/db dove b è il vettore colonna bias
        self.dE_dW = None  # matrice di derivate dE/dW dove W è la matrice del layer
        self.dact_a = None  # derivata della funzione di attivazione nel punto a
        self.out = None  # matrice di output per il layer
        self.weight = None  # matrice di pesi in ingresso
        self.bias = None  # bias nel layer
        self.w_sum = None  # matrice delle somme pesate
        self.neurons = neurons  # numero di neuroni nel layer
        self.type = type  # se il layer è input, hidden o output
        self.activation = activation  # funzione di attivazione associata al layer
        self.deltas = None  # vettore colonna di delta

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

    def forward_prop_step(self, z):
        # Se il layer è di input, si fa un un mirroring del vettore in ingresso
        if self.type == "input":
            self.out = z
        else:
            self.w_sum = np.dot(self.weight, z) + self.bias
            self.out = self.activation(self.w_sum)
        return self.out

    def back_prop_step(self, next_layer, prev_layer, t, local_loss):
        if self.type == "input":
            pass
        elif self.type == "output":
            # BP1
            self.dact_a = self.activation(self.w_sum, derivative=True)  # la g'(a) nella formula, per ogni k nel layer
            self.deltas = np.multiply(self.dact_a, local_loss(c, t, self.out, derivative=True))  # cx1
        else:
            # BP2
            self.dact_a = self.activation(self.w_sum, derivative=True)  # mx1
            self.deltas = np.multiply(self.dact_a, np.dot(next_layer.weight.T, next_layer.deltas))

        # Una volta calcolati i delta dei nodi di output e quelli interni, occorre calcolare
        # La derivata della funzione E rispetto al generico peso wij [formula 1.5] sull'istanza n-esima
        # Quindi costruisco una matrice di derivate una per ogni matrice di pesi (quindi una per ogni livello)
        # Sarà sufficiente moltiplicare (righe per colonne) self.deltas con gli output z del layer a sinistra
        self.dE_dW = np.dot(self.deltas, prev_layer.out.T)

        # Per ogni layer: dE/db = dE/da * da/db = dE/da * 1 = dE/da = delta
        # Quindi la derivata di E risp. al vettore colonna bias è self.deltas
        self.dE_db = self.deltas

        # dbg
        print("deltas shape:", np.shape(self.deltas))
        print(self.deltas)
        print("prev_layer.out.T shape:", np.shape(prev_layer.out.T))
        print(prev_layer.out.T)
        print("dE/dW shape:", np.shape(self.dE_dW))
        print(self.dE_dW)
        print("dE/db shape:", np.shape(self.dE_db))
        print(self.dE_db)
        print("")


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
