from activation_functions import *


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
                layer.init_parameters(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)
            print("neurons:", layer.neurons)
            print("type:", layer.type)
            print("act:", layer.activation)
            print("weight:", np.shape(layer.weight))
            print(layer.weight)
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")

    def train(self, x, y):
        z = x.T
        for layer in self.layers:
            z = layer.forward_prop(z)

        # va dall'output all'input, escludendolo
        for layer in reversed(self.layers):
            if layer.type != "input":
                layer.back_prop()
                print("")

        """# Modo equivalent per andare dall'output all'input escludendo quest'ultimo
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        for layer, succ_layer in zip(self.layers[-2::-1], self.layers[::-1]):
            if layer.type != "input":
                layer.back_prop(succ_layer)
                print("")"""


class Layer:
    """Un layer è composto da uno o più neuroni, una funzione di attivazione unica, matrice di pesi in ingresso
    e bias"""

    def __init__(self, neurons, type=None, activation=None):
        self.out = None
        self.weight = None
        self.bias = None
        self.w_sum = None
        self.neurons = neurons
        self.type = type
        # self.is_input = is_input
        self.activation = activation
        self.deltas = []  # lista dei delta di ciascun neurone in questo layer

    # Inizializza la matrice dei pesi ed il vettore dei bias
    def init_parameters(self, prev_layer_neurons):
        self.weight = np.asmatrix(np.ones((self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.ones(self.neurons)).T
        """self.weight = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T  # vettore colonna"""
        if self.activation is None:
            self.activation = sigmoid

    def forward_prop(self, x):
        # Se il layer è di input, si fa un un mirroring del vettore in ingresso
        if self.type == "input":
            self.out = x
        else:
            self.w_sum = np.dot(self.weight, x) + self.bias
            self.out = self.activation(self.w_sum)
        # show(self.out)  # dbg
        return self.out

    def back_prop(self):
        if self.type == "output":
            for k in range(self.neurons):
                # Calcolo il delta di ciascun nodo d'uscita k - BP1
                # print("self.wsum[%d]:\n" % k, self.w_sum[k])  # la a_k nella formula
                # print("self.out[%d]:\n" % k, self.out[k])  # la z_k = y_k nella formula
                dg_a = self.activation(self.w_sum[k], derivative=True)  # la g'(a_k) nella formula
                # print("dg_a[%d]:\n" % k, df_a)
                delta = dg_a * (self.out[k] - 1)  # delta_k nella formula
                self.deltas.append(delta)  # salvo ciscun delta_k in una lista di quel layer
                # print("delta[%d]:\n" % k, delta)
                # print("")
            print(self.deltas)
        else:
            # Calcolo del delta di ciascun nodo interno h - BP2
            for h in range(self.neurons):
                # print("self.wsum[%d]:\n" % h, self.w_sum[h])  # la a_h nella formula
                df_a = self.activation(self.w_sum[h], derivative=True)  # la f'(a_k) nella formula
                # print("df_a[%d]:\n" % h, df_a)
                # Adesso devo iterare su tutti i neuroni dello strato successivo rispetto a quello di h...
                # ...per calcolarmi la sommatoria


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
    y = np.asarray([0, 0, 1, 0])

    net.train(X[2], y[2])  # [1,0], 1
