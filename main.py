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
                layer.is_input = True
            else:
                layer.init_parameters(self.layers[i - 1].neurons)

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

    def train(self, x, y):
        z = x.T
        for layer in self.layers:
            z = layer.forward_prop(z)

        # va dall'output all'input, escludendolo
        for layer in reversed(self.layers):
            if not layer.is_input:
                layer.back_prop()
                print("")


class Layer:
    """Un layer è composto da uno o più neuroni, una funzione di attivazione unica, matrice di pesi in ingresso
    e bias"""

    def __init__(self, neurons, is_input=False, activation=None):
        self.out = None
        self.weight = None
        self.bias = None
        self.w_sum = None
        self.neurons = neurons
        self.is_input = is_input
        self.activation = activation

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
        if self.is_input:
            self.out = x
        else:
            self.w_sum = np.dot(self.weight, x) + self.bias
            self.out = self.activation(self.w_sum)
        # show(self.out)  # dbg
        return self.out

    def back_prop(self):
        # Se il layer è di output
        if self.neurons == c:
            for k in range(self.neurons):
                # Calcolo il delta di ciascun nodo d'uscita k - BP1
                print("self.wsum[%d]:\n" % k, self.w_sum[k])
                df_a = self.activation(self.w_sum[k], derivative=True)
                print("self.out[%d]:\n" % k, self.out[k])
                print("df_a[%d]:\n" % k, df_a)
                delta = df_a * (self.out[k] - 1)
                print("delta[%d]:\n" % k, delta)
                print("")
        else:
            # Se non è né input (escluso dal for esterno), né output è sicuramente hidden
            # Calcolo del delta di ciascun nodo interno i - BP2
            pass


if __name__ == '__main__':
    net = NeuralNetwork()
    d = 2  # dimensione dell'input
    c = 2  # dimensione dell'output

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
