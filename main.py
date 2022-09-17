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
            print("Layer", i)  # layer id
            print("neurons:", layer.neurons)
            print("type:", layer.type)
            print("act:", layer.activation)
            print("weight:", np.shape(layer.weight))
            print(layer.weight)  # input weight matrix
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")

    def train(self, x, y):
        z = x.T
        for layer in self.layers:
            z = layer.forward_prop(z)

        # Back-propagation
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        for i, layer in enumerate(self.layers[::-1]):
            next_layer = self.layers[-i]
            print("Layer type", layer.type)
            layer.back_prop(next_layer)


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

    def back_prop(self, next_layer):
        # Il layer di input è escluso dalla back-prop
        if self.type == "input":
            pass
        elif self.type == "output":
            for k in range(self.neurons):
                # Calcolo il delta di ciascun nodo d'uscita k - BP1
                # print("self.wsum[%d]:\n" % k, self.w_sum[k])  # la a_k nella formula
                # print("self.out[%d]:\n" % k, self.out[k])  # la z_k = y_k nella formula
                dg_a = self.activation(self.w_sum[k], derivative=True)  # la g'(a_k) nella formula
                delta_k = dg_a * (self.out[k] - 1)  # delta_k nella formula TODO: generalizzare alle altre loss
                self.deltas.append(delta_k)  # salvo ciscun delta_k in una lista di quel layer
        else:
            # Calcolo del delta di ciascun nodo interno h - BP2
            for h in range(self.neurons):
                # print("h:", h)
                df_a = self.activation(self.w_sum[h], derivative=True)  # la f'(a_h) nella formula
                sommatoria = 0
                for k in range(next_layer.neurons):
                    for w_kh in next_layer.weight[:, h]:
                        sommatoria += w_kh * next_layer.deltas[k]
                delta_h = df_a * sommatoria
                self.deltas.append(delta_h)

        # Stampo i deltas di ciascun layer
        print("deltas:")
        print(self.deltas)
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
    y = np.asarray([0, 0, 1, 0])

    net.train(X[2], y[2])  # [1,0], 1
