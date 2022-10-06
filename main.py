import numpy as np
from matplotlib import pyplot as plt

from activation_functions import *
from loss_functions import *

from mnist import MNIST

def generate_data(n_items, n_features, n_classes):
    X = np.asmatrix(np.random.normal(size=(n_items, n_features)))
    targets = np.asarray(np.random.randint(n_classes, size=n_items))
    targets = one_hot(targets)
    return X, targets


# La 1-hot encoding per un'etichetta y restituisce un vettore colonna 1-hot
def one_hot(targets):
    return np.asmatrix(np.eye(np.max(targets) + 1)[targets]).T  # vettore colonna


def plot(epochs, epoch_loss):
    plt.plot(epochs, epoch_loss)
    plt.legend(['Training Loss'])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.type = "input"
            else:
                layer.type = "output" if i == len(self.layers) - 1 else "hidden"
                layer.configure(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)  # layer id
            print("neurons:", layer.neurons)
            print("type:", layer.type)
            print("act:", layer.activation)
            print("weights:", np.shape(layer.weights))
            print(layer.weights)  # input weight matrix
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")

    def fit(self, X, targets):
        MAX_EPOCHS = 200
        epoch_loss = []

        # batch mode
        for epoch in range(MAX_EPOCHS):
            predictions = self.predict(X)
            self.back_prop(targets, cross_entropy)
            self.learning_rule(l_rate=0.01, momentum=0.01)  # 0.00001 - tuning here
            loss = cross_entropy(predictions, targets)
            epoch_loss.append(loss)
            print("E(%d) on TrS is:" % epoch, loss)

        plot(np.arange(MAX_EPOCHS), epoch_loss)

        """for epoch in range(MAX_EPOCHS):
            E = 0  # errore sull'intero DS
            for i, x in enumerate(X):
                target = targets[i].T
                prediction = self.forward_prop(x.T)
                E_n = cross_entropy(prediction, target)
                E += E_n
                self.back_prop(target, local_loss=cross_entropy)
            self.learning_rule(l_rate=0.01, momentum=0.01)  # 0.00001 - tuning here
            epoch_loss.append(E)

        plot(np.arange(MAX_EPOCHS), epoch_loss)"""

    # Predice i target di ciascun elemento del Dataset (anche se singleton)
    # Costruisce una matrice di predizioni dove la i-esima colonna corrisponde
    # alla predizione dell'i-esimo item del DS
    def predict(self, dataset):
        z = dataset.T  # ogni item va considerato come un vettore colonna, quindi traspongo l'intero DS
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z
        # predictions = np.asmatrix(np.zeros(shape=(c, np.shape(dataset)[0])))
        # for i, x in enumerate(dataset):
        #     prediction = self.forward_prop(x.T)
        #     print("Prediction on %d item" % i)  # dbg
        #     print(prediction)  # dbg
        #     print("")
        #     predictions[:, i] = prediction
        # return predictions

    """# Richiede un vettore colonna
    def forward_prop(self, z):
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z  # output della rete sull'input z"""

    def back_prop(self, target, loss):
        # Back-propagation
        # Scarta il layer di input che non è coinvolto nella procedura
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        # E di quello precedente, per il calcolo dE/dW
        for i, layer in enumerate(self.layers[:0:-1]):
            next_layer = self.layers[-i]  # quello sulla destra
            prev_layer = self.layers[-i - 2]  # quello sulla sinistra
            # print("Layer[%d] type:" % i, layer.type)
            layer.back_prop_step(next_layer, prev_layer, target, loss)
            # print("")

    def learning_rule(self, l_rate, momentum):
        # Momentum GD
        for layer in [layer for layer in self.layers if layer.type != "input"]:
            layer.update_weights(l_rate, momentum)
            layer.update_bias(l_rate, momentum)


class Layer:

    def __init__(self, neurons, type=None, activation=None):
        # n è l'n-esimo item se online; altrimenti l'intero DS
        self.dE_dW = 0  # matrice di derivate dE/dW dove W è la matrice del layer sull'intero DS
        self.dE_db = 0  # matrice di derivate dE/db dove b è il vettore colonna bias sull'intero DS
        self.dEn_db = None  # matrice di derivate dE^(n)/db dove b è il vettore colonna bias sull'item n-esimo
        self.dEn_dW = None  # matrice di derivate dE^(n)/dW dove W è la matrice del layer sull'item n-esimo
        self.dact_a = None  # derivata della funzione di attivazione nel punto a
        self.out = None  # matrice di output per il layer
        self.weights = None  # matrice di pesi in ingresso
        self.bias = None  # bias nel layer
        self.w_sum = None  # matrice delle somme pesate
        self.neurons = neurons  # numero di neuroni nel layer
        self.type = type  # se il layer è input, hidden o output
        self.activation = activation  # funzione di attivazione associata al layer
        self.deltas = None  # vettore colonna di delta

    def configure(self, prev_layer_neurons):
        # Solo per dbg
        """self.weights = np.asmatrix(np.ones((self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.ones(self.neurons)).T"""

        self.weights = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T  # vettore colonna

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
            self.w_sum = np.dot(self.weights, z) + self.bias
            self.out = self.activation(self.w_sum)
        return self.out

    def back_prop_step(self, next_layer, prev_layer, target, local_loss):
        if self.type == "output":
            # BP1
            self.dact_a = self.activation(self.w_sum, derivative=True)  # la g'(a) nella formula, per ogni k nel layer
            self.deltas = np.multiply(self.dact_a,
                                      local_loss(self.out, target, derivative=True))  # (c,batch_size)
        else:
            # BP2
            self.dact_a = self.activation(self.w_sum, derivative=True)  # (m,batch_size)
            self.deltas = np.multiply(self.dact_a, np.dot(next_layer.weights.T, next_layer.deltas))

        # Una volta calcolati i delta dei nodi di output e quelli interni, occorre calcolare
        # La derivata della funzione E rispetto al generico peso wij [formula 1.5] sull'istanza n-esima
        # Quindi costruisco una matrice di derivate una per ogni matrice di pesi (quindi una per ogni livello)
        # Sarà sufficiente moltiplicare (righe per colonne) self.deltas con gli output z del layer a sinistra
        self.dEn_dW = self.deltas * prev_layer.out.T

        # Per ogni layer: dE/db = dE/da * da/db = dE/da * 1 = dE/da = delta
        # Quindi la derivata di E risp. al vettore colonna bias è self.deltas
        self.dEn_db = self.deltas

        # La derivata dE/dW sull'intero DS è la somma per n=1 a N di dE/dW sul singolo item
        # self.dE_dW += self.dEn_dW  # todo - probabilmente da togliere
        self.dE_dW = self.dEn_dW  # todo - forse lasciare questo

        # La derivata dE/db sull'intero DS è la somma per n=1 a N di dE/db sul singolo item
        # self.dE_db += self.dEn_db  # todo - probabilmente da togliere
        self.dE_db = self.dEn_db  # todo - forse lasciare questo

        """print(self.dE_dW) # dbg
        print("")"""

        # dbg
        """print("deltas shape:", np.shape(self.deltas))
        # print(self.deltas)
        print("prev_layer.out.T shape:", np.shape(prev_layer.out.T))
        # print(prev_layer.out.T)

        print("dE/dW shape:", np.shape(self.dE_dW))  # dE/dW sull'n-esima istanza (se online), n-esima epoca (se batch)
        # print(self.dE_dW)
        print("dE/db shape:", np.shape(self.dE_db))  # dE/db sull'n-esima istanza (se online), n-esima epoca (se batch)
        # print(self.dE_db)
        print("")"""

    def update_weights(self, l_rate, momentum):
        # Momentum GD
        self.weights = self.weights - l_rate * self.dE_dW
        self.weights = -l_rate * self.dE_dW + momentum * self.weights

    def update_bias(self, l_rate, momentum):
        # Momentum GD
        self.bias = self.bias - l_rate * self.dE_db
        self.bias = -l_rate * self.dE_db + momentum * self.bias


if __name__ == '__main__':

    # Prima di fare MNIST, faccio un caso più semplice
    # Supponiamo una classificazione a 3 classi: cane, gatto, topo
    # La classe cane è 0 -> 000
    # La classe gatto è 1 -> 010
    # La classe topo è 2 -> 001
    net = NeuralNetwork()
    d = 4  # dimensione dell'input (n_features)
    c = 3  # classi in output
    n_items = 10000

    for m in (d, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.build()

    X, targets = generate_data(n_items=n_items, n_features=d, n_classes=c)

    net.fit(X, targets)
