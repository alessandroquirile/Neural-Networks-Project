import numpy as np
from matplotlib import pyplot as plt
from mnist.loader import MNIST
from activation_functions import *
from loss_functions import *
from sklearn.utils import shuffle


def accuracy_score(targets, predictions):
    correct_predictions = 0
    for item in range(np.shape(predictions)[1]):
        # print(predictions[:, item])
        argmax_idx = np.argmax(predictions[:, item])
        # print("argmax idx", argmax_idx)
        # print(targets_test[:, item])
        if targets[argmax_idx, item] == 1:
            correct_predictions += 1
    return correct_predictions / np.shape(predictions)[1]


def generate_data(n_items, n_features, n_classes):
    X = np.asmatrix(np.random.normal(size=(n_items, n_features)))
    targets = np.asarray(np.random.randint(n_classes, size=n_items))
    targets = one_hot(targets)
    return X, targets


# La 1-hot encoding per un'etichetta y restituisce un vettore colonna 1-hot
def one_hot(targets):
    return np.asmatrix(np.eye(10)[targets]).T  # vettore colonna


def plot(epochs, loss_train, loss_val):
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_val, color="orange")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def calculate_gain(activation):
    if activation == sigmoid or activation == identity:
        return 1
    elif activation == relu:
        return np.sqrt(2)
    elif activation == tanh:
        return 5 / 3


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

    def fit(self, X_train, targets_train, X_val, targets_val, max_epochs=50):
        e_loss_train = []
        e_loss_val = []

        # Effettuo una predizione per calcolare l'errore minimo che andrà poi eventualmente aggiornato ad ogni epoca
        predictions_val = self.predict(X_val)
        min_loss_val = cross_entropy(predictions_val, targets_val)

        best_net = self  # rete (intesa come parametri) che minimizza l'errore sul VS
        best_epoch = 0  # epoca in cui l'errore sul VS è minimo

        # batch mode
        for epoch in range(max_epochs):
            predictions_train = self.predict(X_train)
            self.back_prop(targets_train, cross_entropy)
            self.learning_rule(l_rate=0.00001, momentum=0.9)  # tuning here
            loss_train = cross_entropy(predictions_train, targets_train)
            e_loss_train.append(loss_train)

            # Validation
            predictions_val = self.predict(X_val)
            loss_val = cross_entropy(predictions_val, targets_val)
            e_loss_val.append(loss_val)

            print("E(%d) on TrS is:" % epoch, loss_train, " on VS is:", loss_val, " Accuracy:",
                  accuracy_score(targets_val, predictions_val) * 100, "%")

            if loss_val < min_loss_val:
                min_loss_val = loss_val
                best_epoch = epoch
                best_net = self

        print("Validation loss is minimum at epoch:", best_epoch)

        plot(np.arange(max_epochs), e_loss_train, e_loss_val)

        return best_net

    # Predice i target di ciascun elemento del Dataset (anche se singleton)
    # Costruisce una matrice di predizioni dove la i-esima colonna corrisponde
    # alla predizione dell'i-esimo item del DS
    def predict(self, dataset):
        z = dataset.T  # ogni item va considerato come un vettore colonna, quindi traspongo l'intero DS
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z

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
        self.dE_dW = None  # matrice di derivate dE/dW dove W è la matrice del layer sull'intero DS
        self.dE_db = None  # matrice di derivate dE/db dove b è il vettore colonna bias sull'intero DS
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
        self.set_activation()

        # Gaussian
        self.weights = np.asmatrix(np.random.normal(-0.1, 0.02, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(-0.1, 0.02, self.neurons)).T  # vettore colonna

        # self.xavier_init(prev_layer_neurons)

    def set_activation(self):
        if self.activation is None:
            # th approx universale
            if self.type == "hidden":
                self.activation = sigmoid
            elif self.type == "output":
                self.activation = identity

    def xavier_init(self, prev_layer_neurons, distribution="normal"):
        gain = calculate_gain(self.activation)
        if distribution == "normal":
            std = gain * np.sqrt(2 / (self.neurons + prev_layer_neurons))
            self.weights = np.asmatrix(np.random.normal(0, std, (self.neurons, prev_layer_neurons)))
            self.bias = np.asmatrix(np.random.normal(0, std, self.neurons)).T  # vettore colonna
        elif distribution == "uniform":
            a = gain * np.sqrt(6 / (self.neurons + prev_layer_neurons))
            self.weights = np.asmatrix(np.random.uniform(a, -a, (self.neurons, prev_layer_neurons)))
            self.bias = np.asmatrix(np.random.uniform(a, -a, self.neurons)).T  # vettore colonna

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
        # La derivata della funzione E rispetto al generico peso wij [formula 1.5]
        # Quindi costruisco una matrice di derivate una per ogni matrice di pesi (quindi una per ogni livello)
        # Sarà sufficiente moltiplicare (righe per colonne) self.deltas con gli output z del layer a sinistra
        self.dE_dW = self.deltas * prev_layer.out.T

        # Per ogni layer: dE/db = dE/da * da/db = dE/da * 1 = dE/da = delta
        # Quindi la derivata di E risp. al vettore colonna bias è self.deltas
        self.dE_db = np.sum(self.deltas, axis=1)

    def update_weights(self, l_rate, momentum):
        # Momentum GD
        self.weights = self.weights - l_rate * self.dE_dW
        self.weights = -l_rate * self.dE_dW + momentum * self.weights

    def update_bias(self, l_rate, momentum):
        # Momentum GD
        self.bias = self.bias - l_rate * self.dE_db
        self.bias = -l_rate * self.dE_db + momentum * self.bias


if __name__ == '__main__':
    mndata = MNIST(path="data", return_type="numpy", mode="randomly_binarized")
    X_train, targets_train = mndata.load_training()  # 60.000 immagini da 28*28 colonne (features) ciascuna
    X_val, targets_val = mndata.load_testing()  # 10.000 immagini da 28*28 colonne (features) ciascuna

    X_train, targets_train = shuffle(X_train, targets_train.T)
    X_val, targets_val = shuffle(X_val, targets_val.T)

    # Ricavo il test set spaccando in due metà uguali il validation set
    # Il validation set passa da 10.000 immagini a 5000 immagini
    X_val, X_test = np.split(X_val, 2)  # 5000 immagini da 28*28 colonne (feature) ciascuna
    targets_val, targets_test = np.split(targets_val, 2)
    X_test, targets_test = shuffle(X_test, targets_test.T)

    targets_train = one_hot(targets_train)
    targets_val = one_hot(targets_val)
    targets_test = one_hot(targets_test)

    net = NeuralNetwork()
    d = np.shape(X_train)[1]  # dimensione dell'input = 28 * 28
    c = np.shape(targets_train)[0]  # dimensione dell'output = 10

    for m in (d, 1, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.build()

    best_net = net.fit(X_train, targets_train, X_val, targets_val, max_epochs=50)

    # Una lista di metriche che posso ri-implementare per testare il mio classificatore
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # Nota che il dataset MNIST è bilanciato
    predictions = best_net.predict(X_test)
    print("Accuracy score on test set is:", accuracy_score(targets_test, predictions) * 100, "%")