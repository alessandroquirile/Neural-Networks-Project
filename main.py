from mnist.loader import MNIST
from sklearn.utils import shuffle

from activation_functions import *
from loss_functions import *
from utils import *


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add(self, layer):
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
            # print(layer.weights)  # input weight matrix
            print("bias:", np.shape(layer.bias))
            # print(layer.bias)
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
            self.learning_rule(l_rate=0.001, momentum=0.9)  # tuning here
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
        self.type = type  # se il layer è input, hidden oppure output
        self.activation = activation  # funzione di attivazione associata al layer
        self.deltas = None  # vettore colonna di delta

    def configure(self, prev_layer_neurons):
        self.set_activation()
        self.weights = np.asmatrix(np.random.normal(-0.1, 0.02, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.normal(-0.1, 0.02, self.neurons)).T  # vettore colonna

    def set_activation(self):
        if self.activation is None:
            # th approx universale
            if self.type == "hidden":
                self.activation = sigmoid
            elif self.type == "output":
                self.activation = identity

    def forward_prop_step(self, z):
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
    mndata = MNIST(path="data", return_type="numpy")
    X, labels = mndata.load_training()

    # Rescale whole dataset within [0;1]
    X = X / 255

    X, labels = shuffle(X, labels.T)

    labels = one_hot(labels)

    # Per questioni di velocità riduco il dataset e considero i primi 500 item
    X_r = X[0:500, :]
    labels_r = labels[:, 0:500]

    # Split
    X_train = X_r[0:200, :]
    targets_train = labels_r[:, 0:200]
    X_val = X_r[200:400, :]
    targets_val = labels_r[:, 200:400]
    X_test = X_r[400:500, :]
    targets_test = labels_r[:, 400:500]

    net = NeuralNetwork()
    d = np.shape(X_train)[1]  # dimensione dell'input = 28x28
    c = np.shape(targets_train)[0]  # dimensione dell'output = 10

    for m in (d, 500, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add(layer)

    net.build()

    best_net = net.fit(X_train, targets_train, X_val, targets_val, max_epochs=200)

    # Una lista di metriche che posso ri-implementare per testare il mio classificatore
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # Nota che il dataset MNIST è bilanciato
    predictions = best_net.predict(X_test)
    accuracy = accuracy_score(targets_test, predictions) * 100
    print("Accuracy score on test set is:", accuracy, "%")