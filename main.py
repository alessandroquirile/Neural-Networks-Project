from matplotlib import pyplot as plt

from activation_functions import *
from loss_functions import *


def generate_data(n_items, n_features, c):
    X = np.asmatrix(np.random.normal(size=(n_items, n_features)))
    targets = np.asarray(np.random.randint(c, size=n_items))
    targets = np.asmatrix(np.eye(np.max(targets) + 1)[targets])  # 1-hot encoding
    return X, targets


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
            print("l_step_weights:", np.shape(layer.step_weights))
            print(layer.step_weights)
            print("l_step_bias:", np.shape(layer.step_bias))
            print(layer.step_bias)
            print("")

    def train(self, X, targets):
        # batch mode
        MAX_EPOCHS = 200
        loss_function = cross_entropy
        epoch_loss = []
        for epoch in range(MAX_EPOCHS):
            E = 0  # errore sull'intero DS
            for i, x in enumerate(X):
                target = targets[i].T
                prediction = self.forward_prop(x.T)
                E_n = loss_function(prediction, target, post_process=True)
                E += E_n
                self.back_prop(target, local_loss=loss_function)
            self.learning_rule(l_rate=0.01, momentum=0.01)  # 0.00001
            # print("E(%d) on TrS is:" % epoch, E)
            epoch_loss.append(E)

            print("E(%d) on TrS is:" % epoch, E)

        plot(np.arange(MAX_EPOCHS), epoch_loss)

    # Corrisponde a simnet
    # Restituisce una matrice di predizioni dove la i-esima riga corrisponde alla i-esima predizione
    # Cioè quella calcolata sull'item i-esimo
    def predict(self, X):
        predictions = np.asmatrix(np.zeros(shape=(np.shape(X)[0], c)))
        for i, x in enumerate(X):
            prediction = self.forward_prop(x.T)
            """print("Prediction on %d item" % i)  # dbg
            print(prediction)  # dbg
            print("")  # dbg"""
            predictions[i, :] = prediction.T
        return predictions

    def forward_prop(self, z):
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z  # output della rete sull'input z

    def back_prop(self, target, local_loss):
        # Back-propagation
        # Scarta il layer di input che non è coinvolto nella procedura
        # Tiene traccia anche del layer successivo ad ognuno, per la BP2
        # E di quello precedente, per il calcolo dE/dW
        for i, layer in enumerate(self.layers[:0:-1]):
            next_layer = self.layers[-i]  # quello sulla destra
            prev_layer = self.layers[-i - 2]  # quello sulla sinistra
            # print("Layer[%d] type:" % i, layer.type)
            layer.back_prop_step(next_layer, prev_layer, target, local_loss)
            # print("")

    """def rprop(self, epoch, eta_minus, eta_plus, minstep, maxstep):
        # RProp - alternativa al GD
        for layer in [layer for layer in self.layers if layer.type != "input"]:
            g_t_w = layer.dE_dW  # dE/dW per il layer corrente, all'epoca t (attuale)
            g_tprev_w = layer.dE_dW_tprev  # dE/dW per il layer corrente, all'epoca t-1
            g_t_b = layer.dE_db  # dE/db per il layer corrente, all'epoca t (attuale)
            g_tprev_b = layer.dE_db_tprev  # dE/db per il layer corrente, all'epoca t-1

            # Aggiorna i Deltaij associati ai pesi e ai bias
            layer.update_steps_weights(np.multiply(g_t_w, g_tprev_w), eta_plus, eta_minus, maxstep, minstep)
            layer.update_steps_bias(np.multiply(g_t_b, g_tprev_b), eta_plus, eta_minus, maxstep, minstep)

            # Valutare se saltare l'aggiornamento quando e=0 (t=1)
            layer.weights -= np.multiply(np.sign(g_t_w), layer.step_weights)
            layer.bias -= np.multiply(np.sign(g_t_b), layer.step_bias)

            # l'aggiornamento richiede la copia in modo tale che le due variabili puntino a oggetti distinti
            layer.dE_dW_tprev = np.copy(g_t_w)
            layer.dE_db_tprev = np.copy(g_t_b)"""

    def learning_rule(self, l_rate, momentum):
        # Momentum GD
        for layer in [layer for layer in self.layers if layer.type != "input"]:
            layer.update_weights(l_rate, momentum)
            layer.update_bias(l_rate, momentum)


class Layer:

    def __init__(self, neurons, type=None, activation=None):
        """self.dE_dW_tprev = 0  # matrice delle derivate dE/dW all'epoca precedente per la RProp
        self.dE_db_tprev = 0  # matrice delle derivate dE/db all'epoca precedente per la RProp
        self.step_weights = None  # matrice di Delta maiuscolo associato ad ogni peso per la RProp
        self.step_bias = None  # matrice di Delta maiuscolo associato ad ogni bias per la RProp"""

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
        self.step_weights = np.asmatrix(np.ones(np.shape(self.weights)))
        self.dE_dW_tprev = np.zeros(np.shape(self.weights))  # Per la RProp
        self.dE_db_tprev = np.zeros(np.shape(self.bias))  # Per la RProp
        self.bias = np.asmatrix(np.ones(self.neurons)).T
        self.step_bias = np.asmatrix(np.random.normal(1,1, self.neurons)).T"""

        self.weights = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        """self.step_weights = np.asmatrix(np.random.normal(0, 0.5, np.shape(self.weights)))  # Per la RProp
        self.dE_dW_tprev = np.zeros(np.shape(self.weights))  # Per la RProp
        self.dE_db_tprev = np.zeros(np.shape(self.bias))  # Per la RProp"""
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T  # vettore colonna
        """self.step_bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T"""  # Per la RProp

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
                                      local_loss(self.out, target, derivative=True, post_process=True))  # cx1
        else:
            # BP2
            self.dact_a = self.activation(self.w_sum, derivative=True)  # mx1
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
        self.dE_dW += self.dEn_dW

        # La derivata dE/db sull'intero DS è la somma per n=1 a N di dE/db sul singolo item
        self.dE_db += self.dEn_db

        """print(self.dE_dW) # dbg
        print("")"""

        # dbg
        """print("deltas shape:", np.shape(self.deltas))
        print(self.deltas)
        print("prev_layer.out.T shape:", np.shape(prev_layer.out.T))
        print(prev_layer.out.T)"""

        """print("dE/dW shape:", np.shape(self.dE_dW))  # dE/dW sull'n-esima istanza
        print(self.dE_dW)
        print("dE/db shape:", np.shape(self.dE_db))  # dE/db sull'n-esima istanza
        print(self.dE_db)
        print("dE/dW_tot shape:", np.shape(self.dE_dW_tot))  # dE/dW
        print(self.dE_dW_tot)
        print("dE/db_tot shape:", np.shape(self.dE_db_tot))  # dE/db
        print(self.dE_db_tot)
        print("")"""

    """def update_steps_weights(self, epsilon, eta_plus, eta_minus, maxstep, minstep):
        # epsilon corrisponde a g(t) * g(t-1)
        self.step_weights[epsilon > 0] = np.minimum(self.step_weights.A[epsilon > 0] * eta_plus, maxstep)
        self.step_weights[epsilon < 0] = np.maximum(self.step_weights.A[epsilon < 0] * eta_minus, minstep)

        # Oppure
        for (i, j), epsilon_ij in np.ndenumerate(epsilon):
            if epsilon_ij > 0:
                # print("Incremento")
                # print("Before", self.step_weights[i,j])
                self.step_weights[i, j] = np.minimum(eta_plus * self.step_weights[i, j], maxstep)
                # print("After", self.step_weights[i,j])
            elif epsilon_ij < 0:
                # print("Decremento")
                # print("Before", self.step_weights[i,j])
                self.step_weights[i, j] = np.maximum(eta_minus * self.step_weights[i, j], minstep)
                # print("After", self.step_weights[i,j])
            # print("")"""

    """def update_steps_bias(self, epsilon, eta_plus, eta_minus, maxstep, minstep):
        # epsilon corrisponde a g(t) * g(t-1)
        self.step_bias[epsilon > 0] = np.minimum(self.step_bias.A[epsilon > 0] * eta_plus, maxstep)
        self.step_bias[epsilon < 0] = np.maximum(self.step_bias.A[epsilon < 0] * eta_minus, minstep)"""

    def update_weights(self, l_rate, momentum):
        # Momentum GD
        layer.weights = layer.weights - l_rate * layer.dE_dW
        layer.weights = -l_rate * layer.dE_dW + momentum * layer.weights

    def update_bias(self, l_rate, momentum):
        # Momentum GD
        layer.bias = layer.bias - l_rate * layer.dE_db
        layer.bias = -l_rate * layer.dE_db + momentum * layer.bias


if __name__ == '__main__':

    # Prima di fare MNIST, faccio un caso più semplice
    # Supponiamo una classificazione a 3 classi: cane, gatto, topo
    # La classe cane è 0 -> 000
    # La classe gatto è 1 -> 010
    # La classe topo è 2 -> 001
    net = NeuralNetwork()
    d = 4  # dimensione dell'input (n_features)
    c = 3  # classi in output

    for m in (d, 4, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.build()

    X, targets = generate_data(n_items=10, n_features=d, c=c)

    net.train(X, targets)