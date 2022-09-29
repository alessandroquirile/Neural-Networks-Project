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
        MAX_EPOCHS = 5
        loss_function = cross_entropy
        E = 0  # errore sull'intero DS
        # epoch_loss = []
        for epoch in range(MAX_EPOCHS):
            for i, x in enumerate(X):
                target = targets[i]
                prediction = self.forward_prop(x.T)
                E_n = loss_function(prediction, target)
                E += E_n
                self.back_prop(target, local_loss=loss_function)
            # epoch_loss.append(E)
            print("E(%d) on TrS is:" % epoch, E)
            self.learning_rule(epoch, eta_minus=0.5, eta_plus=1.2, minstep=10 ** -6, maxstep=50)
        """df = pd.DataFrame(epoch_loss)
        df_plot = df.plot(kind="line", grid=True).get_figure()
        df_plot.savefig("plot.pdf")"""

    def forward_prop(self, z):
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z  # output della rete

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

    def learning_rule(self, epoch, eta_minus, eta_plus, minstep, maxstep):
        # todo: l'algoritmo va applicato sia per i pesi che per i bias

        for layer in [layer for layer in self.layers if layer.type != "input"]:
            g_t = layer.dE_dW  # dE/dW per il layer corrente, all'epoca t (attuale)
            g_tprev = layer.dE_dW_tprev  # dE/dW per il layer corrente, all'epoca t-1

            # Aggiorna i Deltaij associati ai pesi
            layer.update_steps(np.multiply(g_t, g_tprev), eta_plus, eta_minus, maxstep, minstep)

            # Valutare se saltare l'aggiornamento quando e=0 (t=1)
            layer.weights -= np.multiply(np.sign(g_t), layer.step_weights)

            # l'aggiornamento richiede la copia in modo tale che le due variabili puntino a oggetti distinti
            layer.dE_dW_tprev = np.copy(g_t)

        """for layer in self.layers:
            if layer.type != "input":
                g_t = layer.dE_dW  # dE/dW per il layer corrente, all'epoca t (attuale)
                g_tprev = layer.dE_dW_tprev  # dE/dW per il layer corrente, all'epoca t-1

                # Aggiorna i Deltaij associati ai pesi
                layer.update_steps(np.multiply(g_t, g_tprev), eta_plus, eta_minus, maxstep, minstep)

                # Valutare se saltare l'aggiornamento quando e=0 (t=1)
                layer.weights -= np.multiply(np.sign(g_t), layer.step_weights)

                # l'aggiornamento richiede la copia in modo tale che le due variabili puntino a oggetti distinti
                layer.dE_dW_tprev = np.copy(g_t)"""


class Layer:

    def __init__(self, neurons, type=None, activation=None):
        self.dE_dW_tprev = 0  # matrice delle derivate dE/dW all'epoca precedente per la RProp
        self.step_weights = None  # matrice di Delta maiuscolo associato ad ogni peso per la RProp
        self.step_bias = None  # matrice di Delta maiuscolo associato ad ogni bias per la RProp

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
        """# Solo per dbg
        self.weights = np.asmatrix(np.ones((self.neurons, prev_layer_neurons)))
        self.step_weights = np.asmatrix(np.ones(np.shape(self.weights)))
        self.dE_dW_tprev = np.zeros(np.shape(self.weights))  # Per la RProp
        self.bias = np.asmatrix(np.ones(self.neurons)).T
        self.step_bias = np.asmatrix(np.ones(np.shape(self.neurons))).T"""

        self.weights = np.asmatrix(np.random.normal(0, 0.5, (self.neurons, prev_layer_neurons)))
        self.step_weights = np.asmatrix(np.random.normal(0, 0.5, np.shape(self.weights)))
        self.dE_dW_tprev = np.zeros(np.shape(self.weights))  # Per la RProp
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T  # vettore colonna
        self.step_bias = np.asmatrix(np.random.normal(0, 0.5, self.neurons)).T

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
            self.deltas = np.multiply(self.dact_a, local_loss(self.out, target, derivative=True))  # cx1
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

    def update_steps(self, epsilon, eta_plus, eta_minus, maxstep, minstep):
        # epsilon corrisponde a g(t) * g(t-1)

        self.step_weights[epsilon > 0] = np.minimum(self.step_weights.A[epsilon > 0] * eta_plus, maxstep)
        self.step_weights[epsilon < 0] = np.maximum(self.step_weights.A[epsilon < 0] * eta_minus, minstep)

        """for (i, j), epsilon_ij in np.ndenumerate(epsilon):
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


if __name__ == '__main__':
    net = NeuralNetwork()
    d = 2  # dimensione dell'input
    c = 1  # dimensione dell'output

    for m in (d, 4, 4, c):
        layer = Layer(m)  # costruisco un layer con m neuroni
        net.add_layer(layer)

    net.create()

    # net.summary()  # dbg

    X = np.asmatrix([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
    ])

    targets = np.asarray([1, 0, 0, 0])

    net.train(X, targets)  # gli passo X e i targets interi, del training set
