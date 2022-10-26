from activation_functions import relu, identity
from loss_functions import cross_entropy
from metrics import accuracy_score
from utils import *


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.ty = "input"
            else:
                layer.ty = "output" if i == (len(self.layers) - 1) else "hidden"
                layer.build(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)
            print("neurons:", layer.neurons)
            print("type:", layer.ty)
            print("act:", layer.activation)
            print("weights:", np.shape(layer.weights))
            # print(layer.weights)
            print("bias:", np.shape(layer.bias))
            # print(layer.bias)
            print("")

    def fit(self, X_train, targets_train, X_val, targets_val, max_epochs, l_rate, momentum):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        predictions_val = self.predict(X_val)
        min_val_loss = cross_entropy(predictions_val, targets_val)  # Getting the minimum loss on validation set
        best_model = self  # net which minimize validation loss
        best_epoch = 0  # epoch where the validation loss is the smallest

        # batch mode
        for epoch in range(max_epochs):
            predictions_train = self.predict(X_train)
            train_acc = accuracy_score(targets_train, predictions_train)
            train_accuracies.append(train_acc)
            self.back_prop(targets_train, cross_entropy)
            self.learning_rule(l_rate=l_rate, momentum=momentum)
            train_loss = cross_entropy(predictions_train, targets_train)
            train_losses.append(train_loss)

            # Model selection
            predictions_val = self.predict(X_val)
            val_acc = accuracy_score(targets_val, predictions_val)
            val_accuracies.append(val_acc)
            val_loss = cross_entropy(predictions_val, targets_val)
            val_losses.append(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                best_model = self

            print(f"E({epoch}): "
                  f"train_loss: {train_loss :.14f} "
                  f"val_loss: {val_loss :.14f} "
                  f"train_acc: {train_acc * 100 :.2f} % "
                  f"val_acc: {val_acc * 100 :.2f} %")

        print(f"Validation loss is minimum at epoch: {best_epoch}")
        plot_losses(np.arange(max_epochs), train_losses, val_losses)
        plot_accuracies(np.arange(max_epochs), train_accuracies, val_accuracies)

        return best_model

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_prop_step(X)
        return X

    def back_prop(self, target, loss):
        for i, layer in enumerate(self.layers[:0:-1]):
            next_layer = self.layers[-i]
            prev_layer = self.layers[-i - 2]
            layer.back_prop_step(next_layer, prev_layer, target, loss)

    def learning_rule(self, l_rate, momentum):
        for layer in [lyr for lyr in self.layers if lyr.ty != "input"]:
            layer.update_weights(l_rate, momentum)
            layer.update_bias(l_rate, momentum)


class Layer:

    def __init__(self, neurons, ty=None, activation=None):
        self.neurons = neurons
        self.ty = ty
        self.activation = activation
        self.out = None
        self.weights = None
        self.bias = None
        self.w_sum = None
        self.dact = None
        self.dE_dW = None
        self.dE_db = None
        self.deltas = None
        self.diff_w = None
        self.diff_b = None

    def build(self, prev_layer_neurons):
        self.set_activation()
        self.weights = np.asmatrix(np.random.uniform(-0.02, 0.02, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.uniform(-0.02, 0.02, self.neurons)).T
        self.diff_w = np.asmatrix(np.zeros(shape=np.shape(self.weights)))
        self.diff_b = np.asmatrix(np.zeros(shape=np.shape(self.bias)))

    def set_activation(self):
        if self.activation is None:
            if self.ty == "hidden":
                self.activation = relu
            elif self.ty == "output":
                self.activation = identity

    def forward_prop_step(self, z):
        if self.ty == "input":
            self.out = z
        else:
            self.w_sum = np.dot(self.weights, z) + self.bias
            self.out = self.activation(self.w_sum)
        return self.out

    def back_prop_step(self, next_layer, prev_layer, target, loss):
        if self.ty == "output":
            self.dact = self.activation(self.w_sum, derivative=True)
            self.deltas = np.multiply(self.dact,
                                      loss(self.out, target, derivative=True))
        else:
            self.dact = self.activation(self.w_sum, derivative=True)  # (m,batch_size)
            self.deltas = np.multiply(self.dact, np.dot(next_layer.weights.T, next_layer.deltas))
        self.dE_dW = self.deltas * prev_layer.out.T
        self.dE_db = np.sum(self.deltas, axis=1)

    def update_weights(self, l_rate, momentum):
        self.diff_w = l_rate * self.dE_dW + momentum * self.diff_w
        self.weights = self.weights - self.diff_w

    def update_bias(self, l_rate, momentum):
        self.diff_b = l_rate * self.dE_db + momentum * self.diff_b
        self.bias = self.bias - self.diff_b
