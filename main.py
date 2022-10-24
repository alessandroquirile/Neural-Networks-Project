from mnist.loader import MNIST

from activation_functions import relu, identity
from data_processing import normalize, split, one_hot
from loss_functions import cross_entropy
from metrics import accuracy_score
from utils import *


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def build(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.ty = "input"
            else:
                layer.ty = "output" if i == len(self.layers) - 1 else "hidden"
                layer.configure(self.layers[i - 1].neurons)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print("Layer", i)
            print("neurons:", layer.neurons)
            print("type:", layer.ty)
            print("act:", layer.activation)
            print("weights:", np.shape(layer.weights))
            print(layer.weights)
            print("bias:", np.shape(layer.bias))
            print(layer.bias)
            print("")

    def fit(self, X_train, targets_train, X_val, targets_val, max_epochs):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        # Getting the minimum loss on validation set
        predictions_val = self.predict(X_val)
        min_val_loss = cross_entropy(predictions_val, targets_val)
        best_net = self  # net which minimize validation loss
        best_epoch = 0  # epoch where the validation loss is minimum

        # batch mode
        for epoch in range(max_epochs):
            predictions_train = self.predict(X_train)
            train_acc = accuracy_score(targets_train, predictions_train)
            train_accuracies.append(train_acc)
            self.back_prop(targets_train, cross_entropy)
            self.learning_rule(l_rate=0.000005, momentum=0.9)
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
                best_net = self

            print(f"E({epoch}): "
                  f"train_loss: {train_loss :.14f} "
                  f"val_loss: {val_loss :.14f} "
                  f"train_acc: {train_acc * 100 :.2f} % "
                  f"val_acc: {val_acc * 100 :.2f} %")

        print(f"Validation loss is minimum at epoch: {best_epoch}")

        plot_losses(np.arange(max_epochs), train_losses, val_losses)
        plot_accuracies(np.arange(max_epochs), train_accuracies, val_accuracies)

        return best_net

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
        self.neurons = neurons  # number of neurons
        self.ty = ty  # input, hidden or output
        self.activation = activation  # activation function
        self.out = None  # layer output
        self.weights = None  # input weights
        self.bias = None  # layer bias
        self.w_sum = None  # weighted_sum
        self.dact = None  # derivative of the activation function
        self.dE_dW = None  # derivatives dE/dW where W is the input weights matrix
        self.dE_db = None  # derivatives dE/db where b is the bias
        self.deltas = None  # for back-prop
        self.diff_w = None  # for MGD
        self.diff_b = None  # for MGD

    def configure(self, prev_layer_neurons):
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


if __name__ == '__main__':
    mndata = MNIST(path="data", return_type="numpy")
    X_train, targets_train = mndata.load_training()  # 60.000 images, 28*28 features
    X_test, targets_test = mndata.load_testing()  # 10.000 images, 28*28 features

    X_train, targets_train = normalize(X_train, targets_train, shuffle=True)
    X_test, targets_test = normalize(X_test, targets_test, shuffle=False)

    X_val, targets_val = split(X_train, targets_train, val_size=10000)

    if not balanced(targets_train, targets_val, targets_test):
        raise Exception("Classes are not balanced")

    targets_train = one_hot(targets_train)
    targets_val = one_hot(targets_val)
    targets_test = one_hot(targets_test)

    net = NeuralNetwork()
    d = np.shape(X_train)[0]  # number of features, 28x28
    c = np.shape(targets_train)[0]  # number of classes, 10

    # Net creation
    for m in (d, 100, c):
        net.add(Layer(m))

    net.build()

    best_net = net.fit(X_train, targets_train, X_val, targets_val, max_epochs=50)

    # Model testing
    predictions_test = best_net.predict(X_test)
    test_acc = accuracy_score(targets_test, predictions_test)
    print(f"Accuracy score on test set is: {test_acc * 100 :.2f} %")
