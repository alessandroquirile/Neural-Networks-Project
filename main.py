from mnist.loader import MNIST
from sklearn.utils import shuffle

from activation_functions import *
from loss_functions import cross_entropy
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

        # Getting the minimum loss on validation set
        predictions_val = self.predict(X_val)
        min_loss_val = cross_entropy(predictions_val, targets_val)

        best_net = self  # net which minimize validation loss
        best_epoch = 0  # epoch where the validation loss is minimum

        # batch mode
        for epoch in range(max_epochs):
            predictions_train = self.predict(X_train)
            self.back_prop(targets_train, cross_entropy)
            self.learning_rule(l_rate=0.000005, momentum=0.9)
            loss_train = cross_entropy(predictions_train, targets_train)
            e_loss_train.append(loss_train)

            # Validation
            predictions_val = self.predict(X_val)
            loss_val = cross_entropy(predictions_val, targets_val)
            e_loss_val.append(loss_val)

            print(f"E({epoch}) "
                  f"train_loss: {loss_train} "
                  f"val_loss: {loss_val} "
                  f"val_acc: {accuracy_score(targets_val, predictions_val) * 100} %")

            if loss_val < min_loss_val:
                min_loss_val = loss_val
                best_epoch = epoch
                best_net = self

        print(f"Validation loss is minimum at epoch {best_epoch}")

        plot(np.arange(max_epochs), e_loss_train, e_loss_val)

        return best_net

    # Matrix of predictions where the i-th column corresponds to the i-th item
    def predict(self, dataset):
        z = dataset
        for layer in self.layers:
            z = layer.forward_prop_step(z)
        return z

    def back_prop(self, target, loss):
        for i, layer in enumerate(self.layers[:0:-1]):
            next_layer = self.layers[-i]
            prev_layer = self.layers[-i - 2]
            layer.back_prop_step(next_layer, prev_layer, target, loss)

    def learning_rule(self, l_rate, momentum):
        # Momentum GD
        for layer in [layer for layer in self.layers if layer.type != "input"]:
            layer.update_weights(l_rate, momentum)
            layer.update_bias(l_rate, momentum)


class Layer:

    def __init__(self, neurons, type=None, activation=None):
        self.dE_dW = None  # derivatives dE/dW where W is the weights matrix
        self.dE_db = None  # derivatives dE/db where b is the bias
        self.dact_a = None  # derivative of the activation function
        self.out = None  # layer output
        self.weights = None  # input weights
        self.bias = None  # layer bias
        self.w_sum = None  # weighted_sum
        self.neurons = neurons  # number of neurons
        self.type = type  # input, hidden or output
        self.activation = activation  # activation function
        self.deltas = None  # for back-prop
        self.diff_w = None
        self.diff_b = None

    def configure(self, prev_layer_neurons):
        self.set_activation()
        self.weights = np.asmatrix(np.random.uniform(-0.02, 0.02, (self.neurons, prev_layer_neurons)))
        self.bias = np.asmatrix(np.random.uniform(-0.02, 0.02, self.neurons)).T
        self.diff_w = np.asmatrix(np.zeros(shape=np.shape(self.weights)))
        self.diff_b = np.asmatrix(np.zeros(shape=np.shape(self.bias)))

    def set_activation(self):
        if self.activation is None:
            if self.type == "hidden":
                self.activation = relu
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
            self.dact_a = self.activation(self.w_sum, derivative=True)
            self.deltas = np.multiply(self.dact_a,
                                      local_loss(self.out, target, derivative=True))
        else:
            self.dact_a = self.activation(self.w_sum, derivative=True)  # (m,batch_size)
            self.deltas = np.multiply(self.dact_a, np.dot(next_layer.weights.T, next_layer.deltas))

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

    # In modo tale che lo split sia casuale
    X_train, targets_train = shuffle(X_train, targets_train.T)

    # Data pre processing
    X_train = X_train / 255  # normalization within [0;1]
    X_test = X_test / 255  # normalization within [0;1]
    X_train = X_train.T  # data transposition
    X_test = X_test.T  # data transposition

    # Split: i 60.000 dati di training li divido in 50.000 per il training e 10.000 per il validation
    X_val, X_train = np.hsplit(X_train, [10000])
    targets_val, targets_train = np.hsplit(targets_train, [10000])
    if not balanced(targets_train, targets_val, targets_test):
        raise Exception("Classes are not balanced")

    # One hot
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

    # Testing
    predictions_test = best_net.predict(X_test)
    accuracy_test = accuracy_score(targets_test, predictions_test)
    print(f"Accuracy score on test set is: {accuracy_test * 100} %")
