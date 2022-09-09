"""from main import *

net = NeuralNetwork()
d = 2  # dipenderà da MNIST
c = 1  # sarà 10

hidden_l = int(input("Numero di hidden layer: "))

net.add_layer(Layer(d))
for i in range(hidden_l):
    m = int(input("Numeri di neuroni nell'hidden layer " + str(i + 1) + ": "))
    net.add_layer(Layer(m))
net.add_layer(Layer(c))

net.create()

net.summary()"""
