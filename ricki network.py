import network_modded
import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network_modded.Network([784,30,10])
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
