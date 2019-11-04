import network_modded
import mnist_loader
import json
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def net_encode(net):
    # Convert numpy arrays to lists to be serialised
    bias_data = [b.tolist() for b in net.biases]
    weight_data = [w.tolist() for w in net.weights]
    layer_data = net.num_layers
    size_data = net.sizes

    # Generate an array of data and serialise it
    return json.dumps([bias_data, weight_data, layer_data, size_data])

def net_decode(data):
    # Deserialise the data
    json_list = json.loads(data)

    # Convert the weights and biases back into numpy arrays
    my_bias = [np.asarray(b) for b in json_list[0]]
    my_weights = [np.asarray(w) for w in json_list[1]]

    return [my_bias, my_weights, json_list[2], json_list[3]]

net = network_modded.Network([784,30,10])
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

data = net_encode(net)

# Write to file
f = open("network_data.txt", "w")
f.write(data)
f.close()

# Read from file
f = open("network_data.txt", "r")
my_data = net_decode(f.read())
f.close()

print (my_data)
