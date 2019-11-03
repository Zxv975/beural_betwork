#!/usr/bin/env python
# coding: utf-8

# In[25]:


# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import mnist_loader
import network_modded_bryte as ntwk
from PIL import Image


# In[34]:


w, h = 512, 512
data = np.zeros((h, w, 3), dtype = np.uint8)
for i in range(w):
    for j in range(h):
        for k in range(3):
            data[i, j, k] = np.random.randint(255)

data[4:100, 400:450, 1:] = [200, 200]          
img = Image.fromarray(data, 'RGB')
#img.save('my.png')
img.show()


# In[35]:


w, h = 512, 512
data = np.zeros((h, w, 3), dtype = np.uint8)

data[236:320, 256:512, :] = [200, 200, 200]          
data[5, 6:10, 1:2]
img = Image.fromarray(data, 'RGB')
#img.save('my.png')
img.show()


# In[74]:


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = ntwk.Network([784, 15, 10])
net.SGD(training_data, 50, 10, 0.01, test_data = test_data)


# In[4]:


av = net.feedbackward(5)
x = av[-1].reshape(28, 28) * 256
img = Image.fromarray(x)
img.show()


# In[31]:


net.biases = net.biases[::-1]
net.weights = net.weights[::-1]
#for i in range(len(net.weights)):
#    net.weights[i] = np.transpose(net.weights[i])


# In[93]:


#create the output of 4

y = 4
y_vec = np.zeros((1, 10)) 
y_vec[0, y] = 1
c = create(y_vec)
c = c*256

#chara = np.reshape(c, [28, 28])
img = Image.fromarray(chara)
img.show()


# In[49]:


[n.shape for n in net.weights]
c.shape


# In[72]:


def create(a):
    """Return the reverse input of the network if ``a`` is desired output."""
    net.biases = net.biases[::-1]
    net.weights = net.weights[::-1]
    
    for b, w in zip(net.biases, net.weights):
        a = ntwk.sigmoid(np.dot(a, w))
    
    net.biases = net.biases[::-1]
    net.weights = net.weights[::-1]
    return a


# In[72]:


def fb(y):
    y_vec = np.zeros((10, 1)) + 0.000000001
    y_vec[y, 0] = 0.999999999
    activation = y_vec
    activations = [y_vec] # list to store all the activations, layer by layer
    vs = [] # list to store all the z vectors, layer by layer 
    for b, w in zip(net.biases[::-1], net.weights[::-1]):
        v = ntwk.logit(activation)
        print(activation)
        activation = np.dot(np.linalg.pinv(w), v - b)
        vs.append(v)
        activations.append(activation)
    return activations

def ff(x):
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = ntwk.sigmoid(z)
        activations.append(activation)
    return activations


# In[73]:


input = np.random.rand(784, 1)
act = ff(input)
av = fb(4)


# In[84]:


e = 0.001
y = np.zeros((10, 1)) + e
y[3] = 1 - e
x = input
a = np.dot(np.linalg.pinv(net.weights[1]), ntwk.logit(y) - net.biases[1])
u = ntwk.sigmoid(np.dot(net.weights[0], x) + net.biases[0])

print(a)
print(u)


# In[70]:


ntwk.sigmoid(-0.00001)


# In[88]:


y = 3;
y_vec = np.zeros((10, 1))
y_vec[y, 0]


# In[82]:


np.zeros((10, 1)).shape


# In[41]:


net.feedbackward(3)


# In[ ]:




