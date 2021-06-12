import pickle, gzip
import matplotlib.pyplot as plt 
import numpy as np
import sys

from vae import VAE

with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    train, test, val = pickle.load(f, encoding='latin1')
    mnist_train = train[0]
    mnist_test = test[0]

params = {
    'alpha' : 0.005,
    'iter' : 20000,
    'activation': 'sigmoid',
    'loss': 'squared_error',
    'batch_size': 64
}

import sys, os
def make_dir():
    image_dir = "./sample_scratch_output"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
make_dir()

example = VAE([[784, 200], [200, 784]], 2, params)
example.learn(mnist_train)




fig, ax = plt.subplots(2,3, figsize = (10, 8))

for i in range(3):
    in_digit = mnist_test[i][None,:]
    out_digit = example.encode_decode(in_digit)
    ax[0,i].matshow(in_digit.reshape((28,28)),  cmap='gray', clim=(0,1))
    ax[1,i].matshow(out_digit.reshape((28,28)), cmap='gray', clim=(0,1))
pass





fig, ax = plt.subplots(2,2, figsize = (6, 6))

a = np.array([1, 3])
b = np.array([1, 3])

for i, z1 in enumerate(a):
    for j, z2 in enumerate(b):
        ax[i,j].matshow(example.generate(np.array([z1,z2])).reshape((28,28)),  cmap='gray', clim=(0,1))
pass





plt.show()

