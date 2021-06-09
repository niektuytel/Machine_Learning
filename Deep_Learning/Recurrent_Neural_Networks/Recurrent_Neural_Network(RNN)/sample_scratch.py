import sys, os
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(1, os.getcwd() + "./../../network") 
from layers import *

def data(n_epochs, batch_size, biggest_number):
    X = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)
    y = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)

    for i in range(n_epochs):
        # One-hot encoding of nominal values
        start = np.random.randint(2, 7)
        one_hot = np.zeros((batch_size, biggest_number))
        one_hot[np.arange(batch_size), np.linspace(start, start*batch_size, num=batch_size, dtype=int)] = 1

        X[i] = one_hot
        y[i] = np.roll(X[i], -1, axis=0)
    y[:, -1, 1] = 1

    # return dataset
    return train_test_split(X, y, test_size=0.4)
X_train, X_test, y_train, y_test = data(n_epochs=3000, batch_size=10, biggest_number=61)

# define model
layers = [
    RNN(n_units=10, input_shape=(10, 61))
]
network = Network(layers=layers, loss="CrossEntropy")

# train network
history_loss = network.fit(X_train, y_train, n_epochs=500, batch_size=512)

# network result
for i in range(5):
    print(f"""
        question = [{' '.join(np.argmax(X_test[i], axis=1).astype('str'))}]
        answer   = [{' '.join((np.argmax(y_test, axis=2)[i]).astype('str'))}]
        predict  = [{' '.join((np.argmax(network.predict(X_test), axis=2)[i]).astype('str'))}]
    """)