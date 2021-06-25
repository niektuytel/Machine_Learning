import numpy as np
from . import activations
from . import losses

class Network:

    def __init__(self, dimensions, params):
        '''intializes weights matrix and parameters'''

        # initialize weights of network
        self.weights = {}
        for i in range(len(dimensions)-1):
            self.weights[i] = np.random.uniform(-0.1, 0.1, 
                    (dimensions[i], dimensions[i+1]))

        # hyperparameters
        self.alpha = params['alpha']
        self.iter = params['iter']
        self.batch_size = params['batch_size']

        if type(params['activation']) is str and params['activation'] in activations.activation_table:
            self.activation = activations.activation_table[params['activation']]
        else:
            self.activation = params['activation']

        if type(params['loss']) is str and params['loss'] in losses.loss_table:
            self.loss = losses.loss_table[params['loss']]
        else:
            self.loss = params['loss']

    def _feedforward(self, X):
        '''feedforward update step'''
        self._z = {}
        self._z_act = {0: X}

        for i in range(len(self.weights)):
            self._z[i] = self._z_act[i] @ self.weights[i]
            self._z_act[i+1] = self.activation(self._z[i])[0]
        return self._z_act[i+1]

    def _backprop(self, X, y, yhat):
        '''back-propagation algorithm'''
        n = len(self.weights)
        delta = -1 * self.loss(y, yhat)[1] * self.activation(self._z[n-1])[1]
        grad_weights = {n-1: self._z_act[n-1].T @ delta}

        for i in reversed(range(len(self.weights)-1)):
            delta = delta @ self.weights[i+1].T * self.activation(self._z[i])[1]
            grad_weights[i] = self._z_act[i].T @ delta

        return grad_weights

    def train(self, X, y):
        '''trains model using stochastic gradient descent'''
        X_batch = X
        y_batch = y

        for i in range(self.iter):
            if self.batch_size > 0 and self.batch_size < X.shape[0]:
                k = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
                X_batch = X[k,:]
                y_batch = y[k,:]

            yhat = self._feedforward(X_batch)
            grad_weights = self._backprop(X_batch, y_batch, yhat)

            for j in range(len(self.weights)):
                self.weights[j] -= self.alpha * grad_weights[j]

    def predict(self, X):
        '''predicts on trained model'''
        z_act = X
        for i in range(len(self.weights)):
            z = z_act @ self.weights[i]
            z_act = self.activation(z)[0]
        return z_act
