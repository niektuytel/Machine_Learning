import numpy as np
from .network import Network
from . import losses
from . import activations

class VAE:
    '''Variational Autoencoder'''

    def __init__(self, dimensions, latent_dim, params):

        self.latent_dim = latent_dim
        self.encoder = Network(dimensions[0] + [2], params)
        self.decoder = Network([latent_dim] + dimensions[1], params)
        
        for i in range(len(self.encoder.weights)):
            self.encoder.weights[i] = np.abs(self.encoder.weights[i])
            
        for i in range(len(self.encoder.weights)):
            self.decoder.weights[i] = np.abs(self.decoder.weights[i])
        
        self.batch_size = params['batch_size']
        self.iter = params['iter']
        self.encoder.loss = losses.loss_table['identity']
        self.decoder.loss = losses.loss_table['squared_error']
        
        if type(params['activation']) is str and params['activation'] in activations.activation_table:
            self.activation = activations.activation_table[params['activation']]
        else:
            self.activation = params['activation']

    def _forwardstep(self, X):
        # encoder learns parameters
        latent = self.encoder._feedforward(X)
        self.mu = latent[:,0]
        self.sigma = np.exp(latent[:,1])

        # sample from gaussian with learned parameters
        epsilon = np.random.normal(0, 1, size=(X.shape[0], self.latent_dim))
        z_sample = self.mu[:,None] + np.sqrt(self.sigma)[:,None] * epsilon

        # pass sampled vector through to decoder
        X_hat = self.decoder._feedforward(z_sample)
        return X_hat

    def _kl_divergence_loss(self):
        d_mu = self.mu
        d_s2 = 1 - 1 / (2 * (self.sigma + 1e-6))
        return np.vstack((d_mu, d_s2)).T


    def _backwardstep(self, X, X_hat):
        # propagate reconstuction error through decoder
        n = len(self.decoder.weights)
        delta = -1 * self.decoder.loss(X, X_hat)[1] * self.activation(self.decoder._z[n-1])[1]
        decoder_weights = {n-1: self.decoder._z_act[n-1].T @ delta}

        for i in reversed(range(len(self.decoder.weights)-1)):
            delta = delta @ self.decoder.weights[i+1].T * self.activation(self.decoder._z[i])[1]
            decoder_weights[i] = self.decoder._z_act[i].T @ delta

        # add kl-divergence loss
        m = len(self.encoder.weights)
        kl_loss = self._kl_divergence_loss()
        kl_delta = kl_loss * self.activation(self.encoder._z[m-1])[1]

        delta = delta @ self.decoder.weights[0].T * self.activation(self.encoder._z[m-1])[1] 
        delta = delta + kl_delta
        encoder_weights = {m-1: self.encoder._z_act[n-1].T @ delta}

        # propagate kl error through encoder
        for i in reversed(range(len(self.decoder.weights)-1)):
            delta = delta @ self.encoder.weights[i+1].T * self.activation(self.encoder._z[i])[1]
            encoder_weights[i] = self.encoder._z_act[i].T @ delta

        print(f"\r delta: {delta.mean()}", end="")
        # start delta: 1.7775254505725806e-07

        return encoder_weights, decoder_weights

    def learn(self, X):
        X_batch = X

        for i in range(self.iter):
            if self.batch_size > 0 and self.batch_size < X.shape[0]:
                k = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
                X_batch = X[k,:]

            X_hat = self._forwardstep(X_batch)
            grad_encoder, grad_decoder = self._backwardstep(X_batch, X_hat)

            for j in range(len(self.encoder.weights)):
                self.encoder.weights[j] -= self.encoder.alpha * grad_encoder[j]

            for j in range(len(self.decoder.weights)):
                self.decoder.weights[j] -= self.decoder.alpha * grad_decoder[j]


    def generate(self, z = None):
        if not np.any(z):
            z = np.random.normal(0, 1, size=(1, self.latent_dim))
        return self.decoder.predict(z)

    def encode_decode(self, X):
        return self._forwardstep(X)