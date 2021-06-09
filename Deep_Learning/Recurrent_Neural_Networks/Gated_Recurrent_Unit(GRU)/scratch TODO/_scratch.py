# https://github.com/pangolulu/rnn-from-scratch
import numpy as np
import matplotlib.pyplot as plt
import sys, os, keras

from network.GRU import GRU
sys.path.insert(1, os.getcwd() + "/../") 
import data

class Model:
    def __init__(self, seq_length, seq_step, chars, char2idx, idx2char, n_neurons=100):
        """
        Implementation of simple character-level LSTM using Numpy
        """
        self.seq_length = seq_length # no. of time steps, also size of mini batch
        self.seq_step   = seq_step   # no. size of each time step
        self.vocab_size = len(chars) # no. of unique characters in the training data
        self.char2idx   = char2idx   # characters to indices mapping
        self.idx2char   = idx2char   # indices to characters mapping
        self.n_neurons  = n_neurons  # no. of units in the hidden layer
        
        self.unit = GRU(self.n_neurons, self.vocab_size)
        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length

    def sample(self, h_prev, c_prev, sample_size):
        """
        Outputs a sample sequence from the model
        """
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        sample_string = ""

        for t in range(sample_size):
            y_hat, _, h, _, c, _, _, _, _ = self.unit.forward(x, h, c)

            # get a random index within the probability distribution of y_hat(ravel())
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # find the char with the sampled index and concat to the output string
            char = self.idx2char[idx]
            sample_string += char
        return sample_string

    def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        """
        Implements the forward and backward propagation for one batch
        """
        x, z = {}, {}
        f, i, c_bar, c, o = {}, {}, {}, {}, {}
        y_hat, v, h = {}, {}, {}

        # Values at t= - 1
        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0
        for t in range(self.seq_length):
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][x_batch[t]] = 1

            y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = self.unit.forward(x[t], h[t - 1], c[t - 1])
            loss += -np.log(y_hat[t][y_batch[t], 0])

        self.unit.reset_gradients(0)
        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_length)):
            dh_next, dc_next = self.unit.backward(y_batch[t], y_hat[t], dh_next, dc_next, c[t - 1], z[t], f[t], i[t], c_bar[t], c[t], o[t], h[t])

        return loss, h[self.seq_length - 1], c[self.seq_length - 1]

    def train(self, X, y, epochs=10, learning_rate=0.01, beta1=0.9, beta2=0.999, verbose=True):
        """
        Main method of the LSTM class where training takes place
        """
        losses = [] # return history losses

        for epoch in range(epochs):
            h_prev = np.zeros((self.n_neurons, 1))
            c_prev = np.zeros((self.n_neurons, 1))

            for i in range(len(X)):
                x_batch = X[i] 
                y_batch = np.concatenate([ x_batch[1:], [y[i]] ])
                
                # Forward Pass
                loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

                # smooth out loss and store in list
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

                # keep loss history
                losses.append(self.smooth_loss)

                # overflowding protection
                # self.unit.limit_gradients(5)

                batch_num = epoch * epochs + i / self.seq_length + 1
                self.unit.optimization(batch_num, learning_rate, beta1, beta2)

                # print out loss and sample string
                if verbose:
                    if i % 100 == 0:
                        prediction = self.sample(h_prev, c_prev, sample_size=250)

                        print("-" * 100)
                        print(f"Epoch:[{epoch}] Loss:{round(self.smooth_loss[0], 2)} Index:[{i}/{len(X)}]")
                        print("-" * 88 + " prediction:")
                        print(prediction + "\n")

        return losses



if __name__ == "__main__":
    """
    Implementation of simple character-level LSTM using Numpy
    """
    # get data
    x, y = data.vectorization()
    print(f'data has {len(data.text)} characters, {data.chars} are unique')

    # define model
    model = Model(data.seq_length, data.sequences_step, data.chars, data.char2idx, data.idx2char)

    # train model
    losses = model.train(x, y)

    # display history losses
    plt.plot([i for i in range(len(losses))], losses)
    plt.xlabel("#training iterations")
    plt.ylabel("training loss")
    plt.show()

# Print:
# ----------------------------------------------------------------------------------------------------
# Epoch:[0] Loss:461.54 Index:[0/14266]
# ---------------------------------------------------------------------------------------- prediction:
# vEкmS‘0.NJUTбђгyShsјOVTгф.uгTrAs?Cv(aKrXjCvfтN(јaђsowpCyRAT?вuCTrir5E2FђгH~FTSiveSLCN4CfтAycfTI31~gX9”%AzјnypуpбHTв4ELt"tSOу””rNZгf?CZhDqt

# ......

# ----------------------------------------------------------------------------------------------------
# Epoch:[4] Loss:279.69 Index:[5700/14266]
# ---------------------------------------------------------------------------------------- prediction:
# a’no’r no ctu eunst. I tox gsga lrrntt s fy I’s oorsy, tegoe tam o euesools wr’mo ieoicos, bg auo I, slsm osooinnuetr’t guutom o m souuoand htth ii g uy o imoo goty’goo i tavrcamnuet y”ott tig uog uxctiuo sd ouru, taan H tQub teivwel “geeEuorss te “r

# but, it is a little to slow :)



