# https://miro.medium.com/max/650/1*EgQzN0yoqFZVLMIodlaR7A.png
import numpy as np
from .parameters import *

# did not use the global one as that one contains
# extra features that gives issue to this neuron
class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def derivative(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class LSTM:
    """
    The structure of RNN is very similar to hidden Markov model. 
    However, the main difference is with how parameters are calculated and constructed.
    As LSTM Neuron has his unique parameter structure

    Resources:
    ----------
        https://christinakouridi.blog/2019/06/20/vanilla-lstm-numpy/
        https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
        https://medium.com/coinmonks/character-to-character-rnn-with-pytorchs-lstmcell-cd923a6d0e72

    Parameter:
    ----------
    n_inputs: int
        number of inputs
    vocab_size: int
        the length of the vocabulary set. 
        We need this size to define the input size to the LSTM cell.
    """
    def __init__(self, n_inputs, vocab_size, normalizer=Softmax):
        self.n_inputs= n_inputs
        self.vocab_size = vocab_size
        self.normalizer = normalizer()

        # output
        std_v = (1.0 / np.sqrt(self.vocab_size))
        self.W = np.random.randn(self.vocab_size, self.n_inputs) * std_v
        self.W_gradient  = np.zeros_like(self.W)
        self.W_opt       = np.zeros_like(self.W)
        self.W_opt2      = np.zeros_like(self.W)

        self.b = np.zeros((self.vocab_size, 1))
        self.b_gradient  = np.zeros_like(self.b)
        self.b_opt       = np.zeros_like(self.b)
        self.b_opt2      = np.zeros_like(self.b)

        # random parameter values
        std = (1.0 / np.sqrt(self.vocab_size + self.n_inputs))
        W = np.random.randn(self.n_inputs, self.n_inputs + self.vocab_size) * std
        b = np.ones((self.n_inputs, 1))

        # parameters
        self.state  = CellState(W, b)
        self.forget = ForgetGate(W, b)
        self.input  = InputGate(W, b)
        self.output = OutputGate(W, b)
        self.unit_parts = [self.forget, self.input, self.state, self.output]

    def reset_gradients(self, value):
        """
        Resets gradients to zero before each backpropagation
        """
        for unit_part in self.unit_parts:
            unit_part.W_gradient.fill(value)
            unit_part.b_gradient.fill(value)
        
        # main unit
        self.W_gradient.fill(value)
        self.b_gradient.fill(value)
    
    def limit_gradients(self, limit):
        """
        Limits the magnitude of gradients to avoid exploding gradients
        """
        for unit_part in self.unit_parts:
            np.clip(unit_part.W_gradient, -limit, limit, out=unit_part.W_gradient)
            np.clip(unit_part.b_gradient, -limit, limit, out=unit_part.b_gradient)
        
        # main unit
        np.clip(self.W_gradient, -limit, limit, out=self.W_gradient)
        np.clip(self.b_gradient, -limit, limit, out=self.b_gradient)

    def optimization(self, batch_num, learning_rate, beta1, beta2):
        self.batch_num = batch_num
        self.learning_rate = learning_rate
        self.beta_1 = beta1
        self.beta_2 = beta2
        
        for unit_part in self.unit_parts:
            self._optimization(unit_part)

        # main unit
        self._optimization(self)

    def _optimization(self, obj):
        "Adam optimization used to update to a optimized result"
        # optimization Weights
        obj.W_opt  = obj.W_opt  * self.beta_1 + (1 - self.beta_1) * obj.W_gradient
        obj.W_opt2 = obj.W_opt2 * self.beta_2 + (1 - self.beta_2) * obj.W_gradient ** 2

        m_correlated = obj.W_opt  / (1 - self.beta_1 ** self.batch_num)
        v_correlated = obj.W_opt2 / (1 - self.beta_2 ** self.batch_num)
        obj.W -= self.learning_rate * m_correlated / (np.sqrt(v_correlated) + 1e-8)

        # optimization Biases
        obj.b_opt  = obj.b_opt  * self.beta_1 + (1 - self.beta_1) * obj.b_gradient
        obj.b_opt2 = obj.b_opt2 * self.beta_2 + (1 - self.beta_2) * obj.b_gradient ** 2

        m_correlated = obj.b_opt  / (1 - self.beta_1 ** self.batch_num)
        v_correlated = obj.b_opt2 / (1 - self.beta_2 ** self.batch_num)
        obj.b -= self.learning_rate * m_correlated / (np.sqrt(v_correlated) + 1e-8)

    def forward(self, x, h_prev, c_prev):
        # forget gate result
        f = self.forget.forward(x, h_prev)

        # input gate result + C'
        c_hat, i = self.input.forward(x, h_prev, self.state)

        # cell state result
        c = self.state.forward(f, c_prev, i, c_hat)

        # output gate result + h result
        h, o = self.output.forward(x, c, h_prev)

        # cell output
        xc = np.row_stack((h_prev, x))
        v = np.dot(self.W, h) + self.b

        y_hat = self.normalizer(v)
        return y_hat, v, h, o, c, c_hat, i, f, xc

    def backward(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
        dv = np.copy(y_hat)
        dv[y] -= 1 # yhat - y
        self.b_gradient += dv
        self.W_gradient += np.dot(dv, h.T)

        # gate output
        dh = self.output.derivative(self.W.T, dv, dh_next, c, o, z)

        # state cell
        dc = self.state.derivative(i, o, c, dh, dc_next, c_bar, z)

        # gate input
        self.input.derivative(c_bar, dc, i, z)

        # gate forget
        self.forget.derivative(c_prev, dc, f, z)

        dz = (
            np.dot(self.forget.W.T, self.forget.b_gradient) + 
            np.dot( self.input.W.T,  self.input.b_gradient) + 
            np.dot( self.state.W.T,  self.state.b_gradient) + 
            np.dot(self.output.W.T, self.output.b_gradient)
        )

        dh_prev = dz[:self.n_inputs, :]
        dc_prev = f * dc
        return dh_prev, dc_prev


