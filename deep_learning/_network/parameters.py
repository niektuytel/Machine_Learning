import numpy as np
import sys, os

# activation functions
# sys.path.insert(1, os.getcwd() + "/../../../") 
from algorithms.activation_functions import *

"""
Type of parameters that can contain in a RNN Neuron.
The most common structure of parameters are:
- LSTM 
- GRU 

Resources:
----------
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
"""

class ForgetGate:
    """
    Decide what information we’re going to throw away from the cell state.  
    This decision is made by a sigmoid layer called the “forget gate layer.”  
    It looks at ht−1 and xt, and outputs a number between 0 and 1 for each number in the cell state Ct−1.  
    A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”

    Parameters:
    -----------
    weights: list
        weight property is calculated with his inputs,
    bias: list
        biases property add/reduce calculated weight output
    """
    def __init__(self, weights, bias):
        self.sigmoid = Sigmoid()

        self.W = weights
        self.W_gradient = np.zeros_like(self.W)
        self.W_opt      = np.zeros_like(self.W)
        self.W_opt2     = np.zeros_like(self.W)
        
        self.b = bias
        self.b_gradient = np.zeros_like(self.b)
        self.b_opt      = np.zeros_like(self.b)
        self.b_opt2     = np.zeros_like(self.b)
    
    def forward(self, x, h_prev):
        """
        fᵗ = σ(Wᶠ[xᵗ, hᵗ⁻¹] + bᶠ)
        
        # stacking x(present input xt) and h(t-1)
        xᶜ = [xᵗ, hᵗ⁻¹]

        # dot product of Wf(forget weight matrix and xc + forget bias)
        fᵗ = σ(Wᶠxᶜ + bᶠ)
        """
        
        # xᶜ = [xᵗ, hᵗ⁻¹]
        xc = np.row_stack((h_prev, x))

        # fᵗ = σ(Wᶠxᶜ + bᶠ)
        f = self.sigmoid(np.dot(self.W, xc) + self.b)
        return f
        
    def derivative(self, c_prev, dc, f, xc):
        """
        xᶜ = [xᵗ, hᵗ⁻¹]
        fᵗ = σ(Wᶠxᶜ + bᶠ)

        # this formula:
        f' = c' * cᵗ⁻¹
        b' = b' + f' * fᵗ * (1-fᵗ)
        w' = w' + b' • xᶜ
        """
        df = dc * c_prev
        alpha = df * f * (1-f)

        self.b_gradient += alpha
        self.W_gradient += np.dot(alpha, xc.T)

class InputGate:
    """
    Decide what new information we’re going to store in the cell state. 
    This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. 
    Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state. 
    In the next step, we’ll combine these two to create an update to the state.

    Parameters:
    -----------
    weights: list
        weight property is calculated with his inputs,
    bias: list
        biases property add/reduce calculated weight output
    """
    def __init__(self, weights, bias):
        self.sigmoid = Sigmoid()

        self.W = weights
        self.W_gradient = np.zeros_like(self.W)
        self.W_opt      = np.zeros_like(self.W)
        self.W_opt2     = np.zeros_like(self.W)
        
        self.b = bias
        self.b_gradient = np.zeros_like(self.b)
        self.b_opt      = np.zeros_like(self.b)
        self.b_opt2     = np.zeros_like(self.b)

    def forward(self, x, h_prev, cell_state):
        """
        iᵗ = σ(Wᶦ[xᵗ, hᵗ⁻¹] + bᶦ)
        Ĉᵗ = tanh(Wᶜ[xᵗ, hᵗ⁻¹] + bᶜ)
        
        # stacking x(present input xt) and h(t-1)
        xᶜ = [xᵗ, hᵗ⁻¹]
        
        # dot product of Wi(input weight matrix and xc + input bias)
        iᵗ = σ(Wᶦxᶜ + bᶦ)

        # tanh of the dot product of Wc(cell state weight matrix and xc + cell state bias)
        Ĉᵗ = tanh(Wᶜxᶜ + bᶜ)
        """

        # xᶜ = [xᵗ, hᵗ⁻¹]
        xc = np.row_stack((h_prev, x))

        # iᵗ = σ(Wᶦxᶜ + b)
        i = self.sigmoid(np.dot(self.W, xc) + self.b)

        # Ĉᵗ = tanh(Wᶜxᶜ + bᶜ)
        C_hat = np.tanh(np.dot(cell_state.W, xc) + cell_state.b)

        return (C_hat, i)

    def derivative(self, c_hat, dc, i, xc):
        di = dc * c_hat

        alpha = di * i * (1-i) 
        self.b_gradient += alpha
        self.W_gradient += np.dot(alpha, xc.T)

class CellState:
    """
    Update the old cell state, Ct−1, into the new cell state Ct. 
    The previous steps already decided what to do, we just need to actually do it.
    We multiply the old state by ft, forgetting the things we decided to forget earlier. 
    Then we add it ∗ Chatᵗ. 
    This is the new candidate values, scaled by how much we decided to update each state value.

    Parameters:
    -----------
    weights: list
        weight property is calculated with his inputs,
    bias: list
        biases property add/reduce calculated weight output
    """
    def __init__(self, weights, bias):
        self.sigmoid = Sigmoid()

        self.W = weights
        self.W_gradient = np.zeros_like(self.W)
        self.W_opt      = np.zeros_like(self.W)
        self.W_opt2     = np.zeros_like(self.W)
        
        self.b = bias
        self.b_gradient = np.zeros_like(self.b)
        self.b_opt      = np.zeros_like(self.b)
        self.b_opt2     = np.zeros_like(self.b)

    def forward(self, f, c_prev, i, c_hat):
        return f * c_prev + i * c_hat

    def derivative(self, i, o, c, dh, dc_next, c_hat, xc):      
        dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next
        dc_hat = dc * i

        alpha = dc_hat * (1 - c_hat ** 2)
        self.b_gradient += alpha
        self.W_gradient += np.dot(alpha, xc.T)
        return dc

# etc. etc. etc. 

# class RememberGate:
#     def __call__(self):
#         ""

class OutputGate:
    """
    We need to decide what we’re going to output. 
    This output will be based on our cell state, but will be a filtered version. 
    First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. 
    Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply 
    it by the output of the sigmoid gate, so that we only output the parts we decided to.

    Parameters:
    -----------
    weights: list
        weight property is calculated with his inputs,
    bias: list
        biases property add/reduce calculated weight output
    """
    def __init__(self, weights, bias):
        self.sigmoid = Sigmoid()

        self.W = weights
        self.W_gradient = np.zeros_like(self.W)
        self.W_opt      = np.zeros_like(self.W)
        self.W_opt2     = np.zeros_like(self.W)
        
        self.b = bias
        self.b_gradient = np.zeros_like(self.b)
        self.b_opt      = np.zeros_like(self.b)
        self.b_opt2     = np.zeros_like(self.b)

    def forward(self, x, C, h_prev):
        """
            oᵗ = σ(Wᵒ[xᵗ, hᵗ⁻¹] + bᵒ)
            hᵗ = oᵗ * tanh(Cᵗ)
            return (hᵗ, oᵗ)
        """
        # stacking x(present input xt) and h(t-1)
        # xᶜ = [xᵗ, hᵗ⁻¹]
        # xc = np.hstack((x, h_prev))
        xc = np.row_stack((h_prev, x))


        # dot product of Wo(output weight matrix and xc + output bias)
        # oᵗ = σ(wᵒxᶜ + bᵒ)
        o = self.sigmoid(np.dot(self.W, xc) + self.b)

        # hᵗ = oᵗ * tanh(Cᵗ)
        h = o * np.tanh(C)
        return (h, o)

    def derivative(self, Wv_T, dv, dh_next, c, o, xc):        
        dh = np.dot(Wv_T, dv) + dh_next
        do = dh * np.tanh(c)
        
        alpha = do * o * (1-o)
        self.b_gradient += alpha
        self.W_gradient += np.dot(alpha, xc.T)
        return dh
