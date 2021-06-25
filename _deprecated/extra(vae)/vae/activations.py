import numpy as np

def relu(x):
    f  = np.maximum(0, x)
    df = np.where(x > 0, 1, 0)
    return f, df

def sigmoid(x):
    f  = 1 / (1 + np.exp(-x))
    df = f * (1 - f)
    return f, df

def tanh(x):
    f = -1 + 2 / (1 + np.exp(-2 * x))
    df = 1 - f**2
    return f, df

def identity(x):
    f  = x
    df = 1
    return f, df

def selu(x):
    alpha = 1.67326
    lambd = 1.05070
    f  = lambd * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    df = lambd * np.where(x >= 0, 1, alpha * np.exp(x))
    return f, df

activation_table = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'identity': identity,
    'selu': selu 
}