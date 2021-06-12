import numpy as np

def squared_error(y, yhat):
    return np.sum(0.5 * (y - yhat)**2), y - yhat

def identity(y, yhat):
    return yhat, yhat

loss_table = {
    'squared_error': squared_error,
    'identity': identity
}