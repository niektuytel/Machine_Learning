""" 
(ANN) artificail neural network

play with it, code comes from and has as well a good explanation: 
https://cs231n.github.io/neural-networks-case-study/ 

"""
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.insert(1, os.getcwd() + "/../../_EXTRA/") 
import plot_data

N = 100 # number of points per class 
D = 2   # dimensionality
K = 3   # number of classes
X = np.zeros((N * K, D))# data matrix (each row = single example)
y = np.zeros( N * K, dtype="uint8")# class labels
for j in range(K):
    ix = range(N * j, N*(j+1))
    r = np.linspace(0.0, 1, N)# radius
    t = np.linspace(j*4, (j+1) * 4, N) + np.random.randn(N) * 0.2 #theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)] 
    y[ix] = j

# initialize parameters randomly
h = 100# size of hidden layer
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

# come hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(5000):
    # evaluate class scores with a 2-layer Neural Network
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W) + 0.5 * reg * np.sum(W2*W2)
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropagate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)

    # backprop the  ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # regularization gradient
    dW2 += reg * W2
    dW  += reg * W

    # perform a parameter update 
    W  += -step_size * dW
    b  += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# predict result of network
def predict(inputs):
    ############# layer #############
    hidden_sum = np.dot(inputs, W) + b
    
    # ReLU activation
    hidden_layer = np.maximum(0, hidden_sum)
    #################################
    
    ############# layer #############
    wsum = np.dot(hidden_layer, W2) + b2
    
    # argmax activation
    output = np.argmax(wsum, axis=1)
    #################################
    return output

# visualize predictions
plot_data.visualize(
    X[:, 0], 
    X[:, 1], 
    y, predict
)

predicted_class = predict(X)
print('training accuracy: %.2f' % (np.mean(predicted_class  == y)))
