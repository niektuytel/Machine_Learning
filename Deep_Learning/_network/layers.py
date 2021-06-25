# sample code update Network to his own file (import)
import numpy as np
from copy import copy
from terminaltables import AsciiTable

# functions
from algorithms.activation_functions import act_functions 
from algorithms.loss_functions import loss_functions 
from algorithms.optimizer_functions import opt_functions
from parameters import *


# sys.path.insert(1, "D:\Programming\learn\AI\sample\ML-From-Scratch") 
# from mlfromscratch.deep_learning.optimizers import Adam
# from mlfromscratch.deep_learning.loss_functions import CrossEntropy, SquareLoss
# from mlfromscratch.deep_learning import NeuralNetwork

# import or in Layer class
def default_weights(shape, static, limit=1):
    n_features = shape[0]
    n_units  = shape[1]
    n_total  = n_features * n_units

    if static:
        # linear spaced out:
        #     The first param is 0 & the last param is 1,
        #     all params in between are lineared spaced out.
        #     makes it more readable & more sparsed than randomized
        weights = np.zeros(shape)
        lin_spaced = np.linspace(-limit, limit, n_total)

        for i in range(n_features):
            start = i * n_units
            end = start + n_units
            weights[i] = lin_spaced[start:end]

        # [[0.  0.2] [0.4 0.6] [0.8 1. ]]
        return weights
    else:
        # random
        limit = 1 / np.sqrt(n_features)
        weights = np.random.uniform(-limit, limit, shape)

        # [[0.50466882 0.58159532] [0.77758233 0.78466492] [0.44969037 0.5287821 ]]
        return weights

class Cell():
    def __init__(self, optimizer):
        self.values = []

        # weights
        self.W = None
        self.W_gradient = None
        self.W_optimizer = copy(optimizer)

        # bias
        self.b = None
        self.b_gradient = None
        self.b_optimizer = copy(optimizer)

    def initialize(self, n_inputs, n_outputs):
        limit = 1 / np.sqrt(n_inputs)
        size = (n_inputs, n_outputs)
        
        self.W = np.random.uniform(-limit, limit, size=size)
        self.b = np.zeros((1, n_outputs))
        self.clear_gradients()

    def clear_gradients(self):
        self.W_gradient = np.zeros_like(self.W)
        self.b_gradient = np.zeros_like(self.b)

    def update_weights(self):
        if self.W_optimizer != None:
            self.W = self.W_optimizer.update(self.W, self.W_gradient)

        if self.b_optimizer != None:
            self.b = self.b_optimizer.update(self.b, self.b_gradient)

class Layer_V2():
    def __init__(self, n_units=None, input_shape=None, activation_name=None, optimizer_name=None):
        self.n_units       = n_units
        self.inputs_shape  = input_shape
        self.outputs_shape = input_shape# (n_units,)
        self.layer_input   = None
        self.layer_output  = None
        self.activation    = None
        self.optimizer     = None

        if activation_name != None:
            self.activation = act_functions[activation_name]()
        
        if optimizer_name != None:
            self.optimizer = opt_functions[optimizer_name]()

    def forward(self, X):
        self.layer_output = X
        if self.activation != None:
            return self.activation(X)
        else: 
            return X
  
    def backward(self, gradient):
        if self.activation != None:
            return gradient * self.activation.gradient(self.layer_output)
        else:
            return gradient

    def input_shape(self, new_value=None):
        if new_value != None:
            self.inputs_shape = new_value

        return self.inputs_shape

    def output_shape(self, new_value=None):
        if new_value != None:
            self.outputs_shape = new_value
            
        return self.outputs_shape

    def parameters(self): 
        return 0

    def activation_name(self):
        if self.activation is None:
            return ""
        else:
            return self.activation.__class__.__name__

    # usable on: https://github.com/eriklindernoren/ML-From-Scratch
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = self.outputs_shape = shape

####################################
##   Layers that EDIT data flow   ##
####################################
class Dense_V2(Layer_V2):
    """
    Dense_V2(n_units=512, input_shape=(784,), activation="relu", optimizer="adam")
    
    # optimizer default: "adam"
    Dense_V2(n_units=512, input_shape=(784,), activation="relu")
    
    # activation default: None
    Dense_V2(n_units=512, input_shape=(784,))

    # only on: network.add(...)
    Dense_V2(n_units=512)
    """
    def __init__(self, n_units, input_shape, activation=None, optimizer="adam"):
        super().__init__(n_units, input_shape, activation, optimizer)

        self.cell = Cell(self.optimizer)
        self.cell.initialize(input_shape[0], n_units)

    def forward(self, X):
        self.layer_input = X
        output = X.dot(self.cell.W) + self.cell.b
        return super().forward(output)
        
    def backward(self, gradient):
        gradient = super().backward(gradient)
        W = self.cell.W

        # Calculate gradient w.r.t layer weights
        self.cell.clear_gradients()
        self.cell.W_gradient = self.layer_input.T.dot(gradient)
        self.cell.b_gradient = np.sum(gradient, axis=0, keepdims=True)
        self.cell.update_weights()
        
        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        return gradient.dot(W.T)

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return np.prod(self.cell.W.shape) + np.prod(self.cell.b.shape)

    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape

class Conv2D(Layer_V2):
    def __init__(self, n_units, filter_shape, input_shape, stride=1, padding="same", activation=None, optimizer="adam"):
        super().__init__(n_units, input_shape, activation, optimizer)
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding

        self.cell = Cell(self.optimizer)
        self.initialize(self.optimizer)
        
    def initialize(self, optimizer):
        filter_height, filter_width = self.filter_shape
        limit = 1 / np.sqrt(np.prod(self.filter_shape))
        channels = self.inputs_shape[0]
        size = (self.n_units, channels, filter_height, filter_width)

        self.cell.W  = np.random.uniform(-limit, limit, size=size)
        self.cell.b = np.zeros((self.n_units, 1))
        self.cell.clear_gradients()

    def forward(self, X):
        batch_size, channels, height, width = X.shape
        self.layer_input = X

        # Turn image shape into column shape (enables dot product between input and weights)
        self.X_col = self.image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)

        # Turn weight into column shape
        self.W_col = self.cell.W.reshape((self.n_units, -1))

        # calculate output
        output = self.W_col.dot(self.X_col) + self.cell.b
        output = output.reshape(self.output_shape() + (batch_size, )).transpose(3, 0, 1, 2)
        return super().forward(output)
        
    def backward(self, gradient):
        gradient = super().backward(gradient)

        # Reshape accumulated gradient into column shape
        gradient = gradient.transpose(1, 2, 3, 0).reshape(self.n_units, -1)
        
        # Calculate gradient w.r.t layer weights
        self.cell.clear_gradients()
        self.cell.W_gradient = gradient.dot(self.X_col.T).reshape(self.cell.W.shape)
        self.cell.b_gradient = np.sum(gradient, axis=1, keepdims=True)
        self.cell.update_weights()

        # Recalculate the gradient
        gradient = self.W_col.T.dot(gradient)

        # Reshape from column shape to image shape
        gradient = self.column_to_image(gradient, self.layer_input.shape, self.filter_shape, stride=self.stride, output_shape=self.padding)

        return gradient

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return np.prod(self.cell.W.shape) + np.prod(self.cell.b.shape)
    
    def output_shape(self):
        channels, height, width = self.inputs_shape
        pad_h, pad_w = self.determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_units, int(output_height), int(output_width)


    # Method which turns the image shaped input to column shape.
    # Used during the forward pass.
    # Reference: CS231n Stanford
    def image_to_column(self, images, filter_shape, stride, output_shape='same'):
        filter_height, filter_width = filter_shape

        pad_h, pad_w = self.determine_padding(filter_shape, output_shape)

        # Add padding to the image
        images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

        # Calculate the indices where the dot products are to be applied between weights
        # and the image
        k, i, j =self.get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

        # Get content from image at those indices
        cols = images_padded[:, k, i, j]
        channels = images.shape[1]
        # Reshape content into column shape
        cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
        return cols
        
    # Method which turns the column shaped input to image shape.
    # Used during the backward pass.
    # Reference: CS231n Stanford
    def column_to_image(self, cols, images_shape, filter_shape, stride, output_shape='same'):
        batch_size, channels, height, width = images_shape
        pad_h, pad_w = self.determine_padding(filter_shape, output_shape)
        height_padded = height + np.sum(pad_h)
        width_padded = width + np.sum(pad_w)
        images_padded = np.zeros((batch_size, channels, height_padded, width_padded))

        # Calculate the indices where the dot products are applied between weights
        # and the image
        k, i, j = self.get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

        cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
        cols = cols.transpose(2, 0, 1)
        # Add column content to the images at the indices
        np.add.at(images_padded, (slice(None), k, i, j), cols)

        # Return image without padding
        return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]

    # Method which calculates the padding based on the specified output shape and the
    # shape of the filters
    def determine_padding(self, filter_shape, output_shape="same"):

        # No padding
        if output_shape == "valid":
            return (0, 0), (0, 0)
        # Pad so that the output shape is the same as input shape (given that stride=1)
        elif output_shape == "same":
            filter_height, filter_width = filter_shape

            # Derived from:
            # output_height = (height + pad_h - filter_height) / stride + 1
            # In this case output_height = height and stride = 1. This gives the
            # expression for the padding below.
            pad_h1 = int(np.floor((filter_height - 1)/2))
            pad_h2 = int(np.ceil((filter_height - 1)/2))
            pad_w1 = int(np.floor((filter_width - 1)/2))
            pad_w2 = int(np.ceil((filter_width - 1)/2))

            return (pad_h1, pad_h2), (pad_w1, pad_w2)

    # Reference: CS231n Stanford
    def get_im2col_indices(self, images_shape, filter_shape, padding, stride=1):
        # First figure out what the size of the output should be
        batch_size, channels, height, width = images_shape
        filter_height, filter_width = filter_shape
        pad_h, pad_w = padding
        out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
        out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

        i0 = np.repeat(np.arange(filter_height), filter_width)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(filter_width), filter_height * channels)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

        return (k, i, j)



    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape

class Activation(Layer_V2):
    def __init__(self, activation_name, n_units=None, input_shape=None, optimizer_name=None):
        super().__init__(n_units, input_shape, activation_name, optimizer_name)

    def layer_name(self): 
        return self.__class__.__name__



####################################
## Layers that MAINTAIN data flow ##
####################################
# class Input(Layer_V2):

# class Output(Layer_V2):

class Reshape(Layer_V2):
    def __init__(self, output_shape, input_shape):
        super().__init__()
        self.inputs_shape = input_shape
        self.outputs_shape = output_shape

    def forward(self, X):
        self.layer_input = X
        return X.reshape((X.shape[0], ) + self.outputs_shape)
        
    def backward(self, gradient):
        return gradient.reshape(self.layer_input.shape)

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return 0

    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape

class Flatten(Layer_V2):
    def __init__(self, input_shape):
        super().__init__()
        self.inputs_shape = input_shape
        self.outputs_shape = (np.prod(input_shape),)

    def forward(self, X):
        self.layer_input = X
        return X.reshape((X.shape[0], -1))
        
    def backward(self, gradient):
        return gradient.reshape(self.layer_input.shape)

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return 0

    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape

class Dropout(Layer_V2):
    def __init__(self, lowest_value, input_shape):
        super().__init__()
        self.lowest_value = lowest_value
        self.inputs_shape = input_shape
        self.outputs_shape = self.inputs_shape

    def forward(self, X):
        self.layer_input = np.random.uniform(size=X.shape) > self.lowest_value
        return X * self.layer_input

    def backward(self, gradient):
        return gradient * self.layer_input

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return 0
        
    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape

class BatchNormalization(Layer_V2):
    def __init__(self, momentum=0.99, input_shape=None, optimizer="adam"):
        super().__init__()
        self.momentum = momentum
        self.inputs_shape = self.outputs_shape = input_shape
        self.optimizer = opt_functions[optimizer]()
        
        self.epsilon = 0.01
        self.running_mean = None
        self.running_var = None

        # initialize parameters
        self.gamma = np.ones(self.inputs_shape[0])
        self.beta = np.zeros(self.inputs_shape[0])
        
        # parameter optimizers
        self.gamma_optimizer = copy(self.optimizer)
        self.beta_optimizer  = copy(self.optimizer)

    def forward(self, X):
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        # initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = mean
            self.running_var = var

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        # statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.epsilon)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta
        return output
        
    def backward(self, gradient):
        # save parameters used during the forward pass
        gamma = self.gamma

        # layer update
        X_norm = self.X_centered * self.stddev_inv
        gradient_gamma = np.sum(gradient * X_norm, axis=0)
        gradient_beta = np.sum(gradient, axis=0)

        self.gamma = self.gamma_optimizer.update(self.gamma, gradient_gamma)
        self.beta = self.beta_optimizer.update(self.beta, gradient_beta)

        # The gradient of the loss with the respect to the layer inputs 
        # (use weights and statistics from forward pass)
        batch_size = gradient.shape[0]
        gradient = (1 / batch_size) * gamma * self.stddev_inv * (batch_size * gradient - np.sum(gradient, axis=0) - self.X_centered * self.stddev_inv ** 2) * np.sum(gradient * self.X_centered, axis=0)
        return gradient

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self): 
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    # bypass 
    def backward_pass(self, gradient): return self.backward(gradient)
    def forward_pass(self, X, training=True): return self.forward(X)
    def set_input_shape(self, shape): self.inputs_shape = shape



####################################
##            Network             ##
####################################
class Network_V2():
    def __init__(self, loss_name="MSE"):
        self.layers = []
        self.loss_function = loss_functions[loss_name]()

    def add(self, layer):
        self.layers.append(layer)

    def remove(self, layer_index):
        del self.layers[layer_index]
        return self.layers
    
    def test_on_batch(self, X, y):
        y_pred = self._forward(X)

        loss = np.mean(self.loss_function(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        y_pred = self._forward(X)

        loss = np.mean(self.loss_function(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        loss_grad = self._backward(loss_grad)

        return loss, acc, loss_grad

    def _backward(self, gradient_loss, y_pred=None):
        # update weights in each layer, bind the output to next input gradient loss
        for layer in reversed(self.layers):
            gradient_loss = layer.backward_pass(gradient_loss)

        return gradient_loss
            
    def _forward(self, X):
        output = X

        # walk through layers, bind the output to input
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def summary(self, name="Model Summary"):
        # Print model name
        print (AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print ("Input Shape: %s" % str(self.layers[0].input_shape()))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape", "Activation name"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = 000# layer.output_shape()
            activation_name = 000# layer.activation_name()
            table_data.append([layer_name, str(params), str(out_shape), str(activation_name)])
            tot_params += params
        # Print network configuration table
        print (AsciiTable(table_data).table)
        print ("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        return self._forward(X)

    def fit(self, X, y, n_epochs, batch_size):
        loss_history = []
        accuracy = 0.00

        for epoch in range(n_epochs):    
            loss_batch = []
            for X_batch, y_batch in self.batch_iterator(X, y, batch_size=batch_size):
                loss, accuracy, _ = self.train_on_batch(X_batch, y_batch)
                loss_batch.append(loss)

            loss = np.mean(loss_batch)
            loss_history.append(loss)
            print(f"\r[{epoch}/{n_epochs}] loss:{loss}", end="")

        print()
        return loss_history, accuracy

    def batch_iterator(self, X, y=None, batch_size=64):
        """ Simple batch generator """
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i+batch_size, n_samples)
            if y is not None:
                yield X[begin:end], y[begin:end]
            else:
                yield X[begin:end]
                

































                

    # def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
    #     history_loss = []

    #     for epoch in range(n_epochs):
    #         batch_error = []
    #         # for X_batch, y_batch in self._gen_batch(X, y, batch_size):
    #         y_pred = self._forward(X)

    #         # the loss gradient with respect to y_pred
    #         gradient_loss = self.loss_function.gradient(y, y_pred)

    #         # Backpropagate. Update weights
    #         self._backward(gradient_loss, y_pred)

    #         batch_error.append(
    #             np.mean(self.loss_function(y, y_pred))
    #         )
            
    #         history_loss.append(
    #             np.mean(batch_error)
    #         )

    #         # # display trained network state
    #         # # print("called")
    #         # if verbose:
    #         #     print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

    #     # # model accuracy
    #     # if verbose:
    #     #     y_true = np.argmax(y, axis=2)
    #     #     y_pred = np.argmax(self.predict(X), axis=2)
    #     #     accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
    #     #     print(f"\nAccuracy: {accuracy}%\n")

    #     return history_loss




class Layer(object):
    def __init__(self, n_units, input_shape, activation, optimizer):
        self.n_units      = n_units
        self.input_shape  = input_shape
        self.output_shape = (n_units,)
        
        self.layer_input   = None
        self.activation    = act_functions[activation]()
        self.optimizer     = opt_functions[optimizer]()

class Gate():
    # Single gate used in recurrent neural networks
    def __init__(self, optimizer, clip_limit=1):
        # timestep values
        self.clip_limit = clip_limit
        self.values = []

        # weights
        self.W = None
        self.W_gradient = None
        self.W_optimizer = copy(optimizer)

        # inputs
        self.U = None
        self.U_gradient = None
        self.U_optimizer = copy(optimizer)

        # bias
        self.b = None
        self.b_gradient = None
        self.b_optimizer = copy(optimizer)

    def initialize(self, shape):
        # initialize Weight
        limit = 1 / np.sqrt(shape[1])
        
        self.W = np.random.uniform(-limit, limit, shape)
        self.U = np.random.uniform(-limit, limit, (shape[0], shape[0]))
        self.b = np.zeros((shape[0], 1))

        self.W_gradient = np.zeros_like(self.W)
        self.U_gradient = np.zeros_like(self.U)
        self.b_gradient = np.zeros_like(self.b)

    def clear_gradients(self):
        self.W_gradient = np.zeros_like(self.W)
        self.U_gradient = np.zeros_like(self.U)
        self.b_gradient = np.zeros_like(self.b)

    def update_weights(self):
        # np.clip(self.W_gradient, -self.clip_limit, self.clip_limit, out=self.W_gradient)
        # np.clip(self.b_gradient, -self.clip_limit, self.clip_limit, out=self.b_gradient)

        self.W = self.W_optimizer.update(self.W, self.W_gradient)
        self.U = self.U_optimizer.update(self.U, self.U_gradient)
        self.b = self.b_optimizer.update(self.b, self.b_gradient)

class Dense(Layer):
    """
        ## Params:
        ```txt
            n_units: int
                Number of neurons in a layer\n
            input_shape: (int, int)
                Amount of input params on each neuron\n
            activation: string
                Function name used to set neuron inputs between(e.g. [0 - 1])\n
            optimizer: string
                Function name used to update params on back propergation\n
            static_weights: bool
                Set default static weights.
                Keep the out coming results the same.\n
        ```

        ## Usable:
        ```python
            Dense(n_units=3)
            Dense(n_units=3)(input_shape=(3,))
            Dense(n_units=3, input_shape=(3,), activation="relu", optimizer="adam", static_weights=True)
        ```

        ### Sample:
        ```python
        from layers import *
        X = np.array([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0,0,1],[0,1,1]]) # train data
        Y = np.array([  [0],    [0],    [0],    [1],    [1],    [1]  ]) # train targets
        
        # define model
        network = Network(loss="MSE")
        network.add(Dense(n_units=3, input_shape=(3,)))
        network.add(Dense(n_units=1))
        
        # train network
        history_loss = network.fit(X,Y,20000)
            
        # network result
        print(f"question:[0, 1, 1] --- predict:{network.predict(np.array([0, 1, 1]))} --- target:1")
        print(f"question:[1, 1, 0] --- predict:{network.predict(np.array([1, 1, 0]))} --- target:0")
        print(f"question:[1, 0, 1] --- predict:{network.predict(np.array([1, 0, 1]))}")
        ```
    """
    def __init__(self, n_units, input_shape=None, activation="relu", optimizer="adam", static_weights=True):
        super().__init__(n_units, input_shape, activation, optimizer)
        self.static_weights = static_weights

        # optimizers for layer params
        self.W_optimizer = copy(self.optimizer)
        self.b_optimizer = copy(self.optimizer)
        
        if self.input_shape != None:
            self.__call__(self.input_shape)

    def __call__(self, input_shape=None):
        # input shape from previous layer
        if self.input_shape is None:
            self.input_shape = input_shape

        # initialize weights
        shape = (self.input_shape[0], self.n_units)
        self.states_W = default_weights(shape, self.static_weights, 1)
        self.b = np.zeros((1, self.n_units))

    def forward(self, X):
        # need for backprop
        self.layer_input = X

        # pass trough neuron
        self.wsum = self.layer_input.dot(self.states_W) + self.b
        return self.activation(self.wsum)
        
    def backward(self, loss):
        # backward gradient (chain-rule)
        gradient = loss * self.activation.gradient(self.wsum)
        gradient_W = self.layer_input.T.dot(gradient)
        gradient_b = np.sum(gradient, axis=0, keepdims=True)

        # Update the layer params
        old_W  = self.states_W.T
        self.states_W = self.W_optimizer.update(self.states_W, gradient_W)
        self.b = self.b_optimizer.update(self.b, gradient_b)

        # Return gradient for next layer
        # Calculated based on the weights used during the forward pass
        loss = gradient.dot(old_W)
        return loss

class RNN(Layer):
    '''
        ## Params:
        ```txt
            n_units: int
                Number of neurons in a layer\n
            input_shape: (int, int)
                First value is the input size on each neuron,  
                the second value is the batch size of 1 input value.\n
            activation: string
                Function name used to interact on given inputs/states, inside the neuron\n
            activation_output: string
                Function name used to interact on given outputs, outputs in the neuron\n
            optimizer: string
                Function name used to update params on back propergation\n
            bptt_trunc: int
                Decides how many time steps the gradient should be propagated backwards  
                through states given the loss gradient for time step t.\n
        ```

        ## Usable:
        ```python
            RNN(n_units=10)
            RNN(n_units=10)(input_shape=(3,))
            RNN(n_units=10, input_shape=(3,50), activation="relu", activation_output="softmax", optimizer="adam", bptt_trunc=5)
        ```

        ## Sample:
        ```python
        import numpy as np
        from layers import *
        from sklearn.model_selection import train_test_split

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
        ```
    '''
    def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam", bptt_trunc=5):
        super().__init__(n_units, input_shape, activation, optimizer)
        self.activation_output = act_functions[activation_output]()
        self.bptt_trunc = bptt_trunc

        # input
        self.inputs = None
        self.input_W = None
        self.input_Wd = None
        self.input_W_optimizer = copy(self.optimizer)
        
        # states
        self.states = None
        self.states_W = None
        self.states_Wd = None
        self.states_W_optimizer = copy(self.optimizer)

        # output
        self.outputs = None
        self.output_W = None
        self.output_Wd = None
        self.output_W_optimizer = copy(self.optimizer)

        if self.input_shape != None:
            self.__call__(self.input_shape)

    def __call__(self, input_shape=None):
        if self.input_shape is None:
            self.input_shape = input_shape
    
        timesteps, input_dim = self.input_shape
        limit = 1 / np.sqrt(input_dim)

        # initialize input Weight
        shape_input  = (self.n_units, input_dim)
        self.input_W = np.random.uniform(-limit, limit, shape_input)
        
        limit = 1 / np.sqrt(self.n_units)

        # initialize states Weight
        shape_state  = (self.n_units, self.n_units)
        self.states_W = np.random.uniform(-limit, limit, shape_state)

        # initialize output Weight
        shape_output = (input_dim, self.n_units)
        self.output_W = np.random.uniform(-limit, limit, shape_output)

    def forward(self, X):
        # need for backprop
        self.layer_input = X

        batch_size, timesteps, input_dim = X.shape
        self.inputs = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))
        
        # pass trough neuron
        for t in range(timesteps):
            self.inputs[:, t] = X[:, t].dot(self.input_W.T) + self.states[:, t-1].dot(self.states_W.T)
            self.states[:, t] = self.activation(self.inputs[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.output_W.T)

        return self.activation_output(self.outputs)

    def backward(self, loss):
        # gradients
        self.input_Wd  = np.zeros_like(self.input_W)
        self.states_Wd = np.zeros_like(self.states_W)
        self.output_Wd = np.zeros_like(self.output_W)

        # backward gradient (chain-rule)
        gradient = loss * self.activation_output.gradient(self.outputs)
        gradient_next = np.zeros_like(gradient)

        # update layer weights/params
        _, timesteps, _ = gradient.shape
        for t in reversed(range(timesteps)):
            gradient_state = gradient[:, t].dot(self.output_W) * self.activation.gradient(self.inputs[:, t])
            gradient_next[:, t] = gradient_state.dot(self.input_W)

            self.output_Wd += gradient[:, t].T.dot(self.states[:, t])
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                self.input_Wd += gradient_state.T.dot(self.layer_input[:, t_])
                self.states_Wd += gradient_state.T.dot(self.states[:, t_-1])
                
                gradient_state = gradient_state.dot(self.states_W) * self.activation.gradient(self.inputs[:, t_-1])

        self.input_W  = self.input_W_optimizer.update(self.input_W, self.input_Wd)
        self.states_W = self.states_W_optimizer.update(self.states_W, self.states_Wd)
        self.output_W = self.output_W_optimizer.update(self.output_W, self.output_Wd)

        # Return gradient for next layer
        # Calculated based on the weights used during the forward pass
        return gradient_next

# TODO: fix Back Propergation
class LSTM(Layer):
    def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam"):
        super().__init__(n_units, input_shape, activation, optimizer)
        self.sigmoid = act_functions["sigmoid"]()
        self.activation_output = act_functions[activation_output]()

        self.forget = Gate(self.optimizer)
        self.input  = Gate(self.optimizer)
        self.cell   = Gate(self.optimizer)
        self.output = Gate(self.optimizer)
        self.state = Gate(self.optimizer) 

        if self.input_shape != None:
            self.__call__(self.input_shape)

    def __call__(self, input_shape=None):
        if self.input_shape is None:
            self.input_shape = input_shape
        
        timesteps, input_dim = self.input_shape
        shape_input  = (self.n_units, input_dim)
        shape_state  = (self.n_units, self.n_units)
        shape_output = (input_dim, self.n_units)
        
        # initialize gates
        self.forget.initialize(shape=shape_input) 
        self.input.initialize(shape=shape_input)  
        self.cell.initialize(shape=shape_input)   
        self.output.initialize(shape=shape_input) 
        self.state.initialize(shape=shape_input) 

    def forward(self, X):
        batch_size, timesteps, input_dim = X.shape
        self.layer_input = X

        self.forget.values = np.zeros((batch_size, timesteps, self.n_units))
        self.input.values = np.zeros((batch_size, timesteps, self.n_units))
        self.output.values = np.zeros((batch_size, timesteps, self.n_units))
        self.cell.values = np.zeros((batch_size, timesteps, self.n_units))
        self.state.values = np.zeros((batch_size, timesteps, self.n_units))

        outputs = np.zeros((batch_size, timesteps, input_dim))

        # forward pass
        for t in range(timesteps):
            Wf, Wi, Wo, Wc = self.forget.W, self.input.W, self.output.W, self.cell.W
            Uf, Ui, Uo, Uc = self.forget.U, self.input.U, self.output.U, self.cell.U
            xt = X[:, t]
            ct_prev = self.cell.values[:, t-1]

            ft = self.sigmoid(xt.dot(Wf.T) + ct_prev.dot(Uf))
            it = self.sigmoid(xt.dot(Wi.T) + ct_prev.dot(Ui))
            ot = self.sigmoid(xt.dot(Wo.T) + ct_prev.dot(Uo))
            ct_hat = self.activation(xt.dot(Wc.T) + ct_prev.dot(Uc))# g
            ct = ft * ct_prev + it * ct_hat
            st = ot * self.activation(ct)#(ht)/(at)

            self.forget.values[:, t] = ft
            self.input.values[:, t] = it
            self.output.values[:, t] = ot 
            self.cell.values[:, t] = ct
            self.state.values[:, t] = st#(ht)/(at)

            # next ht
            outputs[:, t] = self.state.values[:, t].dot(self.state.W)

        return outputs

    def backward(self, gradient):
        print("TODO: fix Back Propergation")
        gradient_next = np.zeros_like(gradient)

        # gradients
        self.forget.W_gradient = np.zeros_like(self.forget.W)
        self.input.W_gradient = np.zeros_like(self.input.W)
        self.output.W_gradient = np.zeros_like(self.output.W)
        self.cell.W_gradient = np.zeros_like(self.cell.W)
        self.state.W_gradient = np.zeros_like(self.state.W)

        # update layer weights/params
        _, timesteps, _ = gradient.shape
        for t in reversed(range(timesteps)):
            # TODO: fix Back Propergation
            self.forget.W_gradient += 0.00
            self.input.W_gradient  += 0.00
            self.output.W_gradient += 0.00
            self.cell.W_gradient   += 0.00
            self.state.W_gradient  += 0.00

        self.forget.update_weights()
        self.input.update_weights()
        self.output.update_weights()
        self.cell.update_weights()
        self.state.update_weights()

        return gradient_next
    
# TODO: fix Back Propergation
class GRU(Layer):
    def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam"):
        super().__init__(n_units, input_shape, activation, optimizer)
        self.sigmoid = act_functions["sigmoid"]()
        self.activation_output = act_functions[activation_output]()


        self.reset = Gate(self.optimizer)# rt
        self.update = Gate(self.optimizer)# zt
        self.hidden = Gate(self.optimizer)# ht

        


        self.input    = Gate(self.optimizer)
        self.output   = Gate(self.optimizer)
        self.state    = Gate(self.optimizer)

        if self.input_shape != None:
            self.__call__(self.input_shape)

    def __call__(self, input_shape=None):
        if self.input_shape is None:
            self.input_shape = input_shape
    
        timesteps, input_dim = self.input_shape
        shape_input  = (self.n_units, input_dim)
        # shape_state  = (self.n_units, input_dim)
        shape_state  = (self.n_units, self.n_units)
        shape_output = (input_dim, self.n_units)
        
        # initialize gates
        self.reset.initialize(shape=shape_input)
        self.update.initialize(shape=shape_input)
        self.hidden.initialize(shape=shape_input)

        # self.input.initialize(shape=shape_input)
        self.output.initialize(shape=shape_output)
        self.state.initialize(shape=shape_state)

    def forward(self, X):
        # need for backprop
        self.layer_input = X

        batch_size, timesteps, input_dim = X.shape
        self.reset.values = np.zeros((batch_size, timesteps, self.n_units))
        self.update.values = np.zeros((batch_size, timesteps, self.n_units))
        self.hidden.values = np.zeros((batch_size, timesteps, self.n_units))
        self.state.values = np.zeros((batch_size, timesteps, self.n_units))
        self.output.values = np.zeros((batch_size, timesteps, input_dim))
        
        # pass trough neuron
        for t in range(timesteps):
            Wu, Wr, Wo, Wh = self.update.W, self.reset.W, self.output.W, self.hidden.W
            Uu, Ur, Uo, Uh = self.update.U, self.reset.U, self.output.U, self.hidden.U
            xt = X[:, t]
            ht_prev = self.hidden.values[:, t-1]

            zt = self.sigmoid(xt.dot(Wu.T) + ht_prev.dot(Uu))
            rt = self.sigmoid(xt.dot(Wr.T) + ht_prev.dot(Ur))
            ht_hat = self.activation(xt.dot(Wh.T) + np.dot((rt * ht_prev), Uh))
            ht = ((1 - zt) * ht_prev) + (zt * ht_hat)
            ot = ht.dot(Wh)

            self.update.values[:, t] = zt
            self.reset.values[:, t]  = rt
            self.hidden.values[:, t] = ht
            self.output.values[:, t] = ot
            
        return self.output.values

    def backward(self, gradient):
        print("TODO: fix Back Propergation")
        gradient_next = np.zeros_like(gradient)

        # gradients
        self.update.W_gradient = np.zeros_like(self.update.W)
        self.reset.W_gradient = np.zeros_like(self.reset.W)
        self.hidden.W_gradient = np.zeros_like(self.hidden.W)
        self.state.W_gradient = np.zeros_like(self.state.W)
        self.output.W_gradient = np.zeros_like(self.output.W)

        # update layer weights/params
        _, timesteps, _ = gradient.shape
        for t in reversed(range(timesteps)):
            # TODO: fix Back Propergation
            self.update.W_gradient += 0.00
            self.reset.W_gradient  += 0.00
            self.hidden.W_gradient += 0.00
            self.state.W_gradient   += 0.00
            self.output.W_gradient  += 0.00

        self.update.update_weights()
        self.reset.update_weights()
        self.hidden.update_weights()
        self.state.update_weights()
        self.output.update_weights()

        return gradient_next

# Default Structure of layers
# class Input()
# class Output()
# class RNN(Layer):
#     def __init__(self, n_units, input_shape=None, activation="tanh", optimizer="adam"):
#         super().__init__(n_units, input_shape, activation, optimizer)
# 
#     def __call__(self, input_shape, static_weights=True):
#         # input shape from previous layer
#         if self.input_shape is None:
#             self.input_shape = input_shape
# 
#     def forward(self, X):
#         return X
#   
#     def backward(self, loss):
#         return loss

class Network():
    def __init__(self, layers=[], loss="MSE"):
        self.layers = layers
        self.loss_function = loss_functions[loss]()

        # set all layer input params
        for i in range(len(self.layers)):
            prev = min(0, i-1)
            input_shape = (self.layers[prev].n_units,)
            self.layers[i](input_shape)

    def add(self, layer):
        prev = min(0, len(self.layers)-1)

        # set all layer input params
        self.layers.append(layer)
        input_shape = (self.layers[prev].n_units,)
        self.layers[-1](input_shape)

    def remove(self, layer_index):
        del self.layers[layer_index]
        return self.layers
    
    def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
        history_loss = []

        for epoch in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in self._gen_batch(X, y, batch_size):
                y_pred = self._forward(X)

                # print("called")
                # the loss gradient with respect to y_pred
                gradient_loss = self.loss_function.gradient(y, y_pred)

                # Backpropagate. Update weights
                self._backward(gradient_loss)

                batch_error.append(
                    np.mean(self.loss_function(y, y_pred))
                )
            
            history_loss.append(
                np.mean(batch_error)
            )

            # display trained network state
            if verbose:
                print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

        # model accuracy
        if verbose:
            y_true = np.argmax(y, axis=2)
            y_pred = np.argmax(self.predict(X), axis=2)
            accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
            print(f"\nAccuracy: {accuracy}%\n")

        return history_loss

    def predict(self, X):
        return self._forward(X)

    def _backward(self, gradient_loss):
        # update weights in each layer, bind the output to next input gradient loss
        for layer in reversed(self.layers):
            gradient_loss = layer.backward(gradient_loss)
            
    def _forward(self, X):
        output = X

        # walk through layers, bind the output to next input
        for layer in self.layers:
            output = layer.forward(output)

        return output
        
    def _gen_batch(self, X, y=None, batch_size=64):
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i+batch_size, n_samples)
            if y is not None:
                yield X[begin:end], y[begin:end]
            else:
                yield X[begin:end]

# new design
class Network2():
    def __init__(self, layers=[], loss="MSE", activation_output=None):
        self.layers = layers
        self.loss_function = loss_functions[loss]()
        self.activation_output = None

        # activation on last layer
        if activation_output != None:
            self.activation_output = act_functions[activation_output]()

        # set all layer input params
        for i in range(len(self.layers)):
            prev = min(0, i-1)
            input_shape = (self.layers[prev].n_units,)
            self.layers[i](input_shape)

    def add(self, layer):
        prev = min(0, len(self.layers)-1)

        # set all layer input params
        self.layers.append(layer)
        input_shape = (self.layers[prev].n_units,)
        self.layers[-1](input_shape)

    def remove(self, layer_index):
        del self.layers[layer_index]
        return self.layers
    
    def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
        history_loss = []

        for epoch in range(n_epochs):
            batch_error = []
            # for X_batch, y_batch in self._gen_batch(X, y, batch_size):
            y_pred = self._forward(X)

            # the loss gradient with respect to y_pred
            gradient_loss = self.loss_function.gradient(y, y_pred)

            # Backpropagate. Update weights
            self._backward(gradient_loss, y_pred)

            batch_error.append(
                np.mean(self.loss_function(y, y_pred))
            )
            
            history_loss.append(
                np.mean(batch_error)
            )

            # display trained network state
            # print("called")
            if verbose:
                print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

        # model accuracy
        if verbose:
            y_true = np.argmax(y, axis=2)
            y_pred = np.argmax(self.predict(X), axis=2)
            accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
            print(f"\nAccuracy: {accuracy}%\n")

        return history_loss

    def predict(self, X):
        return self._forward(X)

    def _backward(self, gradient_loss, y_pred):
        # derive activation on output (softmax)
        if self.activation_output != None:
            gradient_loss = gradient_loss * self.activation_output.gradient(y_pred)

        # update weights in each layer, bind the output to next input gradient loss
        for layer in reversed(self.layers):
            gradient_loss = layer.backward(gradient_loss)
            
    def _forward(self, X):
        output = X

        # walk through layers, bind the output to input
        for layer in self.layers:
            output = layer.forward(output)

        # activation on output (softmax)
        if self.activation_output != None:
            output = self.activation_output(output)

        return output
        
    def _gen_batch(self, X, y=None, batch_size=64):
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i+batch_size, n_samples)
            if y is not None:
                yield X[begin:end], y[begin:end]
            else:
                yield X[begin:end]



# new design

