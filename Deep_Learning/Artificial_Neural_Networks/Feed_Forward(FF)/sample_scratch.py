import numpy as np

class Sigmoid():
    # σ(x) = 1 / 1+e(−xb)
    def __call__(self, x, bias=0):
        return 1 / (1 + np.exp(-x + bias))
    
    def derivative(self, x, bias=0):
        return self.__call__(x, bias) * (1 - self.__call__(x, bias))

class CrossEntropy():
    # https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon# Close To 0

    def loss(self, yhat, y):
        # Avoid division by zero
        yhat = np.clip(yhat, self.epsilon, 1. - self.epsilon)

        # get losses values 
        return -y * np.log(yhat) - (1 - y)* np.log(1 - yhat)

    def accuracy(self, yhat, y):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(yhat, axis=1))

    def derivative(self, yhat, y):
        # Avoid devision by zero
        yhat = np.clip(yhat, self.epsilon, 1. - self.epsilon)

        # get derivative values
        return -(y / yhat) + (1 - y) / (1 - yhat)

#------------------------------------------------------------------------------------------

class Perceptron(object):
    """
    Perceptron is the Mathamatical word for `Neuron`, 
    Calculate output based on his type of character.
    This character will been made by his unique weights and bias

    Resources:
    -----------
    http://neuralnetworksanddeeplearning.com/chap2.html
    https://cs231n.github.io/neural-networks-1/
    https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backward_passagation-algorithm-a5cd5b904fde
    
    Parameters:
    -----------
    n_input: int
        number of values that will been inputted to this neuron
    n_ouput: int
        number of values that will been outputted from this neuron
    """
    def __init__(self, n_input, n_output): 
        self.n_input        = n_input
        self.n_output       = n_output
        self.weighted_sum   = 0.00
        self.activation = Sigmoid()

    def forward_pass(self, weights, bias, inputs):
        # data input
        self.input = inputs

        # a = σ(wa+b)
        self.weighted_sum = np.dot(weights, inputs) + bias
        
        # predicted input
        self.output = self.activation(self.weighted_sum)

        return self.output
    
    def backward_pass(self, error, prev_yhat):
        # gradient value of weighted sum * error is the delta
        delta = self.activation.derivative(self.weighted_sum) * error

        # new weight = delta * neuron expected output
        weights = np.dot(delta, prev_yhat.transpose())

        return weights, delta

class Neuron(Perceptron):
    # some use `Neuron`, for simplicity in mind
    # https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#neuron
    pass

class Dense(Perceptron):
    # some use `Dense`, one of the reasons is `Dense Layer` makes more sense than `Perceptron Layer`
    # https://stackoverflow.com/questions/43755293/what-does-dense-do
    pass

class Node(Perceptron):
    # some use `Node`, for simplicity in mind, comparing to a Tree structure
    # https://orbograph.com/understanding-ai-what-is-a-deep-learning-node/
    pass

class Unit(Perceptron):
    # if they not like to compare it to Neurons of humans they call it `Units`
    # https://cs231n.github.io/neural-networks-1/
    pass

#------------------------------------------------------------------------------------------

class Layer:
    """
    Layer is a row of Neurons(Perceptrons).
    All the data that flows trough the neurons is stored into the layer,
    as a Neuron is only a function to execute inputs to some outputs.
    The layer is doing the task as `FeedForward` and `Backpropagation`
    for getting the result from 1 layer in the neural network

    Resources:
    -----------
    http://neuralnetworksanddeeplearning.com/chap2.html
    https://cs231n.github.io/neural-networks-1/
    https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backward_passagation-algorithm-a5cd5b904fde
    
    Parameters:
    -----------
    n_input: int
        number of values that will been inputted to this layer
    n_ouput: int
        number of neurons implemented in this layer (so the number of layer outputs)
    """
    def __init__(self, n_input, n_output):
        self.outputs= np.zeros(n_output)
        self.layer  = np.full(n_output, None)
        self.weights= np.full(n_output, None)
        self.biases = np.full(n_output, None)
        self.deltas = np.full(n_output, None)
        for i in range(n_output):
            neuron = Perceptron(
                n_input=n_input, 
                n_output=n_output
            )
            self.layer[i] = neuron
            
            # generate weights randomly between [-1, 1]
            limit = 1 / np.sqrt(n_input)
            rand_weight = np.random.uniform(-limit, limit, n_input)

            self.weights[i] = rand_weight
            self.deltas[i]  = np.zeros(len(rand_weight))
            self.biases[i]  = 0.00

    def forward_pass(self, inputs):
        # get all neuron answers on the input
        for i, neuron in enumerate(self.layer):
            self.outputs[i] = neuron.forward_pass(weights=self.weights[i], bias=self.biases[i], inputs=inputs)
        
        return self.outputs

    def backward_pass(self, errorDerivative, y, prev_yhat, eta):
        # get the new weights based on the error input 
        for i, neuron in enumerate(self.layer):       
            error = errorDerivative[i]
            self.weights[i], self.deltas[i] = neuron.backward_pass(error, prev_yhat.transpose())

        # set new binding weights and biases
        self.weights = np.array(
            [w - (eta/len(y)) * w for w in self.weights]
        )
        self.biases = np.array(
            [b - (eta/len(y))*nb for b, nb in zip(self.biases, self.deltas)]
        )

class Dense(Layer):
    # some use `Dense`, as in the link `Keras` is using it.
    # https://deeplizard.com/learn/video/FK77zZxaBoI
    pass

#------------------------------------------------------------------------------------------

class Neural_Network:
    """
    Neural network is a structure of rows(layers) that contains neurons.
    The structure that is given to the Neural network do really really matters.
    We can see on the README.md what types of neural networks you have.

    Resources:
    -----------
    http://neuralnetworksanddeeplearning.com/chap2.html
    https://cs231n.github.io/neural-networks-1/
    https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backward_passagation-algorithm-a5cd5b904fde
    
    Parameters:
    -----------
    n_input: int
        number of values that will been inputted to this neural network
    n_neurons_per_layer: list
        each index in the list is a layer and the given number is for the amount of neurons.
        so for example [5, 4, 3] has 3 layers and the first layer has 5 neurons etc.
    """
    def __init__(self, n_input, n_neurons_per_layer):
        self.n_input = n_input
        self.n_output = n_neurons_per_layer[-1]
        self.loss_function  = CrossEntropy()

        # Create layers with given amount of neurons
        self.layers = []
        for i, n_neurons in enumerate(n_neurons_per_layer):
            n_layer_input = n_input if i == 0 else n_neurons_per_layer[i-1]

            self.layers.append(
                Layer(
                    n_input=n_layer_input, 
                    n_output=n_neurons
                ) 
            )
        
    def forward_pass(self, inputs):
        # every outputs is on the next layer his inputs
        for _, layer in enumerate(self.layers):
            inputs = outputs = layer.forward_pass(inputs)

        # return neural network output
        return outputs

    def backward_pass(self, x, y, avg_cost, eta):

        # feedforward
        yhat = self.forward_pass(x)
        avg_cost += sum(np.square(yhat - y))

        # last layer
        errorDerivative = self.loss_function.derivative(self.layers[-1].outputs, y)
        self.layers[-1].backward_pass(errorDerivative, y, self.layers[-2].outputs, eta)

        for l in range(2, len(self.layers)):
            # remaining layers
            weights_T = self.layers[-l+1].weights.transpose()
            errorDerivative = np.dot(weights_T, self.layers[-l+1].deltas)

            # update neuron weigths
            self.layers[-l].backward_pass(errorDerivative, y, self.layers[l-1].outputs, eta)

        return avg_cost

    def train(self, X, y, n_epochs = 100):
        n_examples = X[:].shape[1]
        avg_cost = 0
        eta = 3.0
        

        nabla_b = np.full(n_examples, 0.00)
        nabla_w = np.full(n_examples, 0.00)
        
        # online learning
        for i in range(n_examples):
            for n in range(n_epochs):
                avg_cost = self.backward_pass(X[i,:], y[i, :], avg_cost, eta)
                avg_cost = avg_cost / n_examples
            print("train: n_example -> " + str(i) + ", avg_cost:" + str(avg_cost)) 

# Example way of usage (Feed Forwarded)
if __name__ == "__main__":
    # train data
    X = np.array([
        [1,1,1], 
        [1,1,0], 
        [1,0,0], 
        [0,0,0], 
        [0,0,1], 
        [0,1,1]
    ])

    # train data results
    y = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ])


    # define neural network model
    neural_network = Neural_Network(
        n_input = 3, 
        n_neurons_per_layer = [3,3,2]
    )

    # train neural network on known data and predictions
    neural_network.train(X, y)

    """
    as we define the neural network model with known data to let himself train.
    After this you can input any data as well unknown data and than he go guess the result for it.
    We set the result of `[1,0,0]` as `[1, 0]` and our input is `[0, 1, 0]`
    This comes very close to each other as it has 2 zeros and 1 one, 
    this is why the neural net will lay his trust on this answer as this is the closest first answer.
    """
    unknown_input = [0, 1, 0]
    result = neural_network.forward_pass(unknown_input)

    print("\n\n")
    print("the prediction of the neural network: " + str(result))
    print("\n\n")
