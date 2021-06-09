import numpy as np
from numpy import loadtxt

class Sigmoid():
    def __call__(self, x, bias=0):
        return 1 / (1 + np.exp(-x + bias))
    def derivative(self, x, bias=0):
        return self.__call__(x, bias) * (1 - self.__call__(x, bias))

class Perceptron():
    def __init__(self, n_inputs, bias=0.00, activation_function=Sigmoid()):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.weights = 2 * np.random.random((n_inputs, 1)) - 1
        self.n_inputs = n_inputs
        self.bias = bias
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        # inputs as float values
        self.input = inputs.astype(float)

        # passing the inputs via the neuron to get output 
        self.weighted_sum = np.dot(self.input, self.weights) + self.bias
        
        # get firing rate from activation function
        self.output = self.activation_function(self.weighted_sum)
        return self.output

    def train(self, train_inputs, train_results, n_iterations=100):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(n_iterations):
            # siphon the training data via the neuron
            output = self.forward_pass(train_inputs)

            # computing error rate for back-propagation
            error = train_results - output

            # performing weight adjustments
            adjustments = np.dot(train_inputs.T, error * self.activation_function.derivative(output))
            self.weights += adjustments

    def predict(self, inputs):
        return self.forward_pass(inputs)

if __name__ == "__main__":
    # training data consisting of 4 examples -- 3 input values and 1 output
    train_inputs = np.array(
        [
            [0,0,1],
            [1,1,1],
            [1,0,1],
            [0,1,1]
        ]
    )

    train_results = np.array(
        [
            [0],
            [1],
            [1],
            [0]
        ]
    )

    # initializing the neuron class
    n_input = len(train_inputs[0])
    neuron = Perceptron(n_input)


    #training taking place
    print("Weights Before Training: \n", str(neuron.weights))
    neuron.train(train_inputs, train_results, 1500)
    print("Weights After Training: \n", str(neuron.weights))

    predict = np.array([[0, 1, 0]])
    print("\n\npredict: ", str(predict))
    print("result:  ", str(int(neuron.predict(predict)[0]*100)) + "% change it is a 1 as result")
