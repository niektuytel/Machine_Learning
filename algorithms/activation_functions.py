import numpy as np # for math

# Collection of activation functions
# References: 
# https://en.wikipedia.org/wiki/Activation_function
# https://miro.medium.com/proxy/1*RD0lIYqB5L2LrI2VTIZqGw.png
# https://deepai.org/machine-learning-glossary-and-terms/sigmoid-function


class Sigmoid():
    # σ(x) = 1 / 1+e(−xb)
    def __call__(self, x, bias=0):
        return 1 / (1 + np.exp(-x + bias))
    
    def derivative(self, x, bias=0):
        return self.__call__(x, bias) * (1 - self.__call__(x, bias))

class TanH():
    # x = cosh(a) = e(a)+e(−a)/2​   &    y = sinh(a) = e(a)−e(−a)/2
    # https://brilliant.org/wiki/hyperbolic-trigonometric-functions/
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def derivative(self, x):
        return 1 - (self.__call__(x) ** 2)

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)
    
    def derivative(self, x):
        return np.where(x >= 0, 1, 0)

class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))
    
    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x) - 1))
    
    def derivative(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class Softmax():
    def __call__(self, x):
        exponential = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exponential / np.sum(exponential , axis=-1, keepdims=True)
    
    def derivative(self, x):
        p = self.__call__(x)
        return p * (1 - p)

activation_functions = {
    "sigmoid"   : Sigmoid,
    "tanh"      : TanH,
    "relu"      : ReLU,
    "softplus"  : SoftPlus,
    "leakyrelu" : LeakyReLU,
    "elu"       : ELU,
    "selu"      : SELU,
    "softmax"   : Softmax
}
