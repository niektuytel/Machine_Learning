import numpy as np

class Input:
    """
    Input Neuron given his input to his output.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.inputs = np.full(input_shape, None)

    def call(self, X):
        self.inputs = X
        return self.inputs

class InputBackfed:
    def __init__(self):
        ""

class InputNoisy:
    def __init__(self):
        ""


class Hidden:
    def __init__(self):
        ""

class HiddenProbablistic:
    def __init__(self):
        ""

class HiddenSpiking:
    def __init__(self):
        ""


class Output:
    def __init__(self):
        ""

class OutputInputMatch:
    def __init__(self):
        ""


class Recurrent:
    def __init__(self):
        ""

class RecurrentMemory:
    def __init__(self):
        ""

class RecurrentMemoryDifferent:
    def __init__(self):
        ""


class Kernel:
    def __init__(self):
        ""
        
class ConvolutionOrPool:
    def __init__(self):
        ""
