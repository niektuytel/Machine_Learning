# TODO
# class MeanAbsoluteError():
#     def __init__(self): pass

# TODO
# class MeanBiasError():
#     def __init__(self): pass
 
# TODO
# class ClassificationLosses():
#     def __init__(self): pass

# TODO
# class Elbow():
#     def __init__(self): pass

# TODO
# class EuclideanDistance():
#     def __init__(self): pass

# TODO
# class Graussian():
#     def __init__(self): pass

####################################################################

import numpy as np # for math

# Resources
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html


class MeanSquareError():
    def __init__(self): pass

    def loss(self, yhat, y):
        return 0.5 * np.power((y - yhat), 2)
    
    def derivative(self, yhat, y):
        return -(y - yhat)

    def accuracy(self, yhat, y):
        return 0


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

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

if __name__ == "__main__":    
    yhat = np.array(
        [ 
            [0.25,0.25,0.25,0.25], 
            [0.01,0.01,0.01,0.96] 
        ]
    )
    y = np.array(
        [ 
            [0,0,0,1], 
            [0,0,0,1] 
        ]
    )

    mse = MeanSquareError()
    print(mse.loss(yhat, y))


