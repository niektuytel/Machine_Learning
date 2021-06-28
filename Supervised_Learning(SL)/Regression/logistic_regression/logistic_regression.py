"""
Author: Niek Tuytel
Email: niektuytel20@gmail.com
"""
import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, batch_size=64, learning_rate=0.1, epochs=1000):
        """
        Contructor for logistic regression.

        Parameter
        ---------
        batch_size: number of batch size using each iteraction
        epochs: max number of interactions to train logistic regression.
        learning_rate: learning_rate algorithm to update weights.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _hypothesis(self, X, w):
        """
        Compute the Hypothesis.
        """
        return X.dot(w)

    def _sigmoid(self, X, w):
        """
        Sigmoid activation function:
            h = X.w
            s(h) = 1/(1+e^-x)

        Parameter
        ---------
        X: matrix of dataset. shape = (n, d) with n is number of training, d
            is dimension of each vector.

        Return
        ---------
        s(x): value of activation.
        """
        h = self._hypothesis(X, w)
        return 1/(1+np.exp(-h))

    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross entropy loss.
        """
        m = y_true.shape[0]
        return -np.sum(y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred)) / m

    def _gradient(self, X, y_true, y_pred):
        """
        Compute gradient of J with respect to `w`
        """
        m = X.shape[0]
        return (X.T.dot(y_pred - y_true)) / m

    def _early_exit(self, loss_history):
        check = 8

        if len(loss_history) > 8:
            if loss_history[-check] == loss_history[-1]:
                return True
        return False

    def train(self, train_X, train_y, w):
        """
        Wrapper training function, check the prior condition first
        """
        assert type(train_X) is np.ndarray, "Expected train X is numpy array but got %s" % type(train_X)
        assert type(train_y) is np.ndarray, "Expected train y is numpy array but got %s" % type(train_y)
        train_y = train_y.reshape((-1, 1))

        self.w = w
        # self.w = np.random.normal((train_X.shape[1], 1))

        return self.batch_gradient_descent(train_X, train_y, self.w)
        
    def predict(self, test_X, weight):
        """
        Output sigmoid value of trained parameter w, b.
        Choose threshold 0.5
        """
        pred = self._sigmoid(test_X, weight)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def batch_gradient_descent(self, X, y, weight):
        """
        Main training function        
        """
        loss_history = []
        theta_history = []

        for epoch in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            index = 0

            while index < X.shape[0]:
                index_end = index + self.batch_size
                batch_X = X[index:index_end]
                batch_y = y[index:index_end]
                
                y_pred = self._sigmoid(batch_X, weight)
                loss = self._cross_entropy_loss(batch_y, y_pred)
                theta_history.append(weight)
                loss_history.append(loss)

                grad = self._gradient(batch_X, batch_y, y_pred)
                weight -= self.learning_rate * grad

                batch_loss += loss
                index += self.batch_size
                num_batches += 1

                if self._early_exit(loss_history): 
                    print("Early exit at epoch: " + epoch)
                    return
            
            if epoch % 10 == 0:
                print("Loss at epoch " + str(epoch + 1) + ": " + str(batch_loss / num_batches))
            
        return loss_history, theta_history, epoch

# def clean_sentences(string):
#     label_chars = re.compile("[^A-Za-z \n]+")
#     string = string.lower()
#     return re.sub(label_chars, "", string)

# def sample():
#     df = pd.read_csv("./data/amazon_baby_subset.csv")
#     reviews = df.loc[:, "review"].values
#     for ind, review in enumerate(reviews):
#         if type(review) is float:
#             reviews[ind] = ""
    
#     reviews = clean_sentences("\n".join(reviews))
#     with open("./data/important_words.json") as f:
#         important_words = json.load(f)
    
#     reviews = reviews.split("\n")
#     n = len(reviews)
#     d = len(important_words)
#     X = np.zeros((n, d))
#     y = df.loc[:, "sentiment"].values
#     y[y == -1] = 0

#     for ind, review in enumerate(reviews):
#         for ind_w, word in enumerate(important_words):
#             X[ind, ind_w] = review.count(word)
    
#     ones = np.ones((n, 1))
#     X = np.concatenate((X, ones), axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     epochs = 20
#     learning_rate = 0.1
#     batch_size = 64

#     # Calculation
#     weight = np.random.normal(size=(X_train.shape[1], 1))
#     logistic = LogisticRegression(batch_size, learning_rate, epochs)
#     _, _, _ = logistic.train(X_train, y_train, weight)

#     pred = logistic.predict(X_test, logistic.w)
#     y_test = y_test.reshape((-1, 1))
#     print("Accuracy: " + str(len(pred[pred == y_test]) / len(pred)))

# if __name__ == "__main__":
#     sample()
