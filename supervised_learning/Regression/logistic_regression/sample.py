import numpy as np
import matplotlib.pyplot as plt
import argparse

import logistic_regression as algorithm

class LogisticRegression:

    def __init__(self, learning_rate = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.logistic = algorithm.LogisticRegression(
            batch_size = 64, 
            learning_rate = learning_rate, 
            epochs = epochs
        )

    def execute(self, X1, X2, y1, y2):
        X = np.concatenate((X1, X2), axis=0)
        X = np.concatenate((X, np.ones(shape=(2 * len(X1), 1))), axis=1)
        y = np.concatenate((y1, y2), axis=0)
        self.w = np.random.normal(size=(X.shape[1], 1))
        self.iteration = 0

        """Calculation"""
        costs, thetas, iterations = self.logistic.batch_gradient_descent(X, y, self.w)
        y = self.logistic.predict(X, thetas[-1])

        # calculate data visualization
        X1 = X[:len(X1), :]
        X2 = X[len(X1):, :]
        y1 = y[:len(X1), :]
        y2 = y[len(X1):, :]

        # straight line corner to corner
        line = np.linspace(0, 20, num=len(X1)).reshape((-1, 1))
        bias = np.ones((len(X1), 1))

        line_plot = np.concatenate((line, line, bias), axis=1)
        line_plot = self.logistic._sigmoid(line_plot, thetas[-1])
        line_z = np.squeeze(line_plot)

        # display
        plt.figure(1, figsize=(6, 6))

        ax = plt.axes(projection='3d')
        ax.set_title("Loss: " + str(costs[-1]), fontsize=20)
        ax.set_xlabel("x axis", fontsize=14)
        ax.set_ylabel("y axis", fontsize=14)
        ax.set_zlabel("z axis", fontsize=14);

        label = "Iterations: " + str(iterations)
        ax.scatter(line, line, line_z, cmap='viridis', linewidth=0.5, label=label);
        ax.scatter(X1[:, 0], X1[:, 1], y1, c="b", label="Class 1")
        ax.scatter(X2[:, 0], X2[:, 1], y2,  c="r", label="Class 0")
        
        plt.show()

def get_data(length_data = 100):
    X1 = np.random.multivariate_normal([5, 6], [[5, 1], [1, 5]], length_data)
    X2 = np.random.multivariate_normal([14, 15], [[4, 0], [0, 4]], length_data)
    
    y1 = np.ones(
        shape = (length_data, 1)
    )
    
    y2 = np.zeros(
        shape = (length_data, 1)
    )

    return X1, X2, y1, y2

if __name__ == "__main__":
    lr = LogisticRegression(learning_rate=0.1, epochs=3000)
    X1, X2, y1, y2 = get_data(length_data = 50)
    lr.execute(X1, X2, y1, y2)






