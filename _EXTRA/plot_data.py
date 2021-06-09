import numpy as np
import matplotlib.pyplot as plt

def visualize(data_x, data_y, targets, predict_function=None):
    # predict data if contains predict_function
    if predict_function is not None:
        # (+/-)1 => let the border be larger than data points
        x_min = data_x.min() - 1
        x_max = data_x.max() + 1
        y_min = data_y.min() - 1
        y_max = data_y.max() + 1

        # get all data between min & max point
        x_all = np.arange(x_min, x_max, 0.1)
        y_all = np.arange(y_min, y_max, 0.1)

        # merge all x and y data into a vector
        x_vec, y_vec = np.meshgrid(x_all, y_all)
        xy_vec = np.c_[x_vec.ravel(), y_vec.ravel()]

        # predict data points
        yhat = predict_function(xy_vec)
        z_vec = yhat.reshape(x_vec.shape)

        # Plotting decision regions
        plt.contourf(x_vec, y_vec, z_vec, alpha=0.4)

    plt.scatter(data_x, data_y, c=targets, s=20, edgecolor='k')
    plt.title("Visualize dicision boundaries")
    plt.show()
    
    # return predictions
    if predict_function is not None:
        return yhat

# SAMPLE
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    
    # Loading some example data
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf1.fit(X, y)

    # visualize predictions
    visualize(X[:, 0], X[:, 1], y, clf1.predict)

