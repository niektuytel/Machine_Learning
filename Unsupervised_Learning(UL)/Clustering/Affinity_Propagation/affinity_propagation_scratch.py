import numpy as np# for the math
import pandas as pd# for read csv file
import matplotlib.pyplot as plt# for visualization
from itertools import cycle# for generate a cycle in data

class Affinity_Propagation(object):
    def __init__(self, data):
        self.data = data
        
    # as we need to bind all the data points with each other,
    # we need the higest value of the x or y.
    def _empty_matrix(self, X):
        return np.zeros((max(X.shape), max(X.shape)))

    """
    The similarity function is the negative euclidian distance squared function.

    s(i,k) = −||xi − xk|| ** 2

    We can simply implement this similarity function and define a similarity matrix called similar.
    Which is a graph of the similarities between all the points. 

    We also initialize the R and A matrix to zeros.
    """
    def similarity(self):
        S = self._empty_matrix(self.data)

        # when looking in row i, the value means you should compare to column i - value
        for i in range(S.shape[0]):
            for k in range(S.shape[1]):
                S[i, k] = -((self.data[i] - self.data[k]) ** 2).sum()

        return S

    """
    The responsibility messages are defined by:

    r(i,k) ← s(i,k) −max[k′s.t.k′≠k] {a(i,k′)+s(i,k′)}
    
    We could implement this with a nested for loop where we iterate over every row 'i'
    and then determine the max(A + S)(of that row) for every index not equal to 'k' or 'i'
    (The index should not be equal to i as it would be sending messages to itself). 

    The damping factor is just there for nummerical stabilization and can be regarded as a slowly converging learning rate. 
    The authors advised to choose a damping factor within the range of 0.5 to 1.
    """
    def responsibility(self, A, R, S, lmda=0.9, slow=False):

        # the slow way of getting responsibilities
        if slow:
            for i in range(R.shape[0]):
                for k in range(R.shape[1]):
                    v = S[i, :] + A[i, :]
                    v[k] = -np.inf
                    v[i]= -np.inf
                    R[i, k] = R[i, k] * lmda + (1 - lmda) * (S[i, k] - np.max(v))
        else:
            # For every column k, except for the column with the maximum value the max is the same.
            # So we can subtract the maximum for every row, and only need to do something different for k == argmax
            v = S + A
            rows = np.arange(R.shape[0])

            # We only compare the current point to all other points, so the diagonal can be filled with -infinity
            np.fill_diagonal(v, -np.inf)

            # max values
            idx_max = np.argmax(v, axis=1)
            first_max = v[rows, idx_max]

            # Second max values. For every column where k is the max value.
            v[rows, idx_max] = -np.inf
            second_max = v[rows, np.argmax(v, axis=1)]

            # Broadcast the maximum value per row over all the columns per row.
            max_matrix = np.zeros_like(R) + first_max[:, None]
            max_matrix[rows, idx_max] = second_max

            new_val = S - max_matrix

            R = R * lmda + (1 - lmda) * new_val

        return R

    """
    The availability messages are defined by the following formulas. 
    For all points not on the diagonal of A (all the messages going from one data point to all other points), 
    the update is equal to the responsibility that point k assigns to itself and 
    the sum of the responsibilities that other data points (except the current point) assign to k. 
    Note that, due to the min function, this holds only true for negative values.

    a(i,k) ← min{0,r(k,k) + ∑[i′s.t.i′∉{i,k}] max{0,r(i′,k)}
    
    For points on the diagonal of A (the availability value that a data point sends to itself), 
    the message value is equal to the sum of all positive responsibility values send to the current data point.

    a(k,k)←∑i′≠k max(0,r(i′,k))
    """
    def availability(self, A, R, S, lmda=0.9, slow=False):
        if slow:
            for i in range(A.shape[0]):
                for k in range(A.shape[1]):
                    v = np.array(R[:, k])
                    if i != k:
                        v[i] = -np.inf
                        v[k] = - np.inf
                        v[v < 0] = 0

                        A[i, k] = A[i, k] * lmda + (1 - lmda) * min(0, R[k, k] + v.sum())
                    else:
                        v[k] = -np.inf
                        v[v < 0] = 0
                        A[k, k] = A[k, k] * lmda + (1 - lmda) * v.sum()
        else:
            k_k_idx = np.arange(A.shape[0])
            
            # set a(i, k)
            v = np.array(R)
            v[v < 0] = 0
            np.fill_diagonal(v, 0)
            v = v.sum(axis=0) # columnwise sum
            v = v + R[k_k_idx, k_k_idx]

            # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
            v = np.ones(A.shape) * v

            # For every column k, subtract the positive value of k. 
            # This value is included in the sum and shouldn't be
            v -= np.clip(R, 0, np.inf)
            v[v > 0] = 0
            
            # set(a(k, k))
            v_ = np.array(R)
            np.fill_diagonal(v_, 0)

            v_[v_ < 0] = 0

            v[k_k_idx, k_k_idx] = v_.sum(axis=0) # column wise sum
            A = A * lmda + (1 - lmda) * v

        return A

# a(k,k) ← ∑[i′≠k] max(0,r(i′,k))
def plot_iteration(matrix, A, R):
    fig = plt.figure(figsize=(12, 6))
    sol = A + R
    
    labels = np.argmax(sol, axis=1)

    exemplars = np.unique(labels)
    colors = dict(zip(exemplars, cycle('bgrcmyk')))
    
    for i in range(len(labels)):
        X = matrix[i][0]
        Y = matrix[i][1]
        
        if i in exemplars:
            exemplar = i
            edge = 'k'
            ms = 10
        else:
            exemplar = labels[i]
            ms = 3
            edge = None
            plt.plot([X, matrix[exemplar][0]], [Y, matrix[exemplar][1]], c=colors[exemplar])
        plt.plot(X, Y, 'o', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        

    plt.title('Number of exemplars: %s' % len(exemplars))
    return fig, labels, exemplars

if __name__ == "__main__":
    # Loading data
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    matrix = dataset.iloc[:, [3, 4]].values

    model = Affinity_Propagation(matrix)

    # bind data to each other 
    S = model.similarity()
    R = model._empty_matrix(matrix)
    A = model._empty_matrix(matrix)

    preference = np.median(S)
    preference = -1000

    np.fill_diagonal(S, preference)
    damping = 0.5
    figures = []

    for i in range(50):
        R = model.responsibility(A, R, S, damping)
        A = model.availability(A, R, S, damping, 0)
        
        if i % 5 == 0:
            fig, labels, exemplars = plot_iteration(matrix, A, R)
            figures.append(fig)

    plt.show()
    # # Make a give file from outcomming results
    # def make_gif(figures, filename, fps=10, **kwargs):
    #     images = []
    #     for fig in figures:
    #         output = BytesIO()
    #         fig.savefig(output)
    #         plt.close(fig)  
    #         output.seek(0)
    #         images.append(imageio.imread(output))
    #     imageio.mimsave(filename, images, fps=fps, **kwargs)

    # make_gif(figures, './test.gif', 2)
