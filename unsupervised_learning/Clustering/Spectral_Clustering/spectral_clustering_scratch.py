import math # for math
import numpy as np # for math
import pandas as pd # for csv file reading
import matplotlib.pyplot as plt # for data visualization
from sklearn.cluster import KMeans # need in the spectral clustering
from scipy.spatial.distance import pdist, squareform # to generate adjacency values

class SpectralClustering():
    def __init__(self, n_clusters=5, metric="euclidean", similarity_type="knn", knn=10, sigma=1, epsilon=0.5):
        """
        Parameters:
        -----------
        n_clusters: int
            the amount of different colored groups, than will been generated

        metric: string
            the type of metric that will been used to generate the adjacency

        similarity_type: string
            the type of graph similarity calculation: 
            'fully_connect', 'eps_neighbor', 'knn', 'mutual_knn'.
            
        knn: int
            used when similarity_type is ['mutual_knn', 'knn'], 
            the amount of values of the weights are been checked

        sigma: int 
            used when similarity_type is 'fully_connect', 
            for Standard deviation (Gaussian noise). 

        epsilon: float
            used when similarity_type is 'eps_neighbor', 
            to get all the values in the length of the epsilon value
        """
        self.n_clusters=n_clusters
        self.metric=metric
        self.similarity_type=similarity_type
        self.knn=knn
        self.sigma=sigma
        self.epsilon=epsilon

    def _get_adjacency_weights(self):
        """ 
        Compute the weighted adjacency matrix based on the self.similarity_type:
            knn: (k-nearest neighbors)
                return of 1's if the weight slice is below the self.knn index else 0's
            mutual_knn: (mutual k-nearest neighbors)
                return a array with True and False values, (1=True, 0=False) on self.knn index per array
            fully_connect: 
                return the exponential value of the negative adjacency devided by self.sigma
            eps_neighbor:
                return all float adjacency values, that contains in the self.epsilon length
        """
        adjacency = self.adjacency_matrix

        if "knn" in self.similarity_type:
            # Sort the adjacency matrx by rows and record the indices
            adjacency_sort = np.argsort(adjacency, axis=1)

            if self.similarity_type == 'knn':
                # Set the weight (i,j) to 1 when either i or j is within the k-nearest neighbors of each other
                weights = np.zeros(adjacency.shape)
                for i in range(adjacency_sort.shape[0]):
                    weights[i,adjacency_sort[i,:][:(self.knn + 1)]] = 1

            elif self.similarity_type == 'mutual_knn':
                # Set the weight W1[i,j] to 0.5 when either i or j is within the k-nearest neighbors of each other (Flag)
                # Set the weight W1[i,j] to 1 when both i and j are within the k-nearest neighbors of each other
                W1 = np.ones(adjacency.shape)
                for i in range(adjacency.shape[0]):
                    for j in adjacency_sort[i,:][:(self.knn+1)]:
                        if W1[i,j] == 0 and W1[j,i] == 0:
                            W1[i,j] = 0.5

                weights = np.copy((W1 > 0.5).astype('float64'))

        elif self.similarity_type ==  'fully_connect':
            weights = np.exp(-adjacency/(2 * self.sigma))
        elif self.similarity_type == 'eps_neighbor':
            weights = (adjacency <= self.epsilon).astype('float64')
        else:
            raise ValueError(
                """
                    The 'similarity_type' should be one of the following types: 
                    [
                        'fully_connect', 
                        'eps_neighbor', 
                        'knn', 
                        'mutual_knn'
                    ]
                """
            )
            
        return weights

    def _project_and_transpose(self, weights, normalized=1):
        # Compute the degree matrix and the unnormalized graph Laplacian
        D = np.diag(np.sum(weights, axis=1))
        L = D - weights
        
        # Compute the matrix with the first K eigenvectors as columns based on the normalized type of L
        if normalized == 1:   ## Random Walk normalized version
            # Compute the inverse of the diagonal matrix
            D_inv = np.diag(1/np.diag(D))
            
            # Compute the eigenpairs of L_{rw}
            Lambdas, V = np.linalg.eig(np.dot(D_inv, L))
            
            # Sort the eigenvalues by their L2 norms and record the indices
            ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
            V_K = np.real(V[:, ind[:self.n_clusters]])

        elif normalized == 2:   ## Graph cut normalized version
            # Compute the square root of the inverse of the diagonal matrix
            D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D)))
            
            # Compute the eigenpairs of L_{sym}
            Lambdas, V = np.linalg.eig(np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt))
            
            # Sort the eigenvalues by their L2 norms and record the indices
            ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
            V_K = np.real(V[:, ind[:self.n_clusters]])
            
            # Normalize the row sums to have norm 1
            V_K = V_K/np.reshape(np.linalg.norm(V_K, axis=1), (V_K.shape[0], 1))

        else:   ## Unnormalized version
            
            # Compute the eigenpairs of L
            Lambdas, V = np.linalg.eig(L)
            
            # Sort the eigenvalues by their L2 norms and record the indices
            ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
            V_K = np.real(V[:, ind[:self.n_clusters]])

        return V_K

    def fit_predict(self, X, is_adjacency_matrix=False, type_normalization=1):
        """
        Parameters:
        -----------
        X: numpy Array
            given data points
        is_adjacency_matrix: bool
            if X is already a adjacency matrix than the value need to be True,
            so the generation is been ignored to a new adjacency matrix
        type_normalization: int
            for normalizing the data of adjacency weights
        """
        self.X = X

        # Compute the adjacency matrix
        if not is_adjacency_matrix:
            dist = pdist(X, metric=self.metric)
            self.adjacency_matrix = squareform(dist)
        else:
            self.adjacency_matrix = X

        # Get the adjacency weights from given user input 
        weights = self._get_adjacency_weights()
        
        # Similarity Graph
        similarities = self._project_and_transpose(weights, type_normalization)
        
        # Conduct K-Means on the matrix with the first K eigenvectors as columns
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0)
        yhat = kmeans.fit_predict(similarities)

        return yhat 
        
if __name__ == '__main__':
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values

    # define the model
    model = SpectralClustering(n_clusters=5)

    # assign a cluster to each example
    yhat = model.fit_predict(X)

    # Retrieve unique clusters
    clusters = np.unique(yhat)

    # Create scatter plot samples for each example
    for cluster in clusters:

        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)

        # create scatter for this sample
        plt.scatter(X[row_ix, 0], X[row_ix, 1])

    # show the window
    plt.show()
