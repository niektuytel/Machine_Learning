import numpy as np # for math
import pandas as pd # for csv file reader
import matplotlib.pyplot as plt # for visualization
from sklearn.preprocessing import StandardScaler # scale data to a same value factor
# http://www.oranlooney.com/post/ml-from-scratch-part-6-pca/

class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = bool(whiten)

    def _householder_reflection(self, a, e):
        '''
        Given a vector a and a unit vector e,
        (where a is non-zero and not collinear with e)
        returns an orthogonal matrix which maps a
        into the line of e.
        '''
        
        assert a.ndim == 1
        assert np.allclose(1, np.sum(e**2))
        
        u = a - np.sign(a[0]) * np.linalg.norm(a) * e  
        v = u / np.linalg.norm(u)
        H = np.eye(len(a)) - 2 * np.outer(v, v)
        
        return H

    def _qr_decomposition(self, A):
        '''
        Given an n x m invertable matrix A, returns the pair:
            Q an orthogonal n x m matrix
            R an upper triangular m x m matrix
        such that QR = A.
        '''
        
        n, m = A.shape
        assert n >= m
        
        Q = np.eye(n)
        R = A.copy()
        
        for i in range(m - int(n==m)):
            r = R[i:, i]
            
            if np.allclose(r[1:], 0):
                continue
                
            # e is the i-th basis vector of the minor matrix.
            e = np.zeros(n-i)
            e[0] = 1  
            
            H = np.eye(n)
            H[i:, i:] = self._householder_reflection(r, e)

            Q = Q @ H.T
            R = H @ R
        
        return Q, R

    def _eigen_decomposition(self, A, max_iter=100):
        A_k = A
        Q_k = np.eye( A.shape[1] )
        
        for k in range(max_iter):
            Q, R = qr_decomposition(A_k)
            Q_k = Q_k @ Q
            A_k = R @ Q

        eigenvalues = np.diag(A_k)
        eigenvectors = Q_k
        return eigenvalues, eigenvectors

    def fit(self, X):
        n, m = X.shape
        
        # subtract off the mean to center the data.
        self.mu = X.mean(axis=0)
        X = X - self.mu
        
        # whiten if necessary
        if self.whiten:
            self.std = X.std(axis=0)
            X = X / self.std
        
        # Eigen Decomposition of the covariance matrix
        C = X.T @ X / (n-1)
        self.eigenvalues, self.eigenvectors = self._eigen_decomposition(C)
        
        # truncate the number of components if doing dimensionality reduction
        if self.n_components is not None:
            self.eigenvalues = self.eigenvalues[0:self.n_components]
            self.eigenvectors = self.eigenvectors[:, 0:self.n_components]
        
        # the QR algorithm tends to puts eigenvalues in descending order 
        # but is not guarenteed to. To make sure, we use argsort.
        descending_order = np.flip(np.argsort(self.eigenvalues))
        self.eigenvalues = self.eigenvalues[descending_order]
        self.eigenvectors = self.eigenvectors[:, descending_order]

        return self

    def transform(self, X):
        X = X - self.mu
        
        if self.whiten:
            X = X / self.std
        
        return X @ self.eigenvectors
    
    @property
    def proportion_variance_explained(self):
        return self.eigenvalues / np.sum(self.eigenvalues)

if __name__ == "__main__":
    # Loading data
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values # incl 2 for 3D
    X = StandardScaler().fit_transform(X) # Normalize data

    # algorithm
    pca = PCA(whiten=True)
    pca.fit(X)

    # assign PCA to data
    pca_T = pca.transform(X)
    X_new = pca_T# pca.inverse_transform(pca_T)

    # visualization
    plt.scatter(
        X[:, 0], X[:, 1], 
        alpha=0.2, 
        color="b", 
        label="input"
    )
    plt.scatter(
        X_new[:, 0], X_new[:, 1]*0, 
        alpha=0.8, 
        color="b", 
        label="PCA output"
    )

    plt.grid()  # define raster 
    plt.legend()# define label border
    plt.show()  # open window













