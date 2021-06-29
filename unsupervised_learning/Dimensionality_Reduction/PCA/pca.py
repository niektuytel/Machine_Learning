import numpy as np# for math
import pandas as pd# for reading csv file
import matplotlib.pyplot as plt# for visualization
from sklearn.decomposition import PCA# import algorithm
from sklearn.preprocessing import StandardScaler# scale data to a same value factor

if __name__ == "__main__":
    # Loading data
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values # incl 2 for 3D
    X = StandardScaler().fit_transform(X) # Normalize data

    # algorithm
    pca = PCA(n_components=1)
    pca.fit(X)

    # assign PCA to data
    pca_T = pca.transform(X)
    X_new = pca.inverse_transform(pca_T)

    # visualization
    plt.scatter(
        X[:, 0], X[:, 1], 
        alpha=0.2, 
        color="b", 
        label="input"
    )
    plt.scatter(
        X_new[:, 0], X_new[:, 1], 
        alpha=0.8, 
        color="b", 
        label="PCA output"
    )

    plt.grid()  # define raster 
    plt.legend()# define label border
    plt.show()  # open window
