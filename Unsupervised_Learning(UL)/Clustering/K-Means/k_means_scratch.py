"""
Centroid = the center point of grouped data
"""
import numpy as np # for math methods
import pandas as pd # for reading csv file
import matplotlib.pyplot as plt # for data visualization 

class K_Means:
    def __init__(self, k, tol=0.001, max_iterations=300):
        self.k = k
        self.tol = tol
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}

        # set random centroid locations
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            # start with clean empty groups model
            self.classifications = {}

            # set empty groups at given array index 
            for i in range(self.k):
                self.classifications[i] = []

            # add values to there group
            for featureset in data:
                value = self.predict(featureset)
                self.classifications[value].append(featureset)
            
            # set centroids locations on the average of the group value
            # formula in ../../../_EXTRA/images/ml_k_means_clustering_1.png
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # set centroid data & update his optimization
            optimized = True
            for average_centroid in self.centroids:
                prev_centroid = prev_centroids[average_centroid]
                cur_centroid = self.centroids[average_centroid]
                if np.sum((cur_centroid - prev_centroid) / prev_centroid * 100.00) > self.tol:
                    optimized = False

                if optimized:
                    break
    
    # formula in ../../../_EXTRA/images/ml_k_means_clustering_1.png
    def predict(self, dataset):
        distances = [np.linalg.norm(dataset - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

if __name__ == "__main__":
    # the colors that will been used for visualization
    colors = ["red", "blue", "green", "orange", "purple", "brown", "gray"]

    # This is the data we will set into groups
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values

    # Visualize loaded data
    # plt.scatter(X[:, 0], X[:, 1], marker="o")    

    n_clusters=5
    model = K_Means(n_clusters)
    model.fit(X)

    # display all Data
    for classification in model.classifications:
        color = colors[classification]
        for featureset in model.classifications[classification]:
            plt.scatter(
                featureset[0], featureset[1], 
                marker="o", color=color,
            )
    
    # display all Centroids
    plt.scatter(
        model.centroids[:,0], model.centroids[:,1], 
        marker="x", color="black"
    )

    # keep display open 
    plt.show()
