import numpy as np # for the math
import pandas as pd # for read csv file
import matplotlib.pyplot as plt # for visualizing data
import random # random 

# type of kernel we use in this sample
def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

class MeanShift():
    STOP_THRESHOLD = 1e-3
    CLUSTER_THRESHOLD = 1e-1

    def __init__(self, bandwidth=20, kernel=gaussian_kernel):
        self.bandwidth=bandwidth
        self.kernel=kernel

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _shift_point(self, point, points):
        shift_x = 0.0
        shift_y = 0.0
        scale = 0.0

        for p in points:
            dist = self._distance(point, p)

            # shift point location based on the distance weight
            weight = self.kernel(dist, self.bandwidth)
            shift_x += p[0] * weight
            shift_y += p[1] * weight
            scale += weight

        # remove scale (weights) value 
        shift_x = shift_x / scale
        shift_y = shift_y / scale

        new_point = [shift_x, shift_y]
        distance = self._distance(new_point, point)
        return (distance, new_point)

    def _cluster_points(self, points):
        clusters = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(clusters) == 0):
                clusters.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = self._distance(point, center)
                    if(dist < CLUSTER_THRESHOLD):
                        clusters.append(cluster_centers.index(center))

                if(len(clusters) < i + 1):
                    clusters.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1

        return clusters

    def fit_predict(self, X):
        shift_points = X.copy()
        shifting = [True] * X.shape[0]

        # prevent infinity loop, 
        prev_dist = 0
        threshold = 0

        while True:
            # update points locations 
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue

                # update point location
                distance, shift_points[i] = self._shift_point(shift_points[i], X)
                max_dist = max(max_dist, distance)
                shifting[i] = (distance > STOP_THRESHOLD)

            if threshold >= 10:
                break
            elif prev_dist >= max_dist:
                threshold += 1

            prev_dist = max_dist


        
        clusters = self._cluster_points(shift_points.tolist())
        return clusters

if __name__ == "__main__":
    # Loading data
    dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values

    # define model
    model = MeanShift(bandwidth=23)

    # assign a cluster to each example
    yhat = model.fit_predict(X)

    # retrieve unique clusters
    clusters = np.unique(yhat)

    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])

    # show the plot
    plt.show()
