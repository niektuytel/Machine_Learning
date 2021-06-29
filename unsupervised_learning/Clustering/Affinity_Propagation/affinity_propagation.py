import matplotlib.pyplot as plt# for visualization
import pandas as pd# for read csv file
from itertools import cycle# for cycle of data
from sklearn.cluster import AffinityPropagation# for clustering model

# Loading data
dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
matrix = dataset.iloc[:, [3, 4]].values

# Compute Affinity Propagation
model = AffinityPropagation(
      preference=-9200
).fit(matrix)

labels = model.labels_
cluster_centers_indices = model.cluster_centers_indices_
clusters_amount = len(cluster_centers_indices)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(clusters_amount), colors):
    class_members = labels == k
    cluster_center = matrix[cluster_centers_indices[k]]

    plt.plot(matrix[class_members, 0], matrix[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in matrix[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % clusters_amount)
plt.show()