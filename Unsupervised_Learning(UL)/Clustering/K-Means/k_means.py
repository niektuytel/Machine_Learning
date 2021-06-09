import numpy as np # for the math
import pandas as pd # for read csv file
import matplotlib.pyplot as plt # for visualizing data
from sklearn.cluster import KMeans, MiniBatchKMeans # for the algorithm

# Loading data
dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# define the model
model = KMeans(n_clusters=5)
# model = MiniBatchKMeans(n_clusters=5)

# fit the data
model.fit(X)

# assign a cluster to each example
yhat = model.predict(X)

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