import numpy as np # for math methods
import pandas as pd # for reading csv file
import matplotlib.pyplot as plt # for data visualization 
from sklearn.cluster import AgglomerativeClustering # get the clustering model
from scipy.cluster.hierarchy import dendrogram, linkage # dendogram visualization etc.

# Loading data
dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# define the model
model = AgglomerativeClustering(n_clusters=5)

# assign a cluster to each example
yhat = model.fit_predict(X)

# retrieve unique clusters
clusters = np.unique(yhat)

# create scatter plot for samples from each cluster
for cluster in clusters:

	# get row indexes for samples with this cluster
	row_idx = np.where(yhat == cluster)
	
    # create scatter of these samples
	plt.scatter(X[row_idx, 0], X[row_idx, 1])

# show the plot
plt.show()