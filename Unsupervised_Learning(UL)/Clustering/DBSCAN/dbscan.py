import numpy as np # for the math
import pandas as pd # for read csv file
import matplotlib.pyplot as plt # for visualizing data
from sklearn.cluster import DBSCAN # for the algorithm
from sklearn.preprocessing import StandardScaler# for rescale our data 

# Loading data
dataset = pd.read_csv("../../../_EXTRA/data/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# normalize dataset (for the epsilon)
X = StandardScaler().fit_transform(X)

# define the model
model = DBSCAN(eps=0.3, min_samples=5)

# fit model and predict clusters
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