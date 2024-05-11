# K-Means Clustering

## What is K-Means Clustering?
K-Means is one of the most popular methods of clustering. It divides a dataset into a predefined number of clusters (K) by minimizing the mean squared distance between points and their centroids.

## How does it work?
The algorithm begins with a random selection of centroids for the clusters. Then, data points are assigned to the nearest centroid, and the centroids are recalculated as the mean position of points in each cluster. This process repeats until convergence.

## Features:
1. Sensitive to the initial selection of centroids, which can affect the final clustering.
2. Suitable for data with clear clusters but may produce poor results for heterogeneous or overlapping clusters.

## Example:
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Creating artificial data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Applying K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Visualizing clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75)
plt.show()
```

# Hierarchical Clustering

## What is Hierarchical Clustering?
This method builds a hierarchy of clusters, represented as a tree (dendrogram), where each node corresponds to a cluster.

## How does it work?
It starts with each data point as a separate cluster, then at each step, the closest clusters are merged until all points belong to one cluster.

## Features:
1. Allows the creation of dendrograms, visually displaying the data structure and enabling the selection of the optimal number of clusters.
2. Can be agglomerative (bottom-up) or divisive (top-down) depending on the approach.

## Example:
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Generating random data
np.random.seed(0)
X = np.random.rand(10, 2)

# Hierarchical clustering
linked = linkage(X, 'single')

# Plotting dendrogram
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

# DBSCAN Clustering

## What is DBSCAN?
DBSCAN is based on data density. It identifies clusters as areas of high density separated by areas of low density.

## How does it work?
The algorithm starts from a random point and looks for neighbors within a specified radius. If a point has enough neighbors, it becomes part of a cluster. This process spreads from point to point until all points are visited.

## Features:
1. Can handle clusters of arbitrary shapes and detect outliers (noise).
2. Does not require a predefined number of clusters but has parameters such as epsilon radius (eps) and minimum number of points in a cluster (min_samples).

## Example:
```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Creating artificial data
X, _ = make_moons(n_samples=200, shuffle=True, noise=0.1)

# Applying K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Visualizing clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.show()

# Applying DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
clusters = dbscan.fit_predict(X)

# Visualizing clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.show()
```
