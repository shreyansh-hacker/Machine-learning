"""
================================================================================
Module Name: K_means_clustering.py
Description: Unsupervised Clustering using Scikit-Learn's KMeans.
Methodology:
  1. Generate synthetic 2D classification dataset.
  2. Fit KMeans clustering with k=3 cluster centers.
  3. Retrieve cluster label assignments and cluster centroids.
  4. Plot the clusters and label their respective centroids.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# Set seed
np.random.seed(42)

# Generate dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=42
)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ==========================
# K-Means Clustering
# ==========================

k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Train model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Centroids
centroids = kmeans.cluster_centers_

# ==========================
# Plot
# ==========================

plt.figure(figsize=(8,6))

cluster_colors = plt.cm.tab10(np.linspace(0, 1, k))
centroid_colors = plt.cm.Set1(np.linspace(0, 1, k))

for i in range(k):

    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        color=cluster_colors[i],
        label=f"Cluster {i}",
        alpha=0.7
    )

    plt.scatter(
        centroids[i, 0],
        centroids[i, 1],
        color=centroid_colors[i],
        marker='X',
        s=250,
        edgecolors='black',
        label=f"Centroid {i}"
    )

plt.title("K-Means Clustering (sklearn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()