import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the mall dataset using pandas
dataset = pd.read_csv('Mall_Customers.csv')

# Selecting the annual income and spending score columns for clustering
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph to choose the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Based on the Elbow Method, it appears that 5 clusters is the optimal choice

# Apply K-Means clustering with the optimal number of clusters (5)
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Use PCA for dimensionality reduction and better visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the clusters with PCA-transformed data
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for cluster_id in range(5):
    plt.scatter(X_pca[y_kmeans == cluster_id, 0], X_pca[y_kmeans == cluster_id, 1],
                s=100, c=colors[cluster_id], label=labels[cluster_id])

# Plot cluster centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Customer Segmentation using K-Means Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the clusters using Silhouette Score
silhouette_avg = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.4f}")

