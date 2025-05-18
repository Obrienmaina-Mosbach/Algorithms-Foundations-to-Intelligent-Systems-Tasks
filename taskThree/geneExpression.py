

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

geneData = pd.read_csv('../taskThree/Data/Spellman.csv')
print(geneData.head())

X = geneData.drop(columns=["time"])
y = geneData["time"]

X = X[:2500]
y = y[:2500]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='y', cmap='viridis', edgecolor='k', s=50)
plt.title('PCA of Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Time')
plt.show()


kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(X_pca)

print(pca.explained_variance_ratio_)

# Plot the KMeans results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
plt.title('KMeans Clustering of PCA Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Plot the silhouette score
plt.figure(figsize=(8, 6))
plt.bar(range(len(kmeans.labels_)), silhouette_score(X_pca, kmeans.labels_, metric='euclidean'))
plt.title('Silhouette Score for KMeans Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Silhouette Score')
plt.show()

# Plot cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o')
plt.axhline(y=0.90, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Variance Explained')
plt.grid(True)
plt.show()