from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# Load Iris dataset
iris = load_iris()
X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
y = iris.target  # 3 species (0=setosa, 1=versicolor, 2=virginica)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_95 = PCA(n_components=0.95)  # Reduce to 2 components
X_pca_95 = pca_95.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:", pca_95.explained_variance_ratio_) # PC1=72.9%, PC2=22.9% (Total: 95.8%)

fig = plt.figure(figsize=(10, 7))

colors = ['navy', 'turquoise', 'darkorange']
species = iris.target_names

for i, color in enumerate(colors):
    plt.scatter(
        X_pca_95[y == i, 0],  # PC1
        X_pca_95[y == i, 1],  # PC2
        color=color,
        label=species[i],

    )

plt.title("PCA of Iris Dataset (2D)")
plt.xlabel("Principal Component 1 (72.9% variance)")
plt.ylabel("Principal Component 2 (22.9% variance)")
plt.legend()
plt.grid()
plt.show()

print("\nResults for PCA(n_components=0.95):")
print("Number of components selected:", pca_95.n_components_)
print("Explained Variance Ratio:", pca_95.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca_95.explained_variance_ratio_))