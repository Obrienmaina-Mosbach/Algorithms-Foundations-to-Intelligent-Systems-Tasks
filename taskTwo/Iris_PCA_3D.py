from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


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

pca_3d = PCA(n_components=3)  # Reduce to 2 components
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:", pca_3d.explained_variance_ratio_) # PC1=72.9%, PC2=22.9% (Total: 95.8%)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['navy', 'turquoise', 'darkorange']
species = iris.target_names

for i, color in enumerate(colors):
    ax.scatter(
        X_pca_3d[y == i, 0],  # PC1
        X_pca_3d[y == i, 1],  # PC2
        X_pca_3d[y == i, 2],  # PC3
        color=color,
        label=species[i],
        s=50,  # Marker size
        alpha=0.8
    )

ax.set_title("PCA of Iris Dataset (3D)")
ax.set_xlabel("PC1 (72.9% variance)")
ax.set_ylabel("PC2 (22.9% variance)")
ax.set_zlabel("PC3 (3.6% variance)")
ax.legend()
plt.show()