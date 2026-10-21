
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Variance Retained:",
      pca.explained_variance_ratio_.sum())

print("First Five Transformed Rows:")
print(X_pca[:5])
