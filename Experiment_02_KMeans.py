
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [1,2],[1,3],[2,2],[2,3],
    [8,8],[9,8],[8,9],[9,9],
    [4,7],[5,7],[4,8],[5,8]
])

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(model.cluster_centers_)

plt.scatter(X[:,0], X[:,1], c=labels, s=80)
plt.scatter(model.cluster_centers_[:,0],
            model.cluster_centers_[:,1],
            marker="X", s=200, label="Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
