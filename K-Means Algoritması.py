import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeansCluster:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Veri noktalarından rastgele küme merkezleri seçme
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        while True:
            # Her bir veri noktasını en yakın küme merkezine atama
            self.labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
            
            # Yeni küme merkezlerini hesaplama
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Eğer küme merkezleri değişmiyorsa döngüyü sonlandır
            if np.allclose(new_centroids, self.centroids):
                break
                
            self.centroids = new_centroids.copy()

        return self

# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Kendi KMeans sınıfımızı kullanarak kümeleme yapma
kmeans_custom = KMeansCluster(n_clusters=4)
kmeans_custom.fit(X)

# Küme merkezlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for i in range(kmeans_custom.n_clusters):
    cluster_points = X[kmeans_custom.labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

plt.scatter(kmeans_custom.centroids[:, 0], kmeans_custom.centroids[:, 1], marker='o', s=200, c='black', edgecolors='w')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()