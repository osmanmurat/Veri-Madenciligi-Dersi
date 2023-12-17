import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class CustomBIRCH:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        self.centroids = []
        self.labels = np.zeros(len(X), dtype=int)
        cluster_counter = 0

        for idx, x in enumerate(X):
            if len(self.centroids) == 0:
                self.centroids.append(x)
                self.labels[idx] = cluster_counter
                cluster_counter += 1
            else:
                closest_centroid_idx = self.find_closest_centroid(x)
                if np.linalg.norm(x - self.centroids[closest_centroid_idx]) < self.threshold:
                    self.update_centroid(closest_centroid_idx, x)
                    self.labels[idx] = closest_centroid_idx
                else:
                    if len(self.centroids) < self.branching_factor:
                        self.centroids.append(x)
                        self.labels[idx] = cluster_counter
                        cluster_counter += 1
                    else:
                        # Doğrusal arama yerine daha etkili bir yöntem kullanılabilir
                        closest_centroid_idx = self.find_closest_centroid(x)
                        self.update_centroid(closest_centroid_idx, x)
                        self.labels[idx] = closest_centroid_idx

        return self

    def find_closest_centroid(self, x):
        distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def update_centroid(self, idx, x):
        self.centroids[idx] = (self.centroids[idx] + x) / 2.0

# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Kendi CustomBIRCH sınıfımızı kullanarak kümeleme yapma
custom_birch = CustomBIRCH(threshold=0.5, branching_factor=50, n_clusters=4)
custom_birch.fit(X)

# Küme etiketlerini al
birch_labels = custom_birch.labels

# Küme etiketlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for label in np.unique(birch_labels):
    plt.scatter(X[birch_labels == label][:, 0], X[birch_labels == label][:, 1], label=f'Cluster {label}')

plt.title('Custom BIRCH Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()