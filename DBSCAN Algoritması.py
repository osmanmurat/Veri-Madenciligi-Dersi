import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        cluster_label = 0
        self.labels = np.zeros(len(X), dtype=int)

        for idx in range(len(X)):
            if self.labels[idx] != 0:
                continue

            neighbors = self.find_neighbors(X, idx)

            if len(neighbors) < self.min_samples:
                self.labels[idx] = -1  # Gürültü noktası olarak işaretle
            else:
                cluster_label += 1
                self.expand_cluster(X, idx, neighbors, cluster_label)

        return self

    def find_neighbors(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def expand_cluster(self, X, idx, neighbors, cluster_label):
        self.labels[idx] = cluster_label

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_label
            elif self.labels[neighbor_idx] == 0:
                self.labels[neighbor_idx] = cluster_label
                new_neighbors = self.find_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            i += 1

# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Kendi CustomDBSCAN sınıfımızı kullanarak kümeleme yapma
custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=5)
custom_dbscan.fit(X)

# Küme etiketlerini al
dbscan_labels = custom_dbscan.labels

# Küme etiketlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for label in np.unique(dbscan_labels):
    if label == -1:
        plt.scatter(X[dbscan_labels == label][:, 0], X[dbscan_labels == label][:, 1], label='Noise', color='black')
    else:
        plt.scatter(X[dbscan_labels == label][:, 0], X[dbscan_labels == label][:, 1], label=f'Cluster {label}')

plt.title('Custom DBSCAN Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()