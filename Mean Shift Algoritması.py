import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class CustomMeanShift:
    def __init__(self, bandwidth=1.0, min_distance=0.001):
        self.bandwidth = bandwidth
        self.min_distance = min_distance
        self.cluster_centers = None
        self.labels = None

    def fit(self, X):
        self.cluster_centers = []
        self.labels = np.zeros(len(X), dtype=int)

        for idx in range(len(X)):
            if self.labels[idx] != 0:
                continue
            
            point = X[idx]
            in_bandwidth = np.linalg.norm(X - point, axis=1) <= self.bandwidth
            in_bandwidth[idx] = False  # Kendi noktamızı dışarıda bırak

            cluster_center = point

            while True:
                prev_center = cluster_center
                points_in_cluster = X[in_bandwidth]

                distances = np.linalg.norm(points_in_cluster - prev_center, axis=1)
                within_bandwidth = distances <= self.bandwidth

                cluster_center = np.mean(points_in_cluster[within_bandwidth], axis=0)

                if np.linalg.norm(cluster_center - prev_center) < self.min_distance:
                    break

            self.cluster_centers.append(cluster_center)
            self.labels[in_bandwidth] = len(self.cluster_centers)

        self.cluster_centers = np.array(self.cluster_centers)

        return self


# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Kendi CustomMeanShift sınıfımızı kullanarak kümeleme yapma
custom_mean_shift = CustomMeanShift(bandwidth=0.8, min_distance=0.01)
custom_mean_shift.fit(X)

# Küme merkezlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for i in range(len(custom_mean_shift.cluster_centers)):
    cluster_points = X[custom_mean_shift.labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

plt.scatter(custom_mean_shift.cluster_centers[:, 0], custom_mean_shift.cluster_centers[:, 1], marker='o', s=200, c='black', edgecolors='w', label='Centroids')
plt.title('Custom Mean Shift Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()