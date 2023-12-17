import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class CustomSTING:
    def __init__(self, n_clusters, grid_size):
        self.n_clusters = n_clusters
        self.grid_size = grid_size
        self.labels = None

    def fit(self, X):
        # Veriyi ızgara üzerinde bölmek için sınırları belirleme
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

        x_step = (x_max - x_min) / (self.grid_size - 1)
        y_step = (y_max - y_min) / (self.grid_size - 1)

        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Her bir veri noktasını ızgaraya atama
        for i, point in enumerate(X):
            x_index = int((point[0] - x_min) / x_step)
            y_index = int((point[1] - y_min) / y_step)

            if x_index == self.grid_size:
                x_index -= 1
            if y_index == self.grid_size:
                y_index -= 1

            grid[x_index, y_index] += 1

        # Yoğunluk tabanlı olarak kümeleri belirleme
        self.labels = np.zeros(len(X), dtype=int)

        threshold = np.percentile(grid, 75)  # Yoğunluk eşik değeri
        for i, point in enumerate(X):
            x_index = int((point[0] - x_min) / x_step)
            y_index = int((point[1] - y_min) / y_step)

            if x_index == self.grid_size - 1:
                x_index -= 1
            if y_index == self.grid_size - 1:
                y_index -= 1

            if grid[x_index, y_index] > threshold:
                self.labels[i] = 1  # Yoğunluğa göre kümeleme

        return self


# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Kendi CustomSTING sınıfımızı kullanarak kümeleme yapma
custom_sting = CustomSTING(n_clusters=2, grid_size=10)
custom_sting.fit(X)

# Küme etiketlerini al
sting_labels = custom_sting.labels

# Küme etiketlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for i in range(2):
    cluster_points = X[sting_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

plt.title('Custom STING-like Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()