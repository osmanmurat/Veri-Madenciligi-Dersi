import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class DIANACluster:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = []

    def fit(self, X):
        self.clusters.append(X)

        while len(self.clusters) < self.n_clusters:
            max_cluster_index = self.find_max_cluster()
            max_cluster = self.clusters[max_cluster_index]

            furthest_point_index = self.find_furthest_point(max_cluster)
            furthest_point = max_cluster[furthest_point_index]

            cluster1, cluster2 = self.split_cluster(max_cluster, furthest_point_index)
            
            self.clusters.pop(max_cluster_index)
            self.clusters.append(cluster1)
            self.clusters.append(cluster2)

        return self

    def find_max_cluster(self):
        sizes = [len(cluster) for cluster in self.clusters]
        return np.argmax(sizes)

    def find_furthest_point(self, cluster):
        distances = np.linalg.norm(cluster - np.mean(cluster, axis=0), axis=1)
        return np.argmax(distances)

    def split_cluster(self, cluster, index):
        return np.delete(cluster, index, axis=0), cluster[index, np.newaxis]

# Veri setini oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Kendi DIANA sınıfımızı kullanarak kümeleme yapma
diana_custom = DIANACluster(n_clusters=4)
diana_custom.fit(X)

# Küme merkezlerini ve noktaları görselleştirme
plt.figure(figsize=(8, 6))

for cluster in diana_custom.clusters:
    plt.scatter(cluster[:, 0], cluster[:, 1])

plt.title('DIANA Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()