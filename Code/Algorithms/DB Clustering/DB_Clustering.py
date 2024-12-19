import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate synthetic radar-like data (x, y points in meters)
np.random.seed(42)
# Simulated cluster for a vehicle
vehicle = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
# Simulated cluster for a pedestrian
pedestrian = np.random.normal(loc=[10, 10], scale=0.2, size=(30, 2))
# Random noise (outliers)
noise = np.random.uniform(low=0, high=15, size=(20, 2))

# Combine all data points
data = np.vstack([vehicle, pedestrian, noise])

# Plot the raw data without clustering
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c="gray", s=50, alpha=0.7, label="Raw Data")
plt.title("Raw Radar Data Without Clustering")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.legend()
plt.grid()
plt.show()

# Apply DBSCAN clustering
eps = 1.0  # Maximum distance between points in a cluster (in meters)
min_samples = 5  # Minimum points to form a cluster
db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(data)

# Plot the clustered data
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        # Noise points
        color = 'black'
        label_name = 'Noise'
    else:
        # Cluster points
        color = plt.cm.tab10(label / len(unique_labels))
        label_name = f'Cluster {label}'
    cluster_points = data[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=label_name, s=50)

plt.title('DBSCAN Clustering of Radar Data')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.legend()
plt.grid()
plt.show()
