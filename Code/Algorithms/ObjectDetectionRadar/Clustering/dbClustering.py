import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle, Ellipse

# -------------------------------
# FUNCTION: Cluster Points
# -------------------------------
def cluster_points(points):
    """ Perform DBSCAN clustering and filter clusters based on priorities. """
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(points[:, :2])  # Use only X and Y for clustering
    labels = dbscan.labels_

    clusters = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Ignore noise
            continue
        cluster_points = points[labels == cluster_id]
        size = len(cluster_points)

        # Ignore clusters with <3 points (Priority 3)
        if size < 3:
            continue

        # Store centroid and priority
        centroid = np.mean(cluster_points, axis=0)
        priority = 1 if size >= 7 else 2  # Priority 1 for 7+, Priority 2 for 3-6
        clusters[cluster_id] = {'centroid': centroid, 'priority': priority, 'points': cluster_points}

    return clusters

# -------------------------------
# FUNCTION: Plot Clusters
# -------------------------------
def plot_clusters(clusters, ax):
    """ Plot clusters and visualize bounding boxes and priorities. """
    for cid, cluster in clusters.items():
        centroid = cluster['centroid']
        ax.scatter(cluster['points'][:, 0], cluster['points'][:, 1], label=f"Cluster {cid}")
        ax.scatter(centroid[0], centroid[1], c='black', marker='x')  # Centroid marker

        # Draw bounding box
        width = np.max(cluster['points'][:, 0]) - np.min(cluster['points'][:, 0])
        height = np.max(cluster['points'][:, 1]) - np.min(cluster['points'][:, 1])
        ax.add_patch(Rectangle(
            (centroid[0] - width / 2, centroid[1] - height / 2), width, height,
            fill=False, edgecolor='purple', linewidth=1.5
        ))

        # Add priority labels
        ax.text(centroid[0], centroid[1], f"P{cluster['priority']}", color='red')