"""!
@file dbCluster.py
@brief Implements DBSCAN clustering for radar point cloud data.

@details This module applies the DBSCAN clustering algorithm to radar detections,
computing cluster properties such as centroids, priorities, range, and azimuth.
Clusters are classified based on size, and their properties are returned for further processing.

@defgroup DB_Cluster
@brief Performs DBSCAN clustering on radar point clouds.
@{
"""

from sklearn.cluster import DBSCAN
import numpy as np

# -------------------------------
# CLASS: ClusterProcessor
# -------------------------------
# PURPOSE: Perform DBSCAN clustering and handle various input formats.
# -------------------------------
class ClusterProcessor:
    """!
    @class ClusterProcessor
    @ingroup DB_Cluster
    @brief Processes radar point clouds using DBSCAN clustering.

    @details This class performs clustering on radar detections and computes key properties
    such as centroid position, priority level, Doppler velocity, range, and azimuth angle.
    """
    # -------------------------------
    # FUNCTION: Initialize the clustering processor with DBSCAN parameters
    # -------------------------------
    def __init__(self, eps=1.0, min_samples=6):
        """!
        @brief Initializes the clustering processor with DBSCAN parameters.
        
        @param in eps Maximum distance between samples for DBSCAN to consider them as neighbors.
        @param in min_samples Minimum number of samples required to form a cluster.
        
        @ingroup DB_Cluster
        """
        self.eps = eps  # Maximum distance between samples for DBSCAN
        self.min_samples = min_samples  # Minimum samples for core point

    # -------------------------------
    # FUNCTION: Calculate priority based on cluster size
    # -------------------------------
    def calculate_priority(self, cluster_size):
        """!
        @brief Determines the priority level of a cluster based on its size.

        @param in cluster_size Number of points in the cluster.

        @return Priority level (1 to 3, where 3 is the highest priority).
        
        @ingroup DB_Cluster
        """
        if cluster_size >= 10:
            return 3  # High priority
        elif 5 <= cluster_size < 10:
            return 2  # Medium priority
        elif cluster_size < 5:
            return 1  # Low priority
        return 4  # Fallback priority

    # -------------------------------
    # FUNCTION: Compute centroid of a cluster
    # -------------------------------
    def compute_centroid(self, points):
        """!
        @brief Computes the centroid of a cluster.
        
        @param in points Numpy array of shape (N, 3) representing the cluster points.

        @return Numpy array representing the centroid.
        
        @ingroup DB_Cluster
        """
        return np.mean(points, axis=0)

    # -------------------------------
    # FUNCTION: Compute range and azimuth of the centroid to the origin
    # -------------------------------
    def compute_range_and_azimuth(self, centroid):
        """!
        @brief Computes the range and azimuth angle of the centroid from the origin.

        @param in centroid Numpy array representing the centroid of a cluster.

        @return Tuple containing (range_to_origin, azimuth_to_origin).
        
        @ingroup DB_Cluster
        """
        range_to_origin = np.linalg.norm(centroid[:2])  # X, Y plane distance
        azimuth_to_origin = np.degrees(np.arctan2(centroid[1], centroid[0]))  # Azimuth angle in degrees
        return range_to_origin, azimuth_to_origin

    # -------------------------------
    # FUNCTION: Perform DBSCAN clustering and compute cluster properties
    # -------------------------------
    def cluster_points(self, points):
        """!
        @brief Performs DBSCAN clustering on radar point cloud data.

        @param in points Numpy array of shape (N, 4) containing detected radar points.

        @return data type with:
            - Dictionary of clusters (centroid, priority, points, doppler_avg).
            - List of tuples (cluster_id, range_to_origin, azimuth_to_origin, cluster_points).
        
        @ingroup DB_Cluster
        """
        # STEP 1: Ensure points are in proper format
        if len(points) == 0:
            return {}, []
        
        # STEP 2: Apply DBSCAN clustering algorithm
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points[:, :3])
        labels = dbscan.labels_

        clusters = {}  # Dictionary to store cluster details
        cluster_range_azimuth = []  # List to store range and azimuth data

        # STEP 3: Process each detected cluster
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # Ignore noise points

            # STEP 4: Extract points belonging to the cluster
            cluster_points = points[labels == cluster_id]
            cluster_size = len(cluster_points)

            # STEP 5: Ignore small clusters
            if cluster_size < 3:
                continue

            # STEP 6: Compute cluster properties
            centroid = self.compute_centroid(cluster_points)
            priority = self.calculate_priority(cluster_size)
            doppler_avg = np.mean(cluster_points[:, 3])  # Average Doppler velocity
            range_to_origin, azimuth_to_origin = self.compute_range_and_azimuth(centroid)

            # STEP 7: Store cluster information
            clusters[cluster_id] = {
                'centroid': centroid,
                'priority': priority,
                'points': cluster_points,
                'doppler_avg': doppler_avg     # Average Doppler velocity
            }

            # STEP 8: Save range and azimuth for further processing
            cluster_range_azimuth.append((cluster_id, range_to_origin, azimuth_to_origin, cluster_points))

        return clusters, cluster_range_azimuth
    
## @}  # End of DB_Cluster group