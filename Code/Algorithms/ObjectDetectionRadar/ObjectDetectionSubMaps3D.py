import os
import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from DataProcessing.radar_utilsProcessing import *
from DataProcessing.radar_utilsPlot import *

SAFETY_BOX_CENTER = [0, 2, 0]  # Center position (X, Y, Z)
SAFETY_BOX_SIZE = [2, 9, 2]   # Width, Height, Depth

# Create a new dictionary with frame numbers and coordinates + Doppler speed
def extract_coordinates_with_doppler(frames_data, y_threshold=None, z_threshold=None, doppler_threshold=None):
    coordinates_dict = {}

    for frame_num, frame_content in frames_data.items():
        # Extract TLV data
        tlvs = frame_content.get("TLVs", [])

        # Find Type 1 data (Detected Points)
        points = []
        for tlv in tlvs:
            if "Type 1 Data" in tlv:  # Look for Type 1 Data
                points = tlv["Type 1 Data"]
                break  # Assume only one Type 1 entry per frame

        # Create a list of dictionaries with required fields and filters
        coordinates = []
        for point in points:
            # Apply threshold filters
            if y_threshold is not None and point["Y [m]"] < y_threshold:
                continue  # Skip if Y is below the threshold

            if z_threshold is not None and not (z_threshold[0] <= point["Z [m]"] <= z_threshold[1]):
                continue  # Skip if Z is outside the range

            if doppler_threshold is not None and abs(point["Doppler [m/s]"]) <= doppler_threshold:
                continue  # Skip if Doppler speed is below the threshold

            # Add the point to the list if it passes all filters
            coordinates.append([
                point["X [m]"],
                point["Y [m]"],
                point["Z [m]"],
                point["Doppler [m/s]"]
            ])

        # Add the filtered coordinates list to the dictionary
        if coordinates:  # Only add frames with valid points
            coordinates_dict[frame_num] = np.array(coordinates)

    return coordinates_dict

# Submap aggregation function
def aggregate_submap(frames_data, start_frame, num_frames=10):
    aggregated_points = []
    for frame in range(start_frame, start_frame + num_frames):
        if frame in frames_data:
            aggregated_points.extend(frames_data[frame])
    return np.array(aggregated_points)

# -------------------------------
# FUNCTION: Cluster Points
# -------------------------------
def cluster_points(points, eps=1.0, min_samples=2):
    """ Perform DBSCAN clustering and filter clusters based on priorities. """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])  # Use X, Y, Z for clustering
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
        if size >= 10:
            priority = 1
        elif size < 10 and size >= 5:
            priority = 2
        elif size < 5:
            priority = 3
        else:
            priority = 4
        clusters[cluster_id] = {'centroid': centroid, 'priority': priority, 'points': cluster_points}

    return clusters

# -------------------------------
# FUNCTION: Plot Clusters
# -------------------------------
def plot_clusters_3d(clusters, ax):
    """ Plot clusters and visualize bounding boxes and priorities in 3D. """
    for cid, cluster in clusters.items():
        centroid = cluster['centroid']
        priority = cluster['priority']
        ax.scatter(cluster['points'][:, 0], cluster['points'][:, 1], cluster['points'][:, 2], label=f"Cluster {cid}")
        ax.scatter(centroid[0], centroid[1], centroid[2], c='black', marker='x')  # Centroid marker

        # Draw bounding box (cube)
        min_vals = np.min(cluster['points'], axis=0)
        max_vals = np.max(cluster['points'], axis=0)
        r = [[min_vals[0], max_vals[0]], [min_vals[1], max_vals[1]], [min_vals[2], max_vals[2]]]
        vertices = [
            [r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][0]], [r[0][0], r[1][1], r[2][0]],
            [r[0][0], r[1][0], r[2][1]], [r[0][1], r[1][0], r[2][1]],
            [r[0][1], r[1][1], r[2][1]], [r[0][0], r[1][1], r[2][1]]
        ]
        edges = [
            [vertices[i] for i in [0, 1, 2, 3]],
            [vertices[i] for i in [4, 5, 6, 7]],
            [vertices[i] for i in [0, 1, 5, 4]],
            [vertices[i] for i in [2, 3, 7, 6]],
            [vertices[i] for i in [1, 2, 6, 5]],
            [vertices[i] for i in [4, 7, 3, 0]]
        ]
        if(priority == 1):
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='red'))
        elif (priority == 2):
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='yellow'))
        elif (priority == 3):
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='green'))
        else:
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='gray'))

        monitor_safety_box(clusters, ax, SAFETY_BOX_CENTER, SAFETY_BOX_SIZE)


    # Draw fixed rectangle (vehicle) at origin
    vertices = [[-0.5, -0.9, 0], [0.5, -0.9, 0], [0.5, 0.9, 0], [-0.5, 0.9, 0],
                [-0.5, -0.9, 0.5], [0.5, -0.9, 0.5], [0.5, 0.9, 0.5], [-0.5, 0.9, 0.5]]
    edges = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [4, 7, 3, 0]]
    ]
    ax.add_collection3d(Poly3DCollection(edges, alpha=0.3, facecolor='cyan'))
# -------------------------------
# FUNCTION: Safety Box
# -------------------------------
def monitor_safety_box(clusters, ax, box_center, box_size):
    """ Monitor clusters for collisions with a static safety box and trigger warnings. """
    # Calculate box boundaries
    box_min = np.array(box_center) - np.array(box_size) / 2
    box_max = np.array(box_center) + np.array(box_size) / 2

    # Draw the safety box in blue
    vertices = [
        [box_min[0], box_min[1], box_min[2]], [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_max[1], box_min[2]], [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]], [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_max[2]], [box_min[0], box_max[1], box_max[2]]
    ]
    edges = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [4, 7, 3, 0]]
    ]
    ax.add_collection3d(Poly3DCollection(edges, alpha=0.3, facecolor='blue'))

    # Check if any point in the cluster lies within the safety box (ignoring Doppler values)
    for cid, cluster in clusters.items():
        points_xyz = cluster['points'][:, :3]  # Only X, Y, Z coordinates
        priority = cluster['priority']
        inside_box = np.all((points_xyz >= box_min) & (points_xyz <= box_max), axis=1)
        if np.any(inside_box):
            print(f"[!] WARNING: Cluster {cid} in safety zone!")
            print(f"[!] WARNING: Cluster with priority: {priority} in safety zone!")


# Interactive slider-based visualization
def plot_with_slider(frames_data, num_frames=10):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set initial view angle (top-down)
    ax.view_init(elev=90, azim=-90)

    # Slider for frame selection
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 1, len(frames_data) - num_frames + 1, valinit=1, valstep=1)

    def update(val):
        start_frame = int(slider.val)
        submap = aggregate_submap(frames_data, start_frame, num_frames)

        # Perform clustering
        clusters = cluster_points(submap)

        ax.clear()
        plot_clusters_3d(clusters, ax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 15)
        ax.set_zlim(-0.30, 10)
        ax.set_title(f"Clusters (Frames {start_frame} to {start_frame + num_frames - 1})")
        #ax.legend()
        plt.draw()

    slider.on_changed(update)
    update(1)  # Initial plot
    plt.show()

# Example Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "DynamicMonitoring", "30fps_straight_3targets_2_log_2024-12-16.csv")
file_path = os.path.normpath(os.path.join(script_dir, relative_path))

y_threshold = 0.0
z_threshold = (-0.30, 2.0)
doppler_threshold = 0.0

print(f"Processing file: {file_path}")
frames_data = process_log_file(file_path, snr_threshold=12, z_min=-2.0, z_max=2.0, doppler_threshold=0.1)

frames_data = extract_coordinates_with_doppler(frames_data, y_threshold, z_threshold, doppler_threshold)

# Plot with slider and clustering
plot_with_slider(frames_data, num_frames=10)
