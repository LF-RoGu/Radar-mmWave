import os
import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle

from DataProcessing.radar_utilsProcessing import *
from DataProcessing.radar_utilsPlot import *

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
def cluster_points(points):
    """ Perform DBSCAN clustering and filter clusters based on priorities. """
    dbscan = DBSCAN(eps=1.0, min_samples=2).fit(points[:, :2])  # Use only X and Y for clustering
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
            priority = 3
        elif size < 10 and size >= 5:
            priority = 2
        elif size < 5:
            priority = 1
        else:
            priority = 0
        clusters[cluster_id] = {'centroid': centroid, 'priority': priority, 'points': cluster_points}

    return clusters

# -------------------------------
# FUNCTION: Plot Clusters
# -------------------------------
def plot_clusters(clusters, ax):
    """ Plot clusters and visualize bounding boxes and priorities. """
    for cid, cluster in clusters.items():
        centroid = cluster['centroid']
        priority = cluster['priority']
        points = cluster['points']
        doppler_avg = np.mean(points[:, 3])  # Calculate average Doppler speed

        ax.scatter(cluster['points'][:, 0], cluster['points'][:, 1], label=f"Cluster {cid}")
        ax.scatter(centroid[0], centroid[1], c='black', marker='x')  # Centroid marker

        # Display the average Doppler speed
        ax.text(centroid[0], centroid[1] + 0.2, f"{doppler_avg:.2f} m/s", color='purple')

        # Draw bounding box
        width = np.max(cluster['points'][:, 0]) - np.min(cluster['points'][:, 0])
        height = np.max(cluster['points'][:, 1]) - np.min(cluster['points'][:, 1])
        ax.add_patch(Rectangle(
            (centroid[0] - width / 2, centroid[1] - height / 2), width, height,
            fill=False, edgecolor='purple', linewidth=1.5
        ))

        # Add priority labels
        ax.text(centroid[0], centroid[1] - 0.2, f"P{cluster['priority']}", color='red')

        draw_vehicle_2d(ax, position=(0, 0), size=(1.0, 1.8), color='cyan')

# Interactive slider-based visualization
def plot_with_slider(frames_data, num_frames=10):
    fig = plt.figure(figsize=(8, 8))  # Fixed plot dimensions
    ax = fig.add_subplot(111)

    # Slider for frame selection
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 1, len(frames_data) - num_frames + 1, valinit=1, valstep=1)

    def update(val):
        start_frame = int(slider.val)
        submap = aggregate_submap(frames_data, start_frame, num_frames)

        # Perform clustering
        clusters = cluster_points(submap)

        ax.clear()
        plot_clusters(clusters, ax)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 15)
        ax.set_title(f"Clusters (Frames {start_frame} to {start_frame + num_frames - 1})")
        ax.legend()
        ax.grid(True)
        plt.draw()

    slider.on_changed(update)
    update(1)  # Initial plot
    plt.show()

def draw_vehicle_2d(ax, position=(0, 0), size=(1.0, 1.8), color='cyan'):
    """Draw a 2D representation of the vehicle on the XY plane."""
    # Extract position and size
    x, y = position  # Vehicle center position (X, Y)
    width, height = size  # Dimensions of the vehicle

    # Calculate vertices relative to the center position
    vertices = [
        [x - width / 2, y - height / 2],  # Bottom-left
        [x + width / 2, y - height / 2],  # Bottom-right
        [x + width / 2, y + height / 2],  # Top-right
        [x - width / 2, y + height / 2]   # Top-left
    ]

    # Create a polygon patch
    from matplotlib.patches import Polygon
    vehicle_patch = Polygon(vertices, closed=True, edgecolor='black', facecolor=color, alpha=0.3)

    # Add the patch to the 2D axis
    ax.add_patch(vehicle_patch)

    # Set axis limits and labels
    ax.set_xlim(-10, 10)  # Adjust as needed
    ax.set_ylim(-10, 10)  # Adjust as needed
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)


# Example Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "DynamicMonitoring", "30fps_straight_3x3_log_2024-12-16.csv")
file_path = os.path.normpath(os.path.join(script_dir, relative_path))

y_threshold = 0.0
z_threshold = (0, 3.0)
doppler_threshold = 0.1

print(f"Processing file: {file_path}")
frames_data = process_log_file(file_path, snr_threshold=15, z_min=-0.30, z_max=2.0, doppler_threshold=0.1)

frames_data = extract_coordinates_with_doppler(frames_data, y_threshold, z_threshold, doppler_threshold)

# Plot with slider and clustering
plot_with_slider(frames_data, num_frames=10)
