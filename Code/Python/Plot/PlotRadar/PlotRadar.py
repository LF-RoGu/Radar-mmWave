import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
import numpy as np
import time

# Constants
FILE_PATH = r"\\wsl$\Ubuntu\root\.vs\Radar\out\build\linux-debug\Radar\OutputFile\detected_points.csv"
AXIS_LIMIT = 3  # Meters
DOPPLER_THRESHOLD = 0.1  # Doppler threshold to separate stationary and moving points
DISTANCE_THRESHOLD = 0.5  # Distance between objects for clustering (meters)
PLOT_UPDATE_DELAY = 0.001  # Delay for plot updates in seconds
BOUNDING_BOX_ALPHA = 0.2  # Transparency for bounding boxes

# Initialize plots
FIG_SIZE = (12, 6)
fig = plt.figure(figsize=FIG_SIZE)

# Create subplots
ax_stationary = fig.add_subplot(121, projection='3d', title="Stationary Objects")
ax_moving = fig.add_subplot(122, projection='3d', title="Moving Objects")

def draw_bounding_box(ax, points, color='blue'):
    """Draw a bounding box around the points."""
    if points.empty:
        return
    # Compute bounding box
    x_min, x_max = points['x'].min(), points['x'].max()
    y_min, y_max = points['y'].min(), points['y'].max()
    z_min, z_max = points['z'].min(), points['z'].max()

    # Define vertices of the box
    vertices = [
        [x_min, y_min, z_min],
        [x_min, y_max, z_min],
        [x_max, y_max, z_min],
        [x_max, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max],
        [x_max, y_min, z_max],
    ]

    # Define the edges of the box
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Left face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Right face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Front face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Back face
    ]

    # Add the box to the plot
    box = Poly3DCollection(edges, alpha=BOUNDING_BOX_ALPHA, edgecolor=color, facecolor=color)
    ax.add_collection3d(box)

def cluster_objects(points, distance_threshold):
    """Cluster points using DBSCAN and return clustered points."""
    if points.empty:
        return points, None

    # Convert points to a numpy array
    coords = points[['x', 'y', 'z']].values

    # Apply DBSCAN for clustering
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(coords)

    # Add cluster labels to the points DataFrame
    points['cluster'] = clustering.labels_
    return points, clustering.labels_

while True:
    try:
        # Read and process the file
        data = pd.read_csv(FILE_PATH)

        # Separate stationary and moving points
        stationary_points = data[abs(data['doppler']) <= DOPPLER_THRESHOLD]
        moving_points = data[abs(data['doppler']) > DOPPLER_THRESHOLD]

        # Clear previous plots
        ax_stationary.cla()
        ax_moving.cla()

        # Cluster and plot stationary objects
        if not stationary_points.empty:
            clustered_stationary, _ = cluster_objects(stationary_points, DISTANCE_THRESHOLD)
            for cluster_id, cluster_points in clustered_stationary.groupby('cluster'):
                ax_stationary.scatter(
                    cluster_points['x'], cluster_points['y'], cluster_points['z'],
                    label=f"Cluster {cluster_id}", marker='o'
                )
                draw_bounding_box(ax_stationary, cluster_points, color='green')
        ax_stationary.set_xlabel('X Coordinate (m)')
        ax_stationary.set_ylabel('Y Coordinate (m)')
        ax_stationary.set_zlabel('Z Coordinate (m)')
        ax_stationary.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_title("Stationary Objects")
        ax_stationary.legend(loc='upper left', fontsize=8)

        # Cluster and plot moving objects
        if not moving_points.empty:
            clustered_moving, _ = cluster_objects(moving_points, DISTANCE_THRESHOLD)
            for cluster_id, cluster_points in clustered_moving.groupby('cluster'):
                ax_moving.scatter(
                    cluster_points['x'], cluster_points['y'], cluster_points['z'],
                    label=f"Cluster {cluster_id}", marker='o'
                )
                draw_bounding_box(ax_moving, cluster_points, color='red')
        ax_moving.set_xlabel('X Coordinate (m)')
        ax_moving.set_ylabel('Y Coordinate (m)')
        ax_moving.set_zlabel('Z Coordinate (m)')
        ax_moving.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_title("Moving Objects")
        ax_moving.legend(loc='upper left', fontsize=8)

        # Pause to update the plots
        plt.pause(PLOT_UPDATE_DELAY)  # Pause for a short time to allow the plots to update

    except Exception as e:
        print(f"Error reading file: {e}")
