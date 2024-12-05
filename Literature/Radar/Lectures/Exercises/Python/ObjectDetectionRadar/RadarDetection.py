import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN
import numpy as np

def dbscan_clustering(data, eps=1.0, min_samples=3):
    """
    Perform DBSCAN clustering on the X and Y coordinates from the data.

    Args:
    - data (DataFrame): The data containing 'X [m]' and 'Y [m]' columns.
    - eps (float): The maximum distance between two samples for one to be considered in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - labels (array): Cluster labels for each point. Noise points are labeled as -1.
    """
    points = data[['X [m]', 'Y [m]']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return db.labels_

def visualize_radar_with_clustering_and_distance_filter(
    file_name, radar_position, plot_x_limits, plot_y_limits, num_frames=0, 
    grid_spacing=1, eps=1.0, min_samples=3, min_distance=1.0, max_distance=float('inf')
):
    """
    Visualize radar field of view, plot points from a CSV file, cluster them using DBSCAN,
    and highlight grid cells where clusters are detected. Filter points by distance range.

    Args:
    - file_name (str): Name of the CSV file relative to the script.
    - radar_position (tuple): Radar position (x, y).
    - plot_x_limits (list): Limits for the X-axis.
    - plot_y_limits (list): Limits for the Y-axis.
    - num_frames (int): Number of frames to plot (0 for all frames).
    - grid_spacing (float): Spacing for grid lines.
    - eps (float): DBSCAN maximum distance between points for clustering.
    - min_samples (int): Minimum samples for a core point in DBSCAN.
    - min_distance (float): Minimum distance from the radar sensor.
    - max_distance (float): Maximum distance from the radar sensor.
    """
    # Construct full file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Filter data by number of frames if specified
    if num_frames > 0:
        frames_to_plot = data['Frame'].unique()[:num_frames]
        data = data[data['Frame'].isin(frames_to_plot)]

    # Calculate distance from radar and filter points within the specified distance range
    data['Distance'] = np.sqrt(data['X [m]']**2 + data['Y [m]']**2)
    data = data[(data['Distance'] > min_distance) & (data['Distance'] <= max_distance)]

    # Perform DBSCAN clustering
    labels = dbscan_clustering(data, eps=eps, min_samples=min_samples)
    data['Cluster'] = labels

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(plot_x_limits)
    ax.set_ylim(plot_y_limits)
    ax.set_aspect('equal')

    # Radar visualization
    radar = Rectangle((radar_position[0] - 1, radar_position[1] - 1), 2, 2, color='blue', label='Radar')
    radar_fov = Wedge((radar_position[0], radar_position[1]), 50, 30, 150, alpha=0.2, color='blue', label='Radar FOV')
    ax.add_patch(radar)
    ax.add_patch(radar_fov)

    # Draw grid with customizable spacing
    x_ticks = range(plot_x_limits[0], plot_x_limits[1] + 1, grid_spacing)
    y_ticks = range(plot_y_limits[0], plot_y_limits[1] + 1, grid_spacing)
    for x in x_ticks:
        ax.plot([x, x], plot_y_limits, linestyle='--', color='gray', linewidth=0.5)
    for y in y_ticks:
        ax.plot(plot_x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)

    # Highlight grid cells for each cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = data[data['Cluster'] == label]
        grid_cells = set((int(np.floor(x)), int(np.floor(y))) for x, y in zip(cluster_points['X [m]'], cluster_points['Y [m]']))
        for cell in grid_cells:
            ax.add_patch(Rectangle((cell[0], cell[1]), grid_spacing, grid_spacing, color='red', alpha=0.3))

    # Plot points from the CSV file with cluster labels
    for label in unique_labels:
        cluster_points = data[data['Cluster'] == label]
        if label == -1:
            color = 'black'  # Noise points
            label_name = 'Noise'
        else:
            color = plt.cm.get_cmap('tab10')(label % 10)  # Distinct color for each cluster
            label_name = f'Cluster {label}'
        ax.scatter(cluster_points['X [m]'], cluster_points['Y [m]'], color=color, label=label_name, s=20)

    # Labels and legends
    ax.set_title("Radar Detection with Clustering and Distance Filtering")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    plt.show()


# Example usage
visualize_radar_with_clustering_and_distance_filter(
    file_name="coordinates.csv",
    radar_position=(0, 0),
    plot_x_limits=[-10, 10],
    plot_y_limits=[0, 15],
    num_frames=30,
    grid_spacing=1,
    eps=0.4,  # DBSCAN clustering radius
    min_samples=3,  # Minimum points to form a cluster
    min_distance=1.5,  # Minimum distance from radar
    max_distance=15.0  # Maximum distance from radar
)
