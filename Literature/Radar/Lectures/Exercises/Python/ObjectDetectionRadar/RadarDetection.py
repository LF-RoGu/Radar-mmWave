import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.patches import Patch
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import numpy as np
from filterpy.kalman import KalmanFilter

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

# Kalman filter initialization
def initialize_kalman_filter():
    """
    Initialize a Kalman filter with constant velocity motion model.

    Returns:
    - kf (KalmanFilter): Initialized Kalman filter object.
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4D state (x, y, vx, vy), 2D measurement (x, y)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],  # x, y, vx, vy
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.R *= 5  # Measurement noise covariance
    kf.P *= 10  # Initial state covariance
    kf.Q = np.eye(4) * 0.1  # Process noise covariance
    return kf
# Function to track object movement
def track_object_movement(clusters, kalman_filters):
    """
    Update cluster positions using Kalman filters.

    Args:
    - clusters (dict): Dictionary of cluster positions (x, y).
    - kalman_filters (dict): Dictionary of Kalman filters for each cluster.

    Returns:
    - updated_clusters (dict): Updated cluster positions.
    """
    updated_clusters = {}
    for cluster_id, cluster_pos in clusters.items():
        if cluster_id not in kalman_filters:
            kf = initialize_kalman_filter()
            kf.x[:2] = cluster_pos.reshape(-1, 1)  # Initialize position
            kalman_filters[cluster_id] = kf
        else:
            kf = kalman_filters[cluster_id]
        kf.predict()  # Predict the next state
        kf.update(cluster_pos.reshape(-1, 1))  # Update with the current position
        updated_clusters[cluster_id] = kf.x[:2].flatten()  # Store updated position
    return updated_clusters
# Function to estimate velocity of clusters
def estimate_velocity(kalman_filters):
    """
    Estimate velocity for each cluster.

    Args:
    - kalman_filters (dict): Dictionary of Kalman filters for each cluster.

    Returns:
    - velocities (dict): Estimated velocities (vx, vy) for each cluster.
    """
    velocities = {}
    for cluster_id, kf in kalman_filters.items():
        velocities[cluster_id] = kf.x[2:]  # Extract velocity (vx, vy)
    return velocities
# Function to predict cluster trajectories
def predict_cluster_trajectory(kalman_filters, steps=10):
    """
    Predict future positions of clusters using Kalman filters.

    Args:
    - kalman_filters (dict): Dictionary of Kalman filters for each cluster.
    - steps (int): Number of time steps to predict.

    Returns:
    - predictions (dict): Predicted future positions for each cluster.
    """
    predictions = {}
    for cluster_id, kf in kalman_filters.items():
        future_pos = []
        state = kf.x.copy()
        for _ in range(steps):
            state = np.dot(kf.F, state)  # Apply state transition
            future_pos.append(state[:2])  # Save predicted position
        predictions[cluster_id] = future_pos
    return predictions

# Function to visualize clusters being formed
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
# Function to visualize movement tracking
def visualize_cluster_movement(file_name, radar_position, plot_x_limits, plot_y_limits, num_frames=0, grid_spacing=1, eps=1.0, min_samples=3, min_distance=1.0, max_distance=float('inf')):
    """
    Visualize movement tracking of clusters using Kalman filters and calculate average Doppler speed for each cluster.
    Also draws dotted lines from the sensor (0, 0) to each cluster point and a vertical centerline to calculate angles.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load data
    data = pd.read_csv(file_path)
    if num_frames > 0:
        frames_to_plot = data['Frame'].unique()[:num_frames]
        data = data[data['Frame'].isin(frames_to_plot)]
    data['Distance'] = np.sqrt(data['X [m]']**2 + data['Y [m]']**2)
    data = data[(data['Distance'] > min_distance) & (data['Distance'] <= max_distance)]
    
    # Perform clustering
    labels = dbscan_clustering(data, eps=eps, min_samples=min_samples)
    data['Cluster'] = labels
    clusters = {label: data[data['Cluster'] == label][['X [m]', 'Y [m]']].mean().values for label in np.unique(labels) if label != -1}

    # Initialize Kalman filters
    kalman_filters = {}
    updated_clusters = track_object_movement(clusters, kalman_filters)

    # Calculate average Doppler speed for each cluster
    cluster_doppler = {}
    for cluster_id in np.unique(labels):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = data[data['Cluster'] == cluster_id]
            avg_doppler = cluster_points['Doppler [m/s]'].mean()  # Compute average Doppler speed
            cluster_doppler[cluster_id] = avg_doppler

    # Generate a colormap for clusters
    cluster_ids = list(clusters.keys())
    colormap = cm.get_cmap('tab10', len(cluster_ids))  # Distinct colors for each cluster


    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(plot_x_limits)
    ax.set_ylim(plot_y_limits)
    ax.set_aspect('equal')

    # Radar and FOV visualization
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

    # Draw a vertical centerline at the midpoint of the Y-axis
    y_center = (plot_y_limits[0] + plot_y_limits[1]) / 2
    ax.plot([0, 0], [plot_y_limits[0], plot_y_limits[1]], linestyle=':', color='green', linewidth=1, label='Centerline')

    # Plot movements, Doppler speeds, and angles
    for i, cluster_id in enumerate(cluster_ids):
        color = colormap(i)  # Get a unique color for the cluster
        original_pos = clusters[cluster_id]
        updated_pos = updated_clusters[cluster_id]
        avg_doppler = cluster_doppler.get(cluster_id, 0)  # Get average Doppler for the cluster

        # Draw dotted line from radar (0,0) to each cluster's updated position
        ax.plot([radar_position[0], updated_pos[0]], [radar_position[1], updated_pos[1]], linestyle=':', color='black', linewidth=1)

        # Plot the cluster movements and annotate Doppler speed
        ax.plot([original_pos[0], updated_pos[0]], [original_pos[1], updated_pos[1]], linestyle='--', color=color, label=f'Cluster {cluster_id}')
        ax.scatter(*original_pos, label=f'Cluster {cluster_id} (Original)', alpha=0.8, color=color)
        ax.scatter(*updated_pos, label=f'Cluster {cluster_id} (Updated)', color=color)
        ax.text(updated_pos[0], updated_pos[1], f"{avg_doppler:.2f} m/s", fontsize=9, color='purple')  # Annotate with Doppler speed

        # Calculate and display the angle from the centerline to the cluster
        angle = np.degrees(np.arctan2(updated_pos[1] - radar_position[1], updated_pos[0] - radar_position[0]))
        if(angle > 90):
            angle = angle - 90
        else:
            angle = 90 - angle
        ax.text(updated_pos[0], updated_pos[1] - 0.5, f"{angle:.1f}°", fontsize=9, color='black')  # Annotate angle

    ax.legend()
    ax.set_title("Cluster Movement with Angles and Average Doppler Speeds")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
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
# Example usage
visualize_cluster_movement(
    file_name="coordinates.csv",
    radar_position=(0, 0),
    plot_x_limits=[-10, 10],
    plot_y_limits=[0, 15],
    num_frames=50,
    grid_spacing=1,
    eps=0.4,
    min_samples=3
)