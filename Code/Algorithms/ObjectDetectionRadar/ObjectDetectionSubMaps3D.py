import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

from DataProcessing.radar_utilsProcessing import *
from DataProcessing.radar_utilsPlot import *
from OccupancyGrid.OccupancyGrid import *

SAFETY_BOX_CENTER = [0, 2, 0]  # Center position (X, Y, Z)
SAFETY_BOX_SIZE = [2, 9, 2]   # Width, Height, Depth

# Create a new dictionary with frame numbers and coordinates + Doppler speed
def extract_coordinates_with_doppler(frames_data, z_threshold=None):
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
            if z_threshold is not None and not (z_threshold[0] <= point["Z [m]"] <= z_threshold[1]):
                continue  # Skip if Z is outside the range

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
def cluster_points(points, eps=1.0, min_samples=6):
    """ Perform DBSCAN clustering and filter clusters based on priorities. """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])  # Use X, Y, Z for clustering
    labels = dbscan.labels_

    clusters = {}
    cluster_range_azimuth = []  # Calculate range and azimuth.
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
        range_to_origin = np.linalg.norm(centroid[:2])  # Range from centroid to origin (X, Y)
        azimuth_to_origin = np.degrees(np.arctan2(centroid[1], centroid[0]))  # Azimuth angle

        # Save range and azimuth
        cluster_range_azimuth.append((cluster_id, range_to_origin, azimuth_to_origin, cluster_points))

        # Store centroid and priority
        centroid = np.mean(cluster_points, axis=0)
        if size >= 10:
            priority = 3
        elif size < 10 and size >= 5:
            priority = 2
        elif size < 5:
            priority = 1
        else:
            priority = 4
        clusters[cluster_id] = {'centroid': centroid, 'priority': priority, 'points': cluster_points}

    return clusters, cluster_range_azimuth

# -------------------------------
# FUNCTION: Safety Boundary
# -------------------------------
def safety_boundary(cluster_range_azimuth, range_threshold, azimuth_range, adjacent_range):
    """ Evaluate clusters based on range and azimuth thresholds. """
    for cluster_id, cluster_range, cluster_azimuth, cluster_points in cluster_range_azimuth:
        
        relative_openness = abs(np.degrees(np.arctan2(cluster_points[0][0], cluster_points[0][1])))
        relative_adjacent = abs(cluster_range*(np.cos(relative_openness)))
        in_openness = relative_openness <= azimuth_range
        in_range = cluster_range <= range_threshold

        # Find the closest point to the origin in the cluster
        distances = np.linalg.norm(cluster_points[:, :2], axis=1)
        closest_point_index = np.argmin(distances)
        closest_point = cluster_points[closest_point_index]

        if(in_openness and in_range):
            # Final check for range and azimuth
            print("----------------------------------------")
            print(f"Cluster ID: {cluster_id}, Range: {cluster_range:.2f}, Azimuth: {cluster_azimuth:.2f}")
            print(f"Openess: {relative_openness}")
            print(f"Adjacent: {relative_adjacent}")
            print(f"Closest Point to Origin: X: {closest_point[0]:.2f}, Y: {closest_point[1]:.2f}, Z: {closest_point[2]:.2f}")



# -------------------------------
# FUNCTION: Plot Clusters
# -------------------------------
def plot_clusters_3d(clusters, ax):
    """ Plot clusters, visualize bounding boxes and priorities in 3D, and display average Doppler speed. """
    for cid, cluster in clusters.items():
        centroid = cluster['centroid']
        priority = cluster['priority']
        points = cluster['points']
        doppler_avg = np.mean(points[:, 2])  # Calculate average Doppler speed

        # Scatter points and centroid
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Cluster {cid}")
        ax.scatter(centroid[0], centroid[1], centroid[2], c='black', marker='x')  # Centroid marker

        # Display the average Doppler speed
        ax.text(centroid[0] + 0.2, centroid[1] + 0.2, centroid[2] + 0.2, f"{doppler_avg:.2f} m/s", color='purple')
        # Add priority labels
        ax.text(centroid[0] - 0.2, centroid[1] - 0.2, centroid[2] - 0.2, f"P{cluster['priority']}", color='red')
        # Draw bounding box (cube)
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
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
        if priority == 1:
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='red'))
        elif priority == 2:
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='yellow'))
        elif priority == 3:
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='green'))
        else:
            ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, facecolor='gray'))

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
# FUNCTION: Plot Clusters in Polar Occupancy Grid
# -------------------------------
def plot_clusters_polar(clusters, ax, range_max, range_bins, angle_bins):
    """
    Plot clusters in a polar occupancy grid.

    Parameters:
        clusters (dict): Clusters with centroid, priority, and points.
        ax (PolarAxes): Matplotlib polar axis.
        range_max (float): Maximum range for the grid.
        range_bins (int): Number of bins for range.
        angle_bins (int): Number of bins for angles.
    """
    # Initialize grid
    polar_grid = np.zeros((range_bins, angle_bins))
    offset = 270

    # Fill the occupancy grid with cluster data
    for cluster_id, cluster in clusters.items():
        centroid = cluster['centroid']
        priority = cluster['priority']
        r = np.sqrt(centroid[0]**2 + centroid[1]**2)
        theta = (np.degrees(np.arctan2(centroid[1], centroid[0])) + offset) % 360


        # Map to bins
        if r < range_max:
            r_bin = int(r / (range_max / range_bins))
            theta_bin = int(theta / (360 / angle_bins))
            polar_grid[r_bin, theta_bin] += 1

            # Plot centroid with color-coded priority
            color = 'green' if priority == 3 else 'yellow' if priority == 2 else 'red'
            ax.scatter(np.radians(theta), r, color=color, s=70, label=f'Cluster {cluster_id}')

    # Plot polar occupancy grid
    r = np.linspace(0, range_max, range_bins)
    theta = np.radians(np.linspace(0, 360, angle_bins, endpoint=False))
    R, Theta = np.meshgrid(r, theta)
    Z = polar_grid.T
    cmap, norm = create_custom_colormap()
    ax.pcolormesh(Theta, R, Z, cmap=cmap)

    # Configure polar plot settings
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_title("Polar Occupancy Grid with Clusters")

# -------------------------------
# FUNCTION: Monitor Cluster Motion for Collision Risk
# -------------------------------
def monitor_cluster_motion(cluster_history, range_threshold, azimuth_range):
    """
    Monitor clusters over multiple frames and warn if their motion indicates potential collision.

    Parameters:
        cluster_history (list): List of cluster dictionaries from previous frames.
        range_threshold (float): Maximum range to monitor.
        azimuth_range (tuple): Azimuth range to monitor (min, max) in degrees.
    """
    for i in range(1, len(cluster_history)):
        for cluster_id, cluster in cluster_history[i].items():
            centroid = cluster['centroid']
            r = np.sqrt(centroid[0]**2 + centroid[1]**2)
            theta = (np.degrees(np.arctan2(centroid[1], centroid[0])) + 270) % 360

            prev_centroid = cluster_history[i - 1].get(cluster_id, {}).get('centroid', None)
            if prev_centroid is not None:
                prev_r = np.sqrt(prev_centroid[0]**2 + prev_centroid[1]**2)
                if r < prev_r:  # Moving closer
                    in_range = r <= range_threshold
                    in_azimuth = azimuth_range[0] <= theta <= azimuth_range[1]
                    if in_range and in_azimuth:
                        print(f"COLLISION WARNING! Cluster {cluster_id} is approaching: Range = {r:.2f}m, Azimuth = {theta:.2f}°")

def initialize_hitmarker(range_bins, angle_bins):
    """
    Initialize a hitmarker grid with zeros.

    Parameters:
        range_bins (int): Number of bins for range.
        angle_bins (int): Number of bins for angles.

    Returns:
        np.ndarray: Initialized hitmarker grid.
    """
    return np.zeros((range_bins, angle_bins), dtype=int)

# -------------------------------
# FUNCTION: Update Hitmarker Grid
# -------------------------------
def update_hitmarker(hitmarker_grid, clusters, range_max, range_bins, angle_bins):
    """
    Update the hitmarker grid based on detected clusters.

    Parameters:
        hitmarker_grid (np.ndarray): Grid to store hit counts.
        clusters (dict): Detected clusters.
        range_max (float): Maximum range for bins.
        range_bins (int): Number of bins for range.
        angle_bins (int): Number of bins for angles.

    """
    for cluster_id, cluster in clusters.items():
        centroid = cluster['centroid']
        r = np.sqrt(centroid[0]**2 + centroid[1]**2)
        theta = (np.degrees(np.arctan2(centroid[1], centroid[0])) + 270) % 360

        # Map cluster to bins
        if r < range_max:
            r_bin = int(r / (range_max / range_bins))
            theta_bin = int(theta / (360 / angle_bins))
            hitmarker_grid[r_bin, theta_bin] += 1

# -------------------------------
# FUNCTION: Validate Hitmarker Grid
# -------------------------------
def validate_hitmarker(hitmarker_grid, valid_mark):
    """
    Validate hitmarkers based on a threshold.

    Parameters:
        hitmarker_grid (np.ndarray): Hitmarker grid.
        valid_mark (int): Threshold for valid hit count.

    Returns:
        np.ndarray: Binary grid indicating valid regions.
    """
    return (hitmarker_grid >= valid_mark).astype(int)

# -------------------------------
# FUNCTION: Plot Hitmarker Grid
# -------------------------------
def plot_hitmarker_polar(hitmarker_grid, ax, range_max, range_bins, angle_bins):
    """
    Plot the hitmarker grid in polar coordinates.

    Parameters:
        hitmarker_grid (np.ndarray): Grid with hit counts.
        ax (PolarAxes): Matplotlib polar axis.
        range_max (float): Maximum range.
        range_bins (int): Number of range bins.
        angle_bins (int): Number of angle bins.
    """
    r = np.linspace(0, range_max, range_bins)
    theta = np.radians(np.linspace(0, 360, angle_bins, endpoint=False))
    R, Theta = np.meshgrid(r, theta)
    Z = hitmarker_grid.T

    cmap, norm = create_custom_colormap()

    ax.pcolormesh(Theta, R, Z, cmap=cmap)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_title("Hitmarker Grid (Polar Coordinates)")

# Interactive slider-based visualization
def plot_with_slider(frames_data, num_frames=10):
    # List to store the history of clusters
    cluster_history = []
    fig = plt.figure(figsize=(15, 8))  # Adjust figure size for 3 plots
    gs = GridSpec(1, 3, figure=fig)    # 1 row, 3 columns

    ax = fig.add_subplot(gs[0, 0], projection='3d')      # 3D Cluster Plot
    ay = fig.add_subplot(gs[0, 1], polar=True)          # Polar Cluster Plot
    az = fig.add_subplot(gs[0, 2], polar=True)          # Polar Hitmarker Plot

    # Set initial view angle (top-down) for 3D
    ax.view_init(elev=90, azim=-90)

    # Slider for frame selection
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 1, len(frames_data) - num_frames + 1, valinit=1, valstep=1)

    # Initialize hitmarker grid
    range_max = 10
    grid_spacing = 1
    range_bins = int(range_max / grid_spacing)
    angle_bins = 360
    hitmarker_grid = initialize_hitmarker(range_bins, angle_bins)
    valid_mark = 3  # Threshold for valid hit counts

    def update(val):
        nonlocal hitmarker_grid  # Access the hitmarker grid in each update

        start_frame = int(slider.val)
        submap = aggregate_submap(frames_data, start_frame, num_frames)

        ax.clear()
        ay.clear()
        az.clear()

        # Plot clusters
        clusters, cluster_range_azimuth = cluster_points(submap, eps=1.0, min_samples=6)

        # Maintain a history of the last 3 frames
        if len(cluster_history) >= 3:
            monitor_cluster_motion(cluster_history, range_threshold=7, azimuth_range=(-30, 30))
            cluster_history.pop(0)  # Remove the oldest frame
        cluster_history.append(clusters)  # Add the latest frame data

        # Plot Clusters
        plot_clusters_3d(clusters, ax)
        plot_clusters_polar(clusters, ay, range_max, range_bins, angle_bins)

        # Update and plot Hitmarker Grid
        update_hitmarker(hitmarker_grid, clusters, range_max, range_bins, angle_bins)
        validated_hitmarker = validate_hitmarker(hitmarker_grid, valid_mark)
        plot_hitmarker_polar(validated_hitmarker, az, range_max, range_bins, angle_bins)

        # Configure Plot Settings
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-0.30, 10)
        ax.set_title(f"Clusters (Frames {start_frame} to {start_frame + num_frames - 1})")

        ay.set_title("Polar Cluster View")
        az.set_title("Hitmarker Grid View")

        plt.draw()

    slider.on_changed(update)
    update(1)  # Initial plot
    plt.show()


# Example Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "DynamicMonitoring", "30fps_straight_3x3_2_log_2024-12-16.csv")
file_path = os.path.normpath(os.path.join(script_dir, relative_path))

y_threshold = 0.0
z_threshold = (-0.30, 2.0)
doppler_threshold = 0.0

print(f"Processing file: {file_path}")
frames_data = process_log_file(file_path, snr_threshold=12)

coordinates_data = extract_coordinates_with_doppler(frames_data, z_threshold)

# Plot with slider and clustering
plot_with_slider(coordinates_data, num_frames=10)

# Count total rows in the file (excluding header)
total_rows = sum(1 for _ in open(file_path)) - 1

# Print summary
print(f"\nParsed {len(frames_data)} frames successfully out of {total_rows} total rows.")


