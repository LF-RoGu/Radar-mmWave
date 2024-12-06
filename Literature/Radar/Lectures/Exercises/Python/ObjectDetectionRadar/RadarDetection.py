import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN

def simulate_static_objects():
    """
    Generates static objects as clusters of points.
    Returns:
    - data (ndarray): Simulated radar data with X, Y coordinates and cluster IDs.
    """
    np.random.seed(42)  # For reproducibility

    # Define static objects as clusters of points
    objects = [
        {'center': (5, 10), 'points': 10, 'spread': 0.5},  # Object 1
        {'center': (-5, 8), 'points': 12, 'spread': 0.7},  # Object 2
        {'center': (3, 5), 'points': 8, 'spread': 0.3},   # Object 3
        {'center': (-7, 12), 'points': 15, 'spread': 1.0}  # Object 4
    ]

    data = []
    for obj_id, obj in enumerate(objects):
        x, y = np.random.normal(loc=obj['center'][0], scale=obj['spread'], size=obj['points']), \
               np.random.normal(loc=obj['center'][1], scale=obj['spread'], size=obj['points'])
        cluster_id = np.full(obj['points'], obj_id)
        data.extend(np.column_stack((x, y, cluster_id)))

    return np.array(data)

def visualized_fps_static(data, radar_position, plot_x_limits, plot_y_limits, eps=1.0, min_samples=3, grid_spacing=1):
    """
    Visualizes static objects with cluster detection and angle annotation.
    Args:
    - data (ndarray): Simulated radar data with X, Y coordinates and cluster IDs.
    - radar_position (tuple): Radar position (x, y).
    - plot_x_limits (list): Limits for the X-axis.
    - plot_y_limits (list): Limits for the Y-axis.
    - eps (float): DBSCAN maximum distance between points for clustering.
    - min_samples (int): Minimum samples for a core point in DBSCAN.
    - grid_spacing (float): Spacing for grid lines.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame):
        ax.clear()
        ax.set_xlim(plot_x_limits)
        ax.set_ylim(plot_y_limits)
        ax.set_aspect('equal')

        # Add radar wedge (60 degrees)
        radar_fov = Wedge(
            radar_position, 50, 90 - 30, 90 + 30,  # Adjusting to center the wedge at 90° (upward direction)
            alpha=0.2, color='blue', label='Radar FOV'
        )
        ax.add_patch(radar_fov)

        # Draw grid with customizable spacing
        x_ticks = range(plot_x_limits[0], plot_x_limits[1] + 1, grid_spacing)
        y_ticks = range(plot_y_limits[0], plot_y_limits[1] + 1, grid_spacing)
        for x in x_ticks:
            ax.plot([x, x], plot_y_limits, linestyle='--', color='gray', linewidth=0.5)
        for y in y_ticks:
            ax.plot(plot_x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)

        # Draw vertical centerline
        ax.plot([0, 0], [plot_y_limits[0], plot_y_limits[1]], linestyle=':', color='green', linewidth=1, label='Centerline')

        # Perform clustering
        points = data[:, :2]  # Extract X, Y coordinates
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_
        clusters = {label: points[labels == label].mean(axis=0) for label in np.unique(labels) if label != -1}

        # Plot clusters and annotate angles
        for cluster_id, cluster_pos in clusters.items():
            angle = np.degrees(np.arctan2(cluster_pos[1] - radar_position[1], cluster_pos[0] - radar_position[0]))
            color = plt.cm.get_cmap('tab10')(cluster_id % 10)

            # Draw dotted line to cluster
            ax.plot([radar_position[0], cluster_pos[0]], [radar_position[1], cluster_pos[1]], linestyle=':', color=color, linewidth=1)
            ax.scatter(cluster_pos[0], cluster_pos[1], color=color, label=f'Cluster {cluster_id}')

            # Annotate the angle
            ax.text(cluster_pos[0], cluster_pos[1] - 0.5, f"Angle: {angle:.1f}°", fontsize=9, color='black')

        ax.set_title(f"Static Objects Visualization (Frame {frame})")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    ani = FuncAnimation(fig, update, frames=1, repeat=False)  # Static frame
    plt.show()

# Simulate static objects and visualize
data = simulate_static_objects()
visualized_fps_static(data, radar_position=(0, 0), plot_x_limits=[-10, 10], plot_y_limits=[0, 15])
