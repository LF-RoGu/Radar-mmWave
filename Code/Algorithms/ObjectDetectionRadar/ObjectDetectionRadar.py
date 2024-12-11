import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

# Utility function to load data
def load_data(file_name, y_threshold=None, doppler_threshold=None):
    """
    Load CSV data from the specified file and organize it by frame, optionally filtering out points
    with Y-values below a specified threshold.
    
    Parameters:
        file_name (str): Path to the CSV file.
        y_threshold (float, optional): Minimum Y-value to include points. Points with Y < y_threshold
                                       will be excluded. Defaults to None (no filtering).
    
    Returns:
        dict: A dictionary where each key is a frame number, and the value is a tuple:
              (coordinates, doppler), where:
              - coordinates: List of tuples (x, y, z) for each point in the frame.
              - doppler: List of Doppler values for each point in the frame.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: File not found at {file_name}")
    
    # Load the CSV data
    df = pd.read_csv(file_name)
    
    # Debug: Print initial data size
    print(f"Initial data size: {df.shape}")

    # Apply filtering based on y_threshold if provided
    if y_threshold is not None:
        df = df[df["Y [m]"] >= y_threshold]
        # Debug: Print data size after filtering
        print(f"Filtered data size (Y [m] >= {y_threshold}): {df.shape}")
    # Apply filtering based on doppler_threshold if provided
    if doppler_threshold is not None:
        df = df[df["Doppler [m/s]"] <= doppler_threshold]
        # Debug: Print data size after filtering
        print(f"Filtered data size (Doppler [m/s] >= {doppler_threshold}): {df.shape}")
    
    # Group data by frame and organize the output
    frames_data = {}
    for frame, group in df.groupby("Frame"):
        coordinates = list(zip(group["X [m]"], group["Y [m]"], group["Z [m]"]))
        doppler = group["Doppler [m/s]"].tolist()
        frames_data[frame] = (coordinates, doppler)
    
    return frames_data

# Function ...
def filter_vehicle_zone(frames_data, forward_distance=0.3, diagonal_distance=0.42, buffer=0.3, azimuth=60, elevation=30):
    """
    Filter out points detected within the vehicle zone in each frame.

    Parameters:
        frames_data (dict): The radar data grouped by frame, as returned by `load_data`.
                            Each key is a frame number, and the value is a tuple (coordinates, doppler).
        forward_distance (float): Distance from the sensor to the front of the vehicle (meters).
        diagonal_distance (float): Diagonal distance from the sensor to the vehicle's edge (meters).
        buffer (float): Additional buffer distance around the vehicle (meters).
        azimuth (float): Sensor azimuth coverage in degrees.
        elevation (float): Sensor elevation coverage in degrees.

    Modifies:
        frames_data (dict): Filters points within each frame to remove only points inside the vehicle zone.
    """
    azimuth_rad = np.radians(azimuth / 2)  # Half azimuth in radians
    elevation_rad = np.radians(elevation / 2)  # Half elevation in radians

    max_distance = diagonal_distance + buffer  # Max radius from sensor
    min_distance = forward_distance - buffer  # Min radius (considering buffer)

    for frame, (coordinates, doppler) in frames_data.items():
        # Prepare new lists for filtered coordinates and Doppler values
        filtered_coordinates = []
        filtered_doppler = []

        for coord, doppler_value in zip(coordinates, doppler):
            x, y, z = coord

            # Calculate spherical coordinates
            radius = np.sqrt(x**2 + y**2 + z**2)
            azimuth_angle = np.arctan2(y, x)  # Azimuth in radians
            elevation_angle = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation in radians

            # Append points only if they are outside the vehicle zone
            if radius > max_distance or radius < min_distance:
                filtered_coordinates.append(coord)
                filtered_doppler.append(doppler_value)
            elif abs(azimuth_angle) > azimuth_rad or abs(elevation_angle) > elevation_rad:
                filtered_coordinates.append(coord)
                filtered_doppler.append(doppler_value)

        # Update the frame data with only filtered points
        frames_data[frame] = (filtered_coordinates, filtered_doppler)

# Function ...
def setup_kalman_filter():
    """
    Set up a Kalman filter for 2D tracking (x, y).
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])  # Initial state [x, y, x_velocity, y_velocity]
    kf.F = np.array([[1, 0, 1, 0],    # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],    # Measurement function
                     [0, 1, 0, 0]])
    kf.P *= 1000  # Initial uncertainty
    kf.R = np.array([[5, 0], [0, 5]])  # Measurement noise covariance
    kf.Q = np.eye(4) * 0.1  # Process noise covariance
    return kf

# Function to draw the sensor's detection area as a wedge
def draw_sensor_area(ax, sensor_origin=(0, -1), azimuth=60, max_distance=12):
    """
    Draw a wedge to simulate the sensor's detection area pointing upwards.

    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axis to draw on.
        sensor_origin (tuple): The (x, y) coordinates of the sensor's origin.
        azimuth (float): The azimuth angle (in degrees) for the sensor's field of view.
        max_distance (float): The maximum detection radius of the sensor.

    Returns:
        None
    """
    # Adjust the angles so the wedge points upwards (positive Y-axis)
    start_angle = 90 - azimuth / 2
    end_angle = 90 + azimuth / 2

    # Create the wedge
    wedge = Wedge(
        center=sensor_origin,
        r=max_distance,
        theta1=start_angle,
        theta2=end_angle,
        facecolor="blue",
        alpha=0.2,
        edgecolor="black",
        linewidth=1
    )

    # Add the wedge to the axis
    ax.add_patch(wedge)

    # Optionally, add the sensor's location as a point
    ax.scatter(*sensor_origin, color="green", label="Sensor Location")

# Function
def draw_lanes(ax, lane_boundaries, y_limits):
    """
    Draw lane representations on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to draw lanes on.
        lane_boundaries (list of tuples): List of (x_min, x_max) tuples for each lane.
        y_limits (tuple): The y-axis limits as (y_min, y_max).
    """
    colors = ["lightblue", "lightgreen", "lightyellow", "lightpink"]
    lane_labels = ["Left Side", "Left Center", "Right Center", "Right Side"]
    # clear the lane so it does not get saturated
    ax.cla()

    for i, (x_min, x_max) in enumerate(lane_boundaries):
        ax.fill_betweenx(y_limits, x_min, x_max, color=colors[i], alpha=0.3, label=lane_labels[i])
        ax.axvline(x=x_min, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(x=x_max, color="black", linestyle="--", linewidth=0.5)

    ax.set_xlim(min([b[0] for b in lane_boundaries]), max([b[1] for b in lane_boundaries]))
    ax.set_ylim(y_limits)
    ax.set_title("Lane Representation")

# Function to cluster points
def cluster_static_objects(coordinates, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering to radar points.

    Parameters:
        coordinates (list of tuples): List of (X, Y) points.
        eps (float): Maximum distance between two points to be considered in the same cluster.
        min_samples (int): Minimum number of points required to form a cluster.

    Returns:
        list: Cluster labels for each point (-1 indicates noise).
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    return clustering.labels_

# Plotting function
def create_interactive_plot(frames_data, x_limits, y_limits, grid_spacing=1, eps=0.5, min_samples=5):
    """
    Create an interactive plot with two subplots, a slider, and radio buttons,
    including a grid with customizable spacing. Annotates points in ax2 with Doppler values.
    
    Parameters:
        frames_data (dict): The frame data dictionary from `load_data`.
        x_limits (tuple): The x-axis limits as (xmin, xmax).
        y_limits (tuple): The y-axis limits as (ymin, ymax).
        grid_spacing (int): Spacing between grid lines (default is 1).
    """
    # Create the figure and subplots
    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    # Define a 2x2 grid layout
    gs = GridSpec(2, 3, figure=fig)

    # Subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left
    ax3 = fig.add_subplot(gs[0, 1])  # Top-right
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right
    ax5 = fig.add_subplot(gs[0, 2])  # Bottom-right
    ax6 = fig.add_subplot(gs[1, 2])  # Bottom-right

    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Create lines for ax1
    (line1,) = ax1.plot([], [], 'o', label="Data - Ax1")
    ax1.set_xlim(*x_limits)
    ax1.set_ylim(*y_limits)
    ax1.legend(["Cumulative dots"], loc="upper left")

    # ax2 settings
    ax2.set_xlim(*x_limits)
    ax2.set_ylim(*y_limits)
    ax2.legend(["Dots Per Frame"], loc="upper left")

    # Create lines for ax3
    (line2,) = ax1.plot([], [], 'o', label="Data - Ax1")
    ax3.set_xlim(*x_limits)
    ax3.set_ylim(*y_limits)
    ax3.legend(["Cumulative Clusters"], loc="upper left")

    # ax4 settings, Cluster plot
    ax4.set_xlim(*x_limits)
    ax4.set_ylim(*y_limits)
    ax4.legend(["Clusters"], loc="upper left")

    # Function to draw the grid with specified spacing
    def draw_grid(ax, x_limits, y_limits, grid_spacing):
        x_ticks = range(int(np.floor(x_limits[0])), int(np.ceil(x_limits[1])) + 1, grid_spacing)
        y_ticks = range(int(np.floor(y_limits[0])), int(np.ceil(y_limits[1])) + 1, grid_spacing)
        for x in x_ticks:
            ax.plot([x, x], y_limits, linestyle='--', color='gray', linewidth=0.5)
        for y in y_ticks:
            ax.plot(x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)

    # Draw grids and wedges on all axes
    for ax in [ax1, ax2, ax3]:
        draw_grid(ax, x_limits, y_limits, grid_spacing)

    # Add slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, "Frame", 1, len(frames_data), valinit=1, valstep=1)

    # Update function
    def update(val):
        current_frame = int(slider.val)  # Get the current frame from the slider

        """
        ax1 starts here
        """
        # Get the data for all frames up to the current frame
        x1, y1 = [], []
        for frame in range(1, current_frame + 1):
            coordinates, _ = frames_data[frame]
            x1.extend([coord[0] for coord in coordinates])
            y1.extend([coord[1] for coord in coordinates])

        # Update ax1 with cumulative data
        line1.set_data(x1, y1)
        #draw_sensor_area(ax1) # Redraw wedge
        """
        ax2 starts here
        """
        # Update ax2 with only the current frame's data
        ax2.cla()  # Clear ax2
        ax2.set_xlim(*x_limits)  # Reset x-axis limits
        ax2.set_ylim(*y_limits)  # Reset y-axis limits
        draw_grid(ax2, x_limits, y_limits, grid_spacing)  # Redraw grid
        draw_sensor_area(ax2) # Redraw wedge
        
        coordinates, doppler = frames_data[current_frame]
        x2 = [coord[0] for coord in coordinates]
        y2 = [coord[1] for coord in coordinates]
        ax2.plot(x2, y2, 'ro')  # Plot current frame data

        # Annotate each point with its Doppler value
        for x, y, d in zip(x2, y2, doppler):
            ax2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax2.set_title(f"Frame {current_frame}")
        ax2.legend(["Current Frame"], loc="upper left")

        """
        ax3 starts here
        """
        # Clear ax3 and reset limits
        ax3.cla()
        ax3.set_xlim(*x_limits)
        ax3.set_ylim(*y_limits)
        draw_grid(ax3, x_limits, y_limits, grid_spacing)

        # Get cumulative data for clusters up to the current frame
        cumulative_coordinates = []
        for frame in range(1, current_frame + 1):
            coordinates, _ = frames_data[frame]
            cumulative_coordinates.extend(coordinates)

        # Perform clustering on cumulative data
        if len(cumulative_coordinates) > 0:
            cumulative_coordinates = np.array(cumulative_coordinates)  # Convert to NumPy array
            cluster_labels = cluster_static_objects(cumulative_coordinates, eps=eps, min_samples=min_samples)

            # Plot each cluster, ignoring noise points
            for cluster in set(cluster_labels):
                if cluster == -1:
                    # Skip noise points
                    continue

                # Get points belonging to this cluster
                cluster_points = cumulative_coordinates[np.array(cluster_labels) == cluster]

                # Assign random color to the cluster
                color = np.random.rand(3,)
                ax3.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f"Cluster {cluster}")

        ax3.set_title(f"Cumulative Clusters up to Frame {current_frame}")
        #ax3.legend()


        """
        ax4 starts here
        """
        # Update ax4 with clustered data
        ax4.cla()  # Clear ax4
        draw_grid(ax4, x_limits, y_limits, grid_spacing)
        if len(coordinates) > 0:
            cluster_labels = cluster_static_objects(np.array(coordinates), eps=eps, min_samples=min_samples)
            for cluster in set(cluster_labels):
                cluster_points = np.array(coordinates)[np.array(cluster_labels) == cluster]
                color = "gray" if cluster == -1 else np.random.rand(3,)
                ax4.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f"Cluster {cluster}")
        #ax3.set_title(f"Clusters for Frame {current_frame}")
        draw_sensor_area(ax4) # Redraw wedge
        #ax3.legend()

        """
        ax5 starts here
        """
        ax5_lane_boundaries = [(-5, -2), (-2, 0), (0, 2), (2, 5)]  # Example lane boundaries
        draw_lanes(ax5, ax5_lane_boundaries, y_limits)

        # Clear ax6 and draw lanes
        ax5.cla()
        draw_lanes(ax5, ax5_lane_boundaries, y_limits)
        draw_sensor_area(ax5, sensor_origin=(0, -1), azimuth=60, max_distance=12) # Redraw wedge

        # Get current frame data
        coordinates, doppler = frames_data[current_frame]

        # Filter points within the inner lanes (-2, 2)
        filtered_coordinates = [
            coord for coord in coordinates if -2 <= coord[0] <= 2
        ]
        filtered_doppler = [
            d for coord, d in zip(coordinates, doppler) if -2 <= coord[0] <= 2
        ]

        # Extract coordinates and apply DBSCAN
        coordinates_np = np.array([[x, y] for x, y, _ in coordinates])  # Ignore Z values if present
        cluster_labels = cluster_static_objects(coordinates_np, eps=0.6, min_samples=2)

        # Prepare filtered data lists
        filtered_x = []
        filtered_y = []

        # Process each cluster separately
        for cluster in set(cluster_labels):
            if cluster == -1:
                continue  # Skip noise points

            # Get points belonging to this cluster
            cluster_points = coordinates_np[cluster_labels == cluster]

            # Apply Kalman filter to this cluster
            kf = setup_kalman_filter()
            for x, y in cluster_points:
                z = np.array([x, y])  # Current measurement
                kf.predict()
                kf.update(z)
                filtered_x.append(kf.x[0])  # Filtered x-coordinate
                filtered_y.append(kf.x[1])  # Filtered y-coordinate

            # Plot filtered cluster points
            ax5.plot(filtered_x, filtered_y, 'o', label=f"Cluster {cluster}")

        # Annotate each filtered point with its Doppler value
        for x, y, d in zip(filtered_x, filtered_y, doppler):
            ax5.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        # Plot filtered data
        ax5.plot(filtered_x, filtered_y, 'go', label="Filtered Data")
        ax5.set_title("Filtered Data with Kalman Filter")

        """
        ax6 starts here
        """      
        ax6_lane_boundaries = [(-5, -3), (-3, 0), (0, 3), (3, 5)]  # Example lane boundaries
        # Draw lane representations on ax5 and ax6
        draw_lanes(ax6, ax6_lane_boundaries, y_limits)

        # Update ax6 with only the current frame's data
        draw_sensor_area(ax6, sensor_origin=(0, -1), azimuth=30, max_distance=12) # Redraw wedge
        
        coordinates, doppler = frames_data[current_frame]
        x2 = [coord[0] for coord in coordinates]
        y2 = [coord[1] for coord in coordinates]
        ax6.plot(x2, y2, 'ro')  # Plot current frame data

        # Annotate each point with its Doppler value
        for x, y, d in zip(x2, y2, doppler):
            ax6.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)

    plt.show()

# Example Usage
# Get the absolute path to the CSV file
file_name = "coordinates.csv"  # Replace with your file path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)

y_threshold = 2.0  # Disregard points with Y < num
doppler_threshold = 0.1 # Disregard points with doppler < num

frames_data = load_data(file_path, y_threshold, doppler_threshold)

# Filter out points inside the vehicle zone
filter_vehicle_zone(
    frames_data,
    forward_distance=0.3,
    diagonal_distance=0.42,
    buffer=0.0,
    azimuth=60,
    elevation=30
)
"""
Having a legen of Cluster -1, means no cluster has been created
Same as having Grey Clusters
"""
create_interactive_plot(frames_data, x_limits=(-8, 8), y_limits=(0, 15), eps=0.4, min_samples=5)