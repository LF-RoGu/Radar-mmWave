import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

# Utility function to load data
def load_data(file_name, y_threshold=None, z_min=None, z_max=None, doppler_threshold=None):
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
    # Apply Z-coordinate constraints
    if z_min is not None:
        df = df[df["Z [m]"] >= z_min]
    if z_max is not None:
        df = df[df["Z [m]"] <= z_max]
    
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
def create_interactive_plot_3d(frames_data, x_limits, y_limits, z_limits, grid_spacing=1, eps=0.5, min_samples=5):
    """
    Create an interactive 3D plot alongside 2D plots with sliders and clustering.
    """
    # Create the figure and subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)

    # 2D Subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # 3D Subplot
    ax3D = fig.add_subplot(gs[:, 2], projection='3d')

    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Draw grids and set limits for 2D plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    # Set limits for 3D plot
    ax3D.set_xlim(*x_limits)
    ax3D.set_ylim(*y_limits)
    ax3D.set_zlim(*z_limits)
    ax3D.set_title("3D Visualization")

    # Slider for frame control
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Frame", 1, len(frames_data), valinit=1, valstep=1)

    def update(val):
        current_frame = int(slider.val)  # Current frame

        """
        ax3D starts here
        """
        # Clear and update the 3D plot
        ax3D.cla()
        ax3D.set_xlim(*x_limits)
        ax3D.set_ylim(*y_limits)
        ax3D.set_zlim(*z_limits)

        # Cumulative 3D points
        cumulative_x, cumulative_y, cumulative_z = [], [], []
        for frame in range(1, current_frame + 1):
            coordinates, _ = frames_data[frame]
            cumulative_x.extend([coord[0] for coord in coordinates])
            cumulative_y.extend([coord[1] for coord in coordinates])
            cumulative_z.extend([coord[2] for coord in coordinates])

        # Current frame 3D points
        coordinates, doppler = frames_data[current_frame]
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
        z = [coord[2] for coord in coordinates]

        # Plot cumulative points in 3D
        ax3D.scatter(cumulative_x, cumulative_y, cumulative_z, c='gray', alpha=0.5, label="Cumulative Data")

        # Plot current frame points in 3D
        sc = ax3D.scatter(x, y, z, c=doppler, cmap='viridis', label="Current Frame")
        ax3D.legend()

        # Annotate Doppler values
        for x_i, y_i, z_i, d in zip(x, y, z, doppler):
            ax3D.text(x_i, y_i, z_i, f"{d:.2f}", fontsize=8, ha="center", color="blue")

        """
        Update other subplots as per your existing 2D logic (e.g., ax1, ax2, ax3, etc.)
        """

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# Example Usage
# Get the absolute path to the CSV file
file_name = "coordinates.csv"  # Replace with your file path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)

y_threshold = 2.0  # Disregard points with Y < num
x_limits=(-8, 8)
y_limits=(0, 15)
z_limits = (0.5, 2.0)
doppler_threshold = 0.1 # Disregard points with doppler < num

frames_data = load_data(file_path, y_threshold, z_limits[0], z_limits[1], doppler_threshold)

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
create_interactive_plot_3d(frames_data, x_limits=(-8, 8), y_limits=(0, 15), z_limits=z_limits, eps=0.4, min_samples=5)
