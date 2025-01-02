import os
import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from DataProcessing.radar_utilsProcessing import *
from DataProcessing.radar_utilsPlot import *

# Create a new dictionary with frame numbers and coordinates + Doppler speed
def extract_coordinates_with_doppler(frames_data, y_threshold=None, z_threshold=None, doppler_threshold=None):
    coordinates_dict = {}

    for frame_num, frame_content in frames_data.items():
        # Extract detected points for the current frame
        points = frame_content["Detected Points"]

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
            coordinates.append({
                "X [m]": point["X [m]"],
                "Y [m]": point["Y [m]"],
                "Z [m]": point["Z [m]"],
                "Doppler [m/s]": point["Doppler [m/s]"]
            })
        
        # Add the filtered coordinates list to the dictionary
        if coordinates:  # Only add frames with valid points
            coordinates_dict[frame_num] = coordinates

    return coordinates_dict

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

def calculate_occupancy_grid(points, x_limits, y_limits, grid_spacing):
    """
    Calculate an occupancy grid for the given points.

    Parameters:
        points (list of tuples): List of (x, y) or (x, y, z) coordinates.
        x_limits (tuple): The x-axis limits as (xmin, xmax).
        y_limits (tuple): The y-axis limits as (ymin, ymax).
        grid_spacing (int): Spacing between grid cells.

    Returns:
        np.ndarray: 2D occupancy grid.
    """
    # Calculate grid size
    x_bins = int((x_limits[1] - x_limits[0]) / grid_spacing)
    y_bins = int((y_limits[1] - y_limits[0]) / grid_spacing)

    # Initialize the grid
    occupancy_grid = np.zeros((x_bins, y_bins))

    # Populate the grid
    for point in points:
        if len(point) == 3:
            x, y, _ = point  # Unpack x, y, z
        elif len(point) == 2:
            x, y = point  # Unpack x, y only
        else:
            raise ValueError(f"Point format not supported: {point}")

        if x_limits[0] <= x < x_limits[1] and y_limits[0] <= y < y_limits[1]:
            x_idx = int((x - x_limits[0]) / grid_spacing)
            y_idx = int((y - y_limits[0]) / grid_spacing)
            occupancy_grid[x_idx, y_idx] += 1

    return occupancy_grid

def dbscan_clustering(data, eps=1.0, min_samples=3):
    """
    Perform DBSCAN clustering on the X and Y coordinates from the data.

    Args:
    - data (list or DataFrame): Data containing X and Y coordinates.
    - eps (float): The maximum distance between two samples for one to be considered in the neighborhood.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - labels (array): Cluster labels for each point. Noise points are labeled as -1.
    """
    # If data is a list, convert it to a numpy array
    if isinstance(data, list):
        if not data:  # Handle empty list
            print("DBSCAN: No data points provided.")
            return np.array([])  # Return empty labels
        data = np.array(data)

    # Ensure data is in 2D format
    if data.shape[0] == 0:
        print("DBSCAN: No valid points for clustering.")
        return np.array([])

    # Perform DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    return db.labels_


# Plotting function
def create_interactive_plots(frames_data, x_limits, y_limits, grid_spacing=1, eps=0.5, min_samples=5, history_frames=5):
    """
    Create an interactive plot with two subplots, a slider, and radio buttons,
    including a grid with customizable spacing. Annotates points in ax2 with Doppler values.
    
    Parameters:
        frames_data (dict): Data for dataset.
        x_limits (tuple): The x-axis limits as (xmin, xmax).
        y_limits (tuple): The y-axis limits as (ymin, ymax).
        grid_spacing (int): Spacing between grid lines (default is 1).
        history_frames (int): Number of frames for history-based visualization.
    """

    # Helper function to draw the grid with specified spacing
    def draw_grid(ax, x_limits, y_limits, grid_spacing):
        x_ticks = range(int(np.floor(x_limits[0])), int(np.ceil(x_limits[1])) + 1, grid_spacing)
        y_ticks = range(int(np.floor(y_limits[0])), int(np.ceil(y_limits[1])) + 1, grid_spacing)
        for x in x_ticks:
            ax.plot([x, x], y_limits, linestyle='--', color='gray', linewidth=0.5)
        for y in y_ticks:
            ax.plot(x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)
    # Helper function to calculate cumulative occupancy over history
    def calculate_cumulative_occupancy(frames_data, frame_idx, x_limits, y_limits, grid_spacing, history_frames):
        """
        Calculate a cumulative occupancy grid over the last `history_frames` frames.

        Parameters:
            frames_data (dict): Frame data dictionary.
            frame_idx (int): Current frame index.
            x_limits (tuple): X-axis limits.
            y_limits (tuple): Y-axis limits.
            grid_spacing (int): Spacing between grid cells.
            history_frames (int): Number of frames to include in the history.

        Returns:
            np.ndarray: Cumulative occupancy grid.
        """
        cumulative_grid = np.zeros((int((x_limits[1] - x_limits[0]) / grid_spacing),
                                    int((y_limits[1] - y_limits[0]) / grid_spacing)))

        for i in range(max(1, frame_idx - history_frames + 1), frame_idx + 1):
            coordinates = frames_data.get(i, [])  # Retrieve the list of points
            # Extract X and Y coordinates as tuples
            points = [(point["X [m]"], point["Y [m]"]) for point in coordinates]
            # Update the cumulative grid
            occupancy_grid = calculate_occupancy_grid(points, x_limits, y_limits, grid_spacing)
            cumulative_grid += occupancy_grid

        # Normalize cumulative grid to [0, 1] (optional for visualization purposes)
        cumulative_grid = np.clip(cumulative_grid, 0, 10)  # Limit max values to 10
        return cumulative_grid
    # Helper function 
    def create_custom_colormap():
        """
        Create a custom colormap for the occupancy grid.
        Returns:
            cmap: Custom colormap with a specific background color.
            norm: Normalizer to map data values to colormap levels.
        """
        # Define colors: First is the background color, followed by density colors
        colors = [
            "white",      # Background color (e.g., for 0)
            "#d1e5f0",    # Light blue (low density)
            "#92c5de",    # Blue
            "#4393c3",    # Medium density
            "#2166ac",    # Dark blue (high density)
            "#053061"     # Very high density
        ]
        cmap = ListedColormap(colors)

        # Define boundaries for each color bin
        boundaries = [0, 1, 2, 3, 4, 5, np.inf]  # Bins for densities
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        return cmap, norm

    # Create the figure and subplots
    fig = plt.figure(figsize=(18, 14))
    # Define a 4x2 grid layout
    gs = GridSpec(4, 2, figure=fig)

    # Subplots
    # Dataset 1 subplots (Cloud point visualization)
    ax1_1 = fig.add_subplot(gs[0, 0])  # Top-left: cumulative data for dataset 1
    ax1_2 = fig.add_subplot(gs[1, 0])  # Middle-left: per-frame data for dataset 1
    ax1_3 = fig.add_subplot(gs[2, 0])  # Bottom-left: occupancy grid for dataset 1
    ax1_4 = fig.add_subplot(gs[3, 0])  # History-based occupancy grid for dataset 1

    # Dataset 2 subplots (DBSCAN applied)
    ax2_1 = fig.add_subplot(gs[0, 1])  # Cumulative data
    ax2_2 = fig.add_subplot(gs[1, 1])  # Per-frame data
    ax2_3 = fig.add_subplot(gs[2, 1])  # Occupancy grid
    ax2_4 = fig.add_subplot(gs[3, 1])  # History-based occupancy grid

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.5, hspace=0.6)

    # Apply grid to all subplots
    for ax in [ax1_1, ax1_2, ax1_3, ax1_4]:
        draw_grid(ax, x_limits, y_limits, grid_spacing)

    # Get the custom colormap and normalizer
    cmap, norm = create_custom_colormap()

    # Initialize cumulative plots
    (line1_1,) = ax1_1.plot([], [], 'o', label="Dataset 1: Cumulative Data")

    for ax, title in zip([ax1_1], ["Point Cloud"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - Cumulative Data")

    # Initialize per-frame plots
    for ax, title in zip([ax1_2], ["Point Cloud"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.legend(["Dots Per Frame"], loc="upper left")
        ax.set_title(f"{title} - Per Frame Data")

    # Initialize occupancy grids
    for ax, title in zip([ax1_3], ["Point Cloud"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - Occupancy Grid")

    # Initialize history-based grids
    for ax, title in zip([ax1_4], ["Point Cloud"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - History-Based Grid")

    for ax, title in zip([ax2_1], ["DBSCAN applied"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        #ax.legend(loc="upper left")
        ax.set_title(f"{title} - Cumulative Data")

    # Initialize per-frame plots
    for ax, title in zip([ax2_2], ["DBSCAN applied"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.legend(["Dots Per Frame"], loc="upper left")
        ax.set_title(f"{title} - Per Frame Data")

    # Initialize occupancy grids
    for ax, title in zip([ax2_3], ["DBSCAN applied"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - Occupancy Grid")

    # Initialize history-based grids
    for ax, title in zip([ax2_4], ["DBSCAN applied"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - History-Based Grid")

    # Update function
    def update(val):
        # Get the current slider value
        frame_idx = int(slider_idx.val)

        # Initialize a persistent cumulative grid for ax2_4
        if not hasattr(update, "cumulative_grid"):
                x_bins = int((x_limits[1] - x_limits[0]) / grid_spacing)
                y_bins = int((y_limits[1] - y_limits[0]) / grid_spacing)
                update.cumulative_grid = np.zeros((x_bins, y_bins))  # Initialize cumulative grid


        """
        Dataset 1
        """
        # Initialize lists to store X, Y coordinates and Doppler values for the current frame
        coordinates1 = []
        doppler1 = []

        # Get the list of points for the current frame
        current_frame_points = frames_data.get(frame_idx, [])

        # Extract X, Y coordinates and Doppler values
        for point in current_frame_points:
            x_coord = point["X [m]"]
            y_coord = point["Y [m]"]
            doppler_speed = point["Doppler [m/s]"]

            coordinates1.append((x_coord, y_coord))
            doppler1.append(doppler_speed)

        # Check if the current frame has no valid points
        if not coordinates1:
            print(f"Frame {frame_idx} for Dataset 1 has no points after filtering.")
            return

        # -----------------------------------------
        # Ax1_1: Update cumulative data for dataset 1
        # -----------------------------------------
        x1, y1 = [], []
        for frame in range(1, frame_idx + 1):  # Accumulate data up to the current frame
            historical_points = frames_data.get(frame, [])  # Retrieve historical points
            for point in historical_points:
                x1.append(point["X [m]"])
                y1.append(point["Y [m]"])

        line1_1.set_data(x1, y1)  # Update the cumulative plot
        ax1_1.set_xlabel("X [m]")
        ax1_1.set_ylabel("Y [m]")

        # -----------------------------------------
        # Ax1_2: Update current frame data for dataset 1
        # -----------------------------------------
        ax1_2.cla()
        ax1_2.set_xlim(*x_limits)
        ax1_2.set_ylim(*y_limits)
        draw_grid(ax1_2, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_2)

        # Extract X and Y coordinates for the current frame
        x1 = [coord[0] for coord in coordinates1]
        y1 = [coord[1] for coord in coordinates1]
        ax1_2.plot(x1, y1, 'ro')  # Plot current frame points

        # Annotate Doppler values on the plot
        for (x, y), d in zip(coordinates1, doppler1):
            ax1_2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax1_2.set_title(f"Frame {frame_idx} - Dataset 1")
        ax1_2.legend(["Current Frame"], loc="upper left")
        ax1_2.set_xlabel("X [m]")
        ax1_2.set_ylabel("Y [m]")

        # -----------------------------------------
        # Ax1_3: Update Occupancy Grid for Dataset 1
        # -----------------------------------------
        occupancy_grid1 = calculate_occupancy_grid(coordinates1, x_limits, y_limits, grid_spacing)
        ax1_3.cla()
        ax1_3.imshow(occupancy_grid1.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax1_3.set_title(f"Occupancy Grid - Dataset 1 (Frame {frame_idx})")
        ax1_3.set_xlabel("X [m]")
        ax1_3.set_ylabel("Y [m]")
        draw_grid(ax1_3, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_3)

        # -----------------------------------------
        # Ax1_4: Update History-Based Occupancy Grid for Dataset 1
        # -----------------------------------------
        cumulative_grid1 = calculate_cumulative_occupancy(
            frames_data, frame_idx, x_limits, y_limits, grid_spacing, history_frames
        )
        ax1_4.cla()
        ax1_4.imshow(cumulative_grid1.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax1_4.set_title(f"History Grid - Dataset 1 (Last {history_frames} Frames)")
        ax1_4.set_xlabel("X [m]")
        ax1_4.set_ylabel("Y [m]")
        draw_grid(ax1_4, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_4)

        """
        Dataset 2 (with DBSCAN clustering)
        """
        eps2 = 0.4
        min_samples2 = 2

        # Initialize lists to store X, Y, and Z coordinates
        coordinates2 = []

        # Get the list of points for the current frame
        current_frame_points = frames_data.get(frame_idx, [])

        # Extract X, Y, and Z coordinates for clustering
        for point in current_frame_points:
            x_coord = point["X [m]"]
            y_coord = point["Y [m]"]
            coordinates2.append([x_coord, y_coord])

        # Check if the current frame has no valid points
        if not coordinates2:
            print(f"Frame {frame_idx} for Dataset 2 has no points after filtering.")
            return

        # Prepare DataFrame for clustering
        # Prepare DataFrame for clustering
        df = pd.DataFrame(coordinates2, columns=["X [m]", "Y [m]"])
        labels = dbscan_clustering(df, eps=eps2, min_samples=min_samples2)

        # -----------------------------------------
        # Ax2_1: Update cumulative clusters
        # -----------------------------------------
        if not hasattr(update, "cumulative_clusters"):
            update.cumulative_clusters = {"x": [], "y": []}  # Initialize cumulative clusters

        for cluster_label in set(labels):
            if cluster_label != -1:  # Ignore noise points
                cluster_points = df.loc[labels == cluster_label, ["X [m]", "Y [m]"]].values
                update.cumulative_clusters["x"].extend(cluster_points[:, 0])
                update.cumulative_clusters["y"].extend(cluster_points[:, 1])

        ax2_1.cla()
        ax2_1.scatter(update.cumulative_clusters["x"], update.cumulative_clusters["y"], c='purple', alpha=0.5, label="Cumulative Clusters")
        ax2_1.set_xlim(*x_limits)
        ax2_1.set_ylim(*y_limits)
        ax2_1.set_title("Cumulative Clusters")
        ax2_1.legend()

        # -----------------------------------------
        # Ax2_2: Current frame clusters
        # -----------------------------------------
        ax2_2.cla()
        ax2_2.set_xlim(*x_limits)
        ax2_2.set_ylim(*y_limits)
        draw_grid(ax2_2, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_2)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:  # Noise points
                col = [0, 0, 0, 1]  # Black for noise
            class_member_mask = (labels == k)
            xy = df[class_member_mask][["X [m]", "Y [m]"]].values
            ax2_2.scatter(xy[:, 0], xy[:, 1], c=[col], label=f"Cluster {k}")

        ax2_2.set_title(f"Frame {frame_idx} - DBSCAN Clusters")
        #ax2_2.legend()

        # -----------------------------------------
        # Ax2_3: Current frame occupancy grid for clusters
        # -----------------------------------------
        ax2_3.cla()
        ax2_3.set_xlim(*x_limits)
        ax2_3.set_ylim(*y_limits)

        # Filter only clustered points
        clustered_points = df[labels != -1][["X [m]", "Y [m]"]].values
        if clustered_points.size == 0:
            pass  # Do nothing if there are no clustered points
        else:
            # Calculate the occupancy grid for clustered points
            frame_grid = calculate_occupancy_grid(clustered_points, x_limits, y_limits, grid_spacing)

            # Update the cumulative grid
            if not hasattr(update, "cumulative_grid"):
                update.cumulative_grid = np.zeros_like(frame_grid)  # Initialize cumulative grid
            update.cumulative_grid += frame_grid

            # Plot the current frame's occupancy grid
            ax2_3.imshow(
                frame_grid.T,  # Transpose for proper orientation
                extent=(*x_limits, *y_limits),
                origin="lower",
                cmap=cmap,
                aspect="auto"
            )

        ax2_3.set_title("Clustered Occupancy Grid")
        draw_grid(ax2_3, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_3)

        # -----------------------------------------
        # Ax2_4: Cumulative history-based clustered occupancy grid
        # -----------------------------------------
        ax2_4.cla()
        ax2_4.set_xlim(*x_limits)
        ax2_4.set_ylim(*y_limits)

        # Normalize the cumulative grid for better visualization
        cumulative_grid_normalized = np.clip(update.cumulative_grid, 0, 10)

        # Plot the cumulative grid
        ax2_4.imshow(
            cumulative_grid_normalized.T,  # Transpose for proper orientation
            extent=(*x_limits, *y_limits),
            origin="lower",
            cmap=cmap,
            aspect="auto"
        )

        ax2_4.set_title("Cumulative Occupancy Grid")
        draw_grid(ax2_4, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_4)


        fig.canvas.draw_idle()
    # Add slider
    # [left, bottom, width, height]
    ax_slider = plt.axes([0.25, 0.10, 0.65, 0.03])  # Slider for Dataset 1
    # Initialize sliders with respective frame ranges
    slider_idx = Slider(
        ax_slider, 
        "Column 1", 
        min(frames_data.keys()), 
        max(frames_data.keys()), 
        valinit=min(frames_data.keys()), 
        valstep=1
    )

    slider_idx.on_changed(update)

    plt.show()


# Example Usage
# Get the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "DynamicMonitoring", "30fps_straight_3x3_log_2024-12-16.csv")
file_path = os.path.normpath(os.path.join(script_dir, relative_path))

y_threshold = 0.0  # Disregard points with Y < num
z_threshold = (0, 3.0)
doppler_threshold = 0.0 # Disregard points with doppler < num

print(f"Processing file: {file_path}")
frames_data = process_log_file(file_path, snr_threshold=15, z_min=-0.30, z_max=2.0, doppler_threshold=0.1)

# Extract new dictionary with frame numbers and coordinates + Doppler
frames_data = extract_coordinates_with_doppler(frames_data, y_threshold, z_threshold, doppler_threshold)

"""
Having a legen of Cluster -1, means no cluster has been created
Same as having Grey Clusters
"""
create_interactive_plots(frames_data, x_limits=(-8, 8), y_limits=(0, 15), eps=0.4, min_samples=4, history_frames = 10)