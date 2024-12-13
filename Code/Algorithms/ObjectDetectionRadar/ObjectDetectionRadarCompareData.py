import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm

# Utility function to load data
def load_data(file_name, y_threshold=None, z_threshold=None, doppler_threshold=None):
    """
    Load CSV data from the specified file and organize it by frame, filtering out rows where
    any value violates the thresholds.

    Parameters:
        file_name (str): Path to the CSV file.
        y_threshold (float, optional): Minimum Y-value to include points. Points with Y < y_threshold
                                       will be excluded. Defaults to None (no filtering).
        z_threshold (tuple, optional): Tuple (lower_bound, upper_bound) for filtering Z values.
                                       Defaults to None (no filtering).
        doppler_threshold (float, optional): Minimum absolute Doppler value to include points. Points with
                                             abs(Doppler) <= doppler_threshold will be excluded. Defaults to None (no filtering).
    
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
    
    # Apply filtering: Remove rows where any condition fails
    if y_threshold is not None:
        df = df[df["Y [m]"] >= y_threshold]
        print(f"Filtered data size (Y [m] >= {y_threshold}): {df.shape}")

    if z_threshold is not None:
        # Ensure z_threshold is a tuple with lower and upper bounds
        if isinstance(z_threshold, tuple) and len(z_threshold) == 2:
            lower_bound, upper_bound = z_threshold
            df = df[(df["Z [m]"] >= lower_bound) & (df["Z [m]"] <= upper_bound)]
            print(f"Filtered data size ({lower_bound} <= Z [m] <= {upper_bound}): {df.shape}")
        else:
            raise ValueError("z_threshold must be a tuple with two elements: (lower_bound, upper_bound).")

    if doppler_threshold is not None:
        df = df[df["Doppler [m/s]"].abs() > doppler_threshold]
        print(f"Filtered data size (Doppler [m/s] > {doppler_threshold}): {df.shape}")
    
    # Handle empty dataset case
    if df.empty:
        print(f"Warning: No data points remain after applying filters.")
        return {}
    
    # Group data by frame and organize the output
    frames_data = {}
    for frame, group in df.groupby("Frame"):
        coordinates = list(zip(group["X [m]"], group["Y [m]"], group["Z [m]"]))
        doppler = group["Doppler [m/s]"].tolist()
        if coordinates:  # Ensure frames with no points are skipped
            frames_data[frame] = (coordinates, doppler)
    
    return frames_data



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
        points (list of tuples): List of (x, y, z) coordinates.
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
    for x, y, _ in points:
        if x_limits[0] <= x < x_limits[1] and y_limits[0] <= y < y_limits[1]:
            x_idx = int((x - x_limits[0]) / grid_spacing)
            y_idx = int((y - y_limits[0]) / grid_spacing)
            occupancy_grid[x_idx, y_idx] += 1

    return occupancy_grid

# Plotting function
def create_interactive_plots(frames_data1, frames_data2, frames_data3, x_limits, y_limits, grid_spacing=1, eps=0.5, min_samples=5, history_frames=5):
    """
    Create an interactive plot with two subplots, a slider, and radio buttons,
    including a grid with customizable spacing. Annotates points in ax2 with Doppler values.
    
    Parameters:
        frames_data1 (dict): Data for dataset 1.
        frames_data2 (dict): Data for dataset 2.
        frames_data3 (dict): Data for dataset 3.
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
            coordinates, _ = frames_data.get(i, ([], []))
            occupancy_grid = calculate_occupancy_grid(coordinates, x_limits, y_limits, grid_spacing)
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
    # Define a 4x3 grid layout
    gs = GridSpec(4, 3, figure=fig)

    # Subplots
    # Dataset 1
    ax1_1 = fig.add_subplot(gs[0, 0])  # Top-left: cumulative data for dataset 1
    ax1_2 = fig.add_subplot(gs[1, 0])  # Middle-left: per-frame data for dataset 1
    ax1_3 = fig.add_subplot(gs[2, 0])  # Bottom-left: occupancy grid for dataset 1
    ax1_4 = fig.add_subplot(gs[3, 0])  # History-based occupancy grid for dataset 1

    # Dataset 2
    ax2_1 = fig.add_subplot(gs[0, 1])  # Top-center: cumulative data for dataset 2
    ax2_2 = fig.add_subplot(gs[1, 1])  # Middle-center: per-frame data for dataset 2
    ax2_3 = fig.add_subplot(gs[2, 1])  # Bottom-center: occupancy grid for dataset 2
    ax2_4 = fig.add_subplot(gs[3, 1])  # History-based occupancy grid for dataset 2

    # Dataset 3
    ax3_1 = fig.add_subplot(gs[0, 2])  # Top-right: cumulative data for dataset 3
    ax3_2 = fig.add_subplot(gs[1, 2])  # Middle-right: per-frame data for dataset 3
    ax3_3 = fig.add_subplot(gs[2, 2])  # Bottom-right: occupancy grid for dataset 3
    ax3_4 = fig.add_subplot(gs[3, 2])  # History-based occupancy grid for dataset 3

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.5, hspace=0.6)

    # Apply grid to all subplots
    for ax in [ax1_1, ax1_2, ax1_3, ax1_4, ax2_1, ax2_2, ax2_3, ax2_4, ax3_1, ax3_2, ax3_3, ax3_4]:
        draw_grid(ax, x_limits, y_limits, grid_spacing)

    # Get the custom colormap and normalizer
    cmap, norm = create_custom_colormap()

    # Initialize cumulative plots
    (line1_1,) = ax1_1.plot([], [], 'o', label="Dataset 1: Cumulative Data")
    (line2_1,) = ax2_1.plot([], [], 'o', label="Dataset 2: Cumulative Data")
    (line3_1,) = ax3_1.plot([], [], 'o', label="Dataset 3: Cumulative Data")

    for ax, title in zip([ax1_1, ax2_1, ax3_1], ["Dataset 1", "Dataset 2", "Dataset 3"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.legend(loc="upper left")
        ax.set_title(f"{title} - Cumulative Data")

    # Initialize per-frame plots
    for ax, title in zip([ax1_2, ax2_2, ax3_2], ["Dataset 1", "Dataset 2", "Dataset 3"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.legend(["Dots Per Frame"], loc="upper left")
        ax.set_title(f"{title} - Per Frame Data")

    # Initialize occupancy grids
    for ax, title in zip([ax1_3, ax2_3, ax3_3], ["Dataset 1", "Dataset 2", "Dataset 3"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - Occupancy Grid")

    # Initialize history-based grids
    for ax, title in zip([ax1_4, ax2_4, ax3_4], ["Dataset 1", "Dataset 2", "Dataset 3"]):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{title} - History-Based Grid")

    # Update function
    def update(val):
        # Get the current slider value
        frame1 = int(slider1.val)
        frame2 = int(slider2.val)
        frame3 = int(slider3.val)

        """
        Dataset 1
        """
        coordinates1, doppler1 = frames_data1.get(frame1, ([], []))  # Current frame's data
        if not coordinates1:
            print(f"Frame {frame1} for Dataset 1 has no points after filtering.")
            return
        
        # Ax1_1: Update cumulative data for dataset 1
        x1, y1 = [], []
        for frame in range(1, frame1 + 1):  # Accumulate data up to the current frame
            x1.extend([coord[0] for coord in coordinates1])
            y1.extend([coord[1] for coord in coordinates1])

        line1_1.set_data(x1, y1)
        ax1_1.set_xlabel("X [m]")
        ax1_1.set_ylabel("Y [m]")

        # Ax1_2: Update current frame data for dataset 1
        ax1_2.cla()
        ax1_2.set_xlim(*x_limits)
        ax1_2.set_ylim(*y_limits)
        draw_grid(ax1_2, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_2)  # Assuming this function visualizes the sensor's area

        x1 = [coord[0] for coord in coordinates1]
        y1 = [coord[1] for coord in coordinates1]
        ax1_2.plot(x1, y1, 'ro')

        # Annotate Doppler values on the plot
        for x, y, d in zip(x1, y1, doppler1):
            ax1_2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax1_2.set_title(f"Frame {frame1} - Dataset 1")
        ax1_2.legend(["Current Frame"], loc="upper left")
        ax1_2.set_xlabel("X [m]")
        ax1_2.set_ylabel("Y [m]")

        # Update ax1_3: Occupancy Grid for Dataset 1
        occupancy_grid1 = calculate_occupancy_grid(coordinates1, x_limits, y_limits, grid_spacing)
        ax1_3.cla()
        ax1_3.imshow(occupancy_grid1.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax1_3.set_title(f"Occupancy Grid - Dataset 1 (Frame {frame1})")
        ax1_3.set_xlabel("X [m]")
        ax1_3.set_ylabel("Y [m]")
        draw_grid(ax1_3, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_3)

        # Update ax1_4: History-Based Occupancy Grid for Dataset 1
        cumulative_grid1 = calculate_cumulative_occupancy(
            frames_data1, frame1, x_limits, y_limits, grid_spacing, history_frames
        )
        ax1_4.cla()
        ax1_4.imshow(cumulative_grid1.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax1_4.set_title(f"History Grid - Dataset 1 (Last {history_frames} Frames)")
        ax1_4.set_xlabel("X [m]")
        ax1_4.set_ylabel("Y [m]")
        draw_grid(ax1_4, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax1_4)


        """
        Dataset 2
        """
        coordinates2, doppler2 = frames_data2.get(frame2, ([], []))  # Current frame's data
        if not coordinates2:
            print(f"Frame {frame2} for Dataset 2 has no points after filtering.")
            return
        # Ax2_1: Update cumulative data for dataset 2
        x2, y2 = [], []
        for frame in range(1, frame2 + 1):  # Accumulate data up to the current frame
            x2.extend([coord[0] for coord in coordinates2])
            y2.extend([coord[1] for coord in coordinates2])

        line2_1.set_data(x2, y2)
        ax2_1.set_xlabel("X [m]")
        ax2_1.set_ylabel("Y [m]")

        # Ax2_2: Update current frame data for dataset 2
        ax2_2.cla()
        ax2_2.set_xlim(*x_limits)
        ax2_2.set_ylim(*y_limits)
        ax2_2.set_xlabel("X [m]")
        ax2_2.set_ylabel("Y [m]")
        draw_grid(ax2_2, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_2)  # Assuming this function visualizes the sensor's area

        x2 = [coord[0] for coord in coordinates2]
        y2 = [coord[1] for coord in coordinates2]
        ax2_2.plot(x2, y2, 'ro')

        # Annotate Doppler values on the plot
        for x, y, d in zip(x2, y2, doppler2):
            ax2_2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax2_2.set_title(f"Frame {frame2} - Dataset 2")
        ax2_2.legend(["Current Frame"], loc="upper left")

        # Update ax2_3: Occupancy Grid for Dataset 2
        occupancy_grid2 = calculate_occupancy_grid(coordinates2, x_limits, y_limits, grid_spacing)
        ax2_3.cla()
        ax2_3.imshow(occupancy_grid2.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax2_3.set_title(f"Occupancy Grid - Dataset 2 (Frame {frame2})")
        ax2_3.set_xlabel("X [m]")
        ax2_3.set_ylabel("Y [m]")
        draw_grid(ax2_3, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_3)

        # Update ax2_4: History-Based Occupancy Grid for Dataset 2
        cumulative_grid2 = calculate_cumulative_occupancy(
            frames_data2, frame2, x_limits, y_limits, grid_spacing, history_frames
        )
        ax2_4.cla()
        ax2_4.imshow(cumulative_grid2.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax2_4.set_title(f"History Grid - Dataset 2 (Last {history_frames} Frames)")
        ax2_4.set_xlabel("X [m]")
        ax2_4.set_ylabel("Y [m]")
        draw_grid(ax2_4, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax2_4)


        """
        Dataset 3
        """
        coordinates3, doppler3 = frames_data3.get(frame3, ([], []))  # Current frame's data
        if not coordinates3:
            print(f"Frame {frame3} for Dataset 3 has no points after filtering.")
            return
        
        # Ax3_1: Update cumulative data for dataset 3
        x3, y3 = [], []
        for frame in range(1, frame3 + 1):  # Accumulate data up to the current frame
            x3.extend([coord[0] for coord in coordinates3])
            y3.extend([coord[1] for coord in coordinates3])

        line3_1.set_data(x3, y3)
        ax3_1.set_xlabel("X [m]")
        ax3_1.set_ylabel("Y [m]")

        # Ax3_2: Update current frame data for dataset 3
        ax3_2.cla()
        ax3_2.set_xlim(*x_limits)
        ax3_2.set_ylim(*y_limits)
        ax3_2.set_xlabel("X [m]")
        ax3_2.set_ylabel("Y [m]")
        draw_grid(ax3_2, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax3_2)  # Assuming this function visualizes the sensor's area

        x3 = [coord[0] for coord in coordinates3]
        y3 = [coord[1] for coord in coordinates3]
        ax3_2.plot(x3, y3, 'ro')

        # Annotate Doppler values on the plot
        for x, y, d in zip(x3, y3, doppler3):
            ax3_2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax3_2.set_title(f"Frame {frame3} - Dataset 3")
        ax3_2.legend(["Current Frame"], loc="upper left")

        # Check if coordinates exist for the current frame
        if not coordinates3:
            print(f"Frame {frame3} for Dataset 3 has no points after filtering.")
            return

        # Calculate the occupancy grid for Dataset 3
        occupancy_grid3 = calculate_occupancy_grid(coordinates3, x_limits, y_limits, grid_spacing)

        # Clear and update ax3_3 with the occupancy grid
        ax3_3.cla()
        ax3_3.imshow(occupancy_grid3.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax3_3.set_title(f"Occupancy Grid - Dataset 3 (Frame {frame3})")
        ax3_3.set_xlabel("X [m]")
        ax3_3.set_ylabel("Y [m]")

        # Draw grid lines and sensor area
        draw_grid(ax3_3, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax3_3)  # Assuming this function visualizes the sensor's area

        # Update ax3_4: History-Based Occupancy Grid for Dataset 3
        # Calculate the cumulative occupancy grid for Dataset 3 over the last `history_frames`
        cumulative_grid3 = calculate_cumulative_occupancy(
            frames_data3, frame3, x_limits, y_limits, grid_spacing, history_frames
        )

        # Clear and update ax3_4 with the cumulative occupancy grid
        ax3_4.cla()
        ax3_4.imshow(cumulative_grid3.T, extent=(*x_limits, *y_limits), origin='lower', cmap=cmap, aspect='auto')
        ax3_4.set_title(f"History Grid - Dataset 3 (Last {history_frames} Frames)")
        ax3_4.set_xlabel("X [m]")
        ax3_4.set_ylabel("Y [m]")

        # Draw grid lines and sensor area
        draw_grid(ax3_4, x_limits, y_limits, grid_spacing)
        draw_sensor_area(ax3_4)  # Assuming this function visualizes the sensor's area

        fig.canvas.draw_idle()

    # Get the union of all frames across datasets
    all_frames = sorted(set(frames_data1.keys()) | set(frames_data2.keys()) | set(frames_data3.keys()))
    # Update the slider to cover the maximum range
    max_len = max(len(frames_data1), len(frames_data2), len(frames_data3))
    # Add slider
    # [left, bottom, width, height]
    ax_slider3 = plt.axes([0.25, 0.02, 0.65, 0.03])  # Slider for Dataset 3
    ax_slider2 = plt.axes([0.25, 0.06, 0.65, 0.03])  # Slider for Dataset 2
    ax_slider1 = plt.axes([0.25, 0.10, 0.65, 0.03])  # Slider for Dataset 1
    # Initialize sliders with respective frame ranges
    slider1 = Slider(
        ax_slider1, 
        "Frame 1", 
        min(frames_data1.keys()), 
        max(frames_data1.keys()), 
        valinit=min(frames_data1.keys()), 
        valstep=1
    )

    slider2 = Slider(
        ax_slider2, 
        "Frame 2", 
        min(frames_data2.keys()), 
        max(frames_data2.keys()), 
        valinit=min(frames_data2.keys()), 
        valstep=1
    )

    slider3 = Slider(
        ax_slider3, 
        "Frame 3", 
        min(frames_data3.keys()), 
        max(frames_data3.keys()), 
        valinit=min(frames_data3.keys()), 
        valstep=1
    )
    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)

    plt.show()


# Example Usage
# Get the absolute path to the CSV file
file_name1 = "coordinates_sl_at1.csv"  # Replace with your file path
script_dir1 = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(script_dir1, file_name1)

# Get the absolute path to the CSV file
file_name2 = "coordinates_sl_at2.csv"  # Replace with your file path
script_dir2 = os.path.dirname(os.path.abspath(__file__))
file_path2 = os.path.join(script_dir2, file_name2)

# Get the absolute path to the CSV file
file_name3 = "coordinates_sl_at3.csv"  # Replace with your file path
script_dir3 = os.path.dirname(os.path.abspath(__file__))
file_path3 = os.path.join(script_dir3, file_name3)

y_threshold = 0.0  # Disregard points with Y < num
z_threshold = (-0.30, 3.0)
doppler_threshold = 0.0 # Disregard points with doppler < num

frames_data1 = load_data(file_path1, y_threshold, z_threshold, doppler_threshold)
frames_data2 = load_data(file_path2, y_threshold, z_threshold, doppler_threshold)
frames_data3 = load_data(file_path3, y_threshold, z_threshold, doppler_threshold)

"""
Having a legen of Cluster -1, means no cluster has been created
Same as having Grey Clusters
"""
create_interactive_plots(frames_data1, frames_data2, frames_data3, x_limits=(-8, 8), y_limits=(0, 15), eps=0.4, min_samples=5, history_frames = 5)