import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Wedge

# Utility function to load data
def load_data(file_name, y_threshold=None):
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
    
    # Group data by frame and organize the output
    frames_data = {}
    for frame, group in df.groupby("Frame"):
        coordinates = list(zip(group["X [m]"], group["Y [m]"], group["Z [m]"]))
        doppler = group["Doppler [m/s]"].tolist()
        frames_data[frame] = (coordinates, doppler)
    
    return frames_data

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


# Plotting function
def create_interactive_plot(frames_data, x_limits, y_limits, grid_spacing=1):
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax1, ax2 = axes
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Create lines for ax1
    (line1,) = ax1.plot([], [], 'o', label="Data - Ax1")  # Change 'o-' to 'o'
    ax1.set_xlim(*x_limits)
    ax1.set_ylim(*y_limits)
    ax1.legend()

    # ax2 settings
    ax2.set_xlim(*x_limits)
    ax2.set_ylim(*y_limits)
    ax2.legend(["Current Frame - Ax2"], loc="upper left")

    # Function to draw the grid with specified spacing
    def draw_grid(ax, x_limits, y_limits, grid_spacing):
        x_ticks = range(int(np.floor(x_limits[0])), int(np.ceil(x_limits[1])) + 1, grid_spacing)
        y_ticks = range(int(np.floor(y_limits[0])), int(np.ceil(y_limits[1])) + 1, grid_spacing)
        for x in x_ticks:
            ax.plot([x, x], y_limits, linestyle='--', color='gray', linewidth=0.5)
        for y in y_ticks:
            ax.plot(x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)

    # Draw grids on both axes
    draw_grid(ax1, x_limits, y_limits, grid_spacing)
    draw_grid(ax2, x_limits, y_limits, grid_spacing)

    # Add slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, "Frame", 1, len(frames_data), valinit=1, valstep=1)

    # Update function
    def update(val):
        current_frame = int(slider.val)  # Get the current frame from the slider

        # Get the data for all frames up to the current frame
        x1, y1 = [], []
        for frame in range(1, current_frame + 1):
            coordinates, _ = frames_data[frame]
            x1.extend([coord[0] for coord in coordinates])
            y1.extend([coord[1] for coord in coordinates])

        # Update ax1 with cumulative data
        line1.set_data(x1, y1)

        # Update ax2 with only the current frame's data
        ax2.cla()  # Clear ax2
        ax2.set_xlim(*x_limits)  # Reset x-axis limits
        ax2.set_ylim(*y_limits)  # Reset y-axis limits
        draw_grid(ax2, x_limits, y_limits, grid_spacing)  # Redraw grid
        
        coordinates, doppler = frames_data[current_frame]
        x2 = [coord[0] for coord in coordinates]
        y2 = [coord[1] for coord in coordinates]
        ax2.plot(x2, y2, 'ro')  # Plot current frame data

        # Annotate each point with its Doppler value
        for x, y, d in zip(x2, y2, doppler):
            ax2.text(x, y, f"{d:.2f}", fontsize=8, ha="center", va="bottom", color="blue")

        ax2.set_title(f"Frame {current_frame}")
        ax2.legend(["Current Frame"], loc="upper left")

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)

    plt.show()

# Example Usage
# Get the absolute path to the CSV file
file_name = "coordinates.csv"  # Replace with your file path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)

y_threshold = 0.0  # Disregard points with Y < num

frames_data = load_data(file_path, y_threshold)

# Filter out points inside the vehicle zone
filter_vehicle_zone(
    frames_data,
    forward_distance=0.3,
    diagonal_distance=0.42,
    buffer=0.3,
    azimuth=60,
    elevation=30
)

create_interactive_plot(frames_data, x_limits=(-5, 10), y_limits=(-5, 15))