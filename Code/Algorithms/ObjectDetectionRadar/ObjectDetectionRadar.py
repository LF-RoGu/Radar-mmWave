import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

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


# Plotting function
def create_interactive_plot(frames_data, x_limits, y_limits, grid_spacing=1):
    """
    Create an interactive plot with two subplots, a slider, and radio buttons,
    including a grid with customizable spacing.
    
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
    (line1,) = ax1.plot([], [], 'o', label="Data - Ax1")
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
        
        coordinates, _ = frames_data[current_frame]
        x2 = [coord[0] for coord in coordinates]
        y2 = [coord[1] for coord in coordinates]
        ax2.plot(x2, y2, 'ro')  # Plot current frame data
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

y_threshold = 0  # Disregard points with Y < 5

frames_data = load_data(file_path, y_threshold)

create_interactive_plot(frames_data, x_limits=(-5, 10), y_limits=(-5, 15))