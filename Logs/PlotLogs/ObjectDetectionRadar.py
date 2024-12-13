import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter
import time


# Common Utility Function
def load_data(file_name):
    """
    Load CSV data from the specified file.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: File not found at {file_name}")
    return pd.read_csv(file_name)

def initialize_plot(plot_x_limits, plot_y_limits):
    """
    Initialize a plot with specified axis limits and return the figure and axis.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(plot_x_limits)
    ax.set_ylim(plot_y_limits)
    ax.set_aspect('equal')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    return fig, ax

# 1. Plot Data by Timestamp
def plot_data_by_timestamp(data, dt=0.5, clear_plot=True):
    """
    Animate data based on timestamp intervals with a delay.

    Args:
    - data (pd.DataFrame): The radar data to animate.
    - dt (float): Time delay (in seconds) between frames.
    - clear_plot (bool): If True, clears the plot between frames. If False, overlays data on the existing plot.
    """
    # Sort data by timestamp
    data = data.sort_values(by="Timestamp")

    # Extract unique timestamps
    unique_timestamps = data["Timestamp"].unique()

    # Initialize the plot
    fig, ax = initialize_plot(
        plot_x_limits=(data["X [m]"].min() - 1, data["X [m]"].max() + 1),
        plot_y_limits=(data["Y [m]"].min() - 1, data["Y [m]"].max() + 1)
    )
    scatter = ax.scatter([], [], c='blue', label="Data Points")
    ax.set_title("Radar Data Animation by Timestamp")
    ax.legend()

    # Animate by updating the scatter plot for each timestamp
    for current_time in unique_timestamps:
        if clear_plot:
            # Clear existing points and re-plot
            scatter.set_offsets(data.iloc[0:0])  # Effectively clears the scatter plot

        # Filter data for the current timestamp
        current_data = data[data["Timestamp"] == current_time]

        # Update scatter plot
        scatter.set_offsets(current_data[["X [m]", "Y [m]"]])
        ax.set_title(f"Radar Data (Time: {current_time:.2f} s)")

        # Pause for dt seconds to create an animation effect
        plt.pause(dt)

    # Keep the final frame displayed
    plt.show()



# 2. Plot Data with Slider
def plot_data_with_slider(data, clear_plot=True):
    """
    Use a slider to navigate through frames and plot the data.

    Args:
    - data (pd.DataFrame): The radar data to animate.
    - clear_plot (bool): If True, clears the plot between frames. If False, overlays data on the existing plot.
    """
    unique_frames = sorted(data["Frame"].unique())
    fig, ax = initialize_plot(data["X [m]"].min() - 1, data["X [m]"].max() + 1)
    scatter = ax.scatter([], [], label="Data Points")
    ax.set_title("Radar Data by Frame")

    def update(frame_idx):
        if clear_plot:
            scatter.set_offsets(data.iloc[0:0])  # Clear existing scatter data

        frame_data = data[data["Frame"] == unique_frames[frame_idx]]
        scatter.set_offsets(frame_data[["X [m]", "Y [m]"]])
        ax.set_title(f"Frame: {unique_frames[frame_idx]}")

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, "Frame", 0, len(unique_frames) - 1, valinit=0, valstep=1)
    slider.on_changed(lambda val: update(int(val)))

    update(0)
    plt.show()


# 3. Plot Data by FPS
def plot_data_by_fps(data, fps=10, clear_plot=True):
    """
    Plot data using frames per second (FPS).

    Args:
    - data (pd.DataFrame): The radar data to animate.
    - fps (int): Frames per second for animation.
    - clear_plot (bool): If True, clears the plot between frames. If False, overlays data on the existing plot.
    """
    unique_frames = sorted(data["Frame"].unique())
    fig, ax = initialize_plot(data["X [m]"].min() - 1, data["X [m]"].max() + 1)
    scatter = ax.scatter([], [], label="Data Points")
    ax.set_title("Radar Data Animation")

    def update(frame_idx):
        if clear_plot:
            scatter.set_offsets(data.iloc[0:0])  # Clear existing scatter data

        frame_data = data[data["Frame"] == unique_frames[frame_idx]]
        scatter.set_offsets(frame_data[["X [m]", "Y [m]"]])
        ax.set_title(f"Frame: {unique_frames[frame_idx]}")

    ani = FuncAnimation(fig, update, frames=len(unique_frames), interval=1000 / fps, repeat=True)
    plt.show()


# Example Usage
# Get the absolute path to the CSV file
file_name = "coordinates.csv"  # Replace with your file path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)
data = load_data(file_path)

# 1. Plot by timestamp
#plot_data_by_timestamp(data, dt=0.5, clear_plot=True) # Clears the plot
#plot_data_by_timestamp(data, dt=0.5, clear_plot=False) # Overlays data

# 2. Plot with slider
plot_data_with_slider(data, clear_plot=True)  # Clears the plot
#plot_data_with_slider(data, clear_plot=False)  # Overlays data

# 3. Plot by FPS
#plot_data_by_fps(data, fps=10, clear_plot=True)  # Clears the plot
#plot_data_by_fps(data, fps=10, clear_plot=False)  # Overlays data

