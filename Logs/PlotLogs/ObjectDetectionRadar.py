import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import numpy as np

def set_grid(radar_position, plot_x_limits, plot_y_limits):
    # Initialize plot
    fig1 = plt.subplots(figsize=(12, 8))
    fig1.set_xlim(plot_x_limits)
    fig1.set_ylim(plot_y_limits)
    fig1.set_aspect('equal')

    # Add radar and centerline visualization
    radar = Rectangle((radar_position[0] - 1, radar_position[1] - 1), 2, 2, color='blue', label='Radar')
    radar_fov = Wedge((radar_position[0], radar_position[1]), 50, 30, 150, alpha=0.2, color='blue', label='Radar FOV')
    fig1.add_patch(radar)
    fig1.add_patch(radar_fov)

    # Title and labels
    fig1.set_title("Radar Data Animation")
    fig1.set_xlabel("X Position (m)")
    fig1.set_ylabel("Y Position (m)")

def obtain_coordinates_list(FrameNum, file_name="coordinates.csv"):
    # Get the absolute path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None, None, None

    # Load and process data
    data = pd.read_csv(file_path)

    # Filter data for the specified frame number
    frame_data = data[data["Frame"] == FrameNum]

    if frame_data.empty:
        print(f"No data found for frame {FrameNum}")
        return None, None, None, None

    # Extract values
    frameNum = FrameNum
    coorXYZ = frame_data[["X [m]", "Y [m]", "Z [m]"]].values.tolist()  # List of [X, Y, Z]
    timestamp = frame_data["Timestamp"].unique().tolist()  # Unique timestamps
    dopplerSpd = frame_data["Doppler [m/s]"].values.tolist()  # List of Doppler speeds

    return frameNum, coorXYZ, timestamp, dopplerSpd

# Example usage:
frameNum, coorXYZ, timestamp, dopplerSpd = obtain_coordinates_list(35, "coordinates.csv")
print(f"Frame: {frameNum}")
print(f"Coordinates (X, Y, Z): {coorXYZ}")
print(f"Timestamps: {timestamp}")
print(f"Doppler Speeds: {dopplerSpd}")
