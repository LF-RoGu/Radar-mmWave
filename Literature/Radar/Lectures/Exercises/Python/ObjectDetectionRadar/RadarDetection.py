import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle

# Function to visualize radar points from a CSV file
def visualize_radar_with_csv(file_name, radar_position, plot_x_limits, plot_y_limits, num_frames=0):
    """
    Visualize radar field of view and plot points from a CSV file.

    Args:
    - file_name (str): Name of the CSV file in the same folder as the script.
    - radar_position (tuple): Radar position (x, y).
    - plot_x_limits (list): Limits for the X-axis.
    - plot_y_limits (list): Limits for the Y-axis.
    - num_frames (int): Number of frames to plot (0 for all frames).
    """
    # Construct full file path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    file_path = os.path.join(script_dir, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Filter data by number of frames if specified
    if num_frames > 0:
        frames_to_plot = data['Frame'].unique()[:num_frames]
        data = data[data['Frame'].isin(frames_to_plot)]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(plot_x_limits)
    ax.set_ylim(plot_y_limits)
    ax.set_aspect('equal')

    # Radar visualization
    radar = Rectangle((radar_position[0] - 1, radar_position[1] - 1), 2, 2, color='blue', label='Radar')
    radar_fov = Wedge((radar_position[0], radar_position[1]), 50, 30, 150, alpha=0.2, color='blue', label='Radar FOV')
    ax.add_patch(radar)
    ax.add_patch(radar_fov)

    # Draw grid with 1x1 meter squares
    for x in range(plot_x_limits[0], plot_x_limits[1] + 1, 1):
        ax.plot([x, x], plot_y_limits, linestyle='--', color='gray', linewidth=0.5)  # Vertical lines
    for y in range(plot_y_limits[0], plot_y_limits[1] + 1, 1):
        ax.plot(plot_x_limits, [y, y], linestyle='--', color='gray', linewidth=0.5)  # Horizontal lines

    # Plot points from the CSV file (X and Y only)
    ax.scatter(data['X [m]'], data['Y [m]'], color='red', label='Detected Points', s=20)

    # Labels and legends
    ax.set_title("Radar Detection Visualization with Points from CSV")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc='upper right')

    # Show plot
    plt.show()


# Example usage
visualize_radar_with_csv(
    file_name="coordinates.csv",
    radar_position=(0, 0),
    plot_x_limits=[-20, 20],
    plot_y_limits=[0, 10],
    num_frames=30  # Plot all frames
)
