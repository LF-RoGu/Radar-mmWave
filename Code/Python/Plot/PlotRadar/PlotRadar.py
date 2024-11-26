import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# Constants
FILE_PATH = r"\\wsl$\Ubuntu\root\.vs\Radar\out\build\linux-debug\Radar\OutputFile\detected_points.csv"
AXIS_LIMIT = 3  # Meters
DOPPLER_THRESHOLD = 0.1  # Doppler threshold to distinguish stationary and moving points
PLOT_UPDATE_DELAY = 0.001  # Delay for plot updates in seconds
BOUNDING_BOX_ALPHA = 0.2  # Transparency for bounding boxes

# Initialize plots
FIG_SIZE = (12, 6)
fig = plt.figure(figsize=FIG_SIZE)

# Create subplots
ax_stationary = fig.add_subplot(121, projection='3d', title="Stationary Objects")
ax_moving = fig.add_subplot(122, projection='3d', title="Moving Objects")

def draw_bounding_box(ax, points, color='blue'):
    """Draw a bounding box around the points."""
    if points.empty:
        return
    # Compute bounding box
    x_min, x_max = points['x'].min(), points['x'].max()
    y_min, y_max = points['y'].min(), points['y'].max()
    z_min, z_max = points['z'].min(), points['z'].max()

    # Define vertices of the box
    vertices = [
        [x_min, y_min, z_min],
        [x_min, y_max, z_min],
        [x_max, y_max, z_min],
        [x_max, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max],
        [x_max, y_min, z_max],
    ]

    # Define the edges of the box
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Left face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Right face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Front face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Back face
    ]

    # Add the box to the plot
    box = Poly3DCollection(edges, alpha=BOUNDING_BOX_ALPHA, edgecolor=color, facecolor=color)
    ax.add_collection3d(box)

while True:
    try:
        # Read and process the file
        data = pd.read_csv(FILE_PATH)

        # Separate stationary and moving points
        stationary_points = data[abs(data['doppler']) <= DOPPLER_THRESHOLD]
        moving_points = data[abs(data['doppler']) > DOPPLER_THRESHOLD]

        # Clear previous plots
        ax_stationary.cla()
        ax_moving.cla()

        # Plot stationary objects
        if not stationary_points.empty:
            ax_stationary.scatter(
                stationary_points['x'], stationary_points['y'], stationary_points['z'], c='green', marker='o'
            )
            draw_bounding_box(ax_stationary, stationary_points, color='green')
        ax_stationary.set_xlabel('X Coordinate (m)')
        ax_stationary.set_ylabel('Y Coordinate (m)')
        ax_stationary.set_zlabel('Z Coordinate (m)')
        ax_stationary.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_stationary.set_title("Stationary Objects")

        # Plot moving objects
        if not moving_points.empty:
            ax_moving.scatter(
                moving_points['x'], moving_points['y'], moving_points['z'], c='red', marker='o'
            )
            draw_bounding_box(ax_moving, moving_points, color='red')
        ax_moving.set_xlabel('X Coordinate (m)')
        ax_moving.set_ylabel('Y Coordinate (m)')
        ax_moving.set_zlabel('Z Coordinate (m)')
        ax_moving.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax_moving.set_title("Moving Objects")

        # Pause to update the plots
        plt.pause(PLOT_UPDATE_DELAY)  # Pause for a short time to allow the plots to update

    except Exception as e:
        print(f"Error reading file: {e}")
