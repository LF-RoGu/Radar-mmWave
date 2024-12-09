import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle

# Constants for Plot Margins
PLOT_X_LIMITS = [-70, 70]  # Define X-axis limits
PLOT_Y_LIMITS = [0, 50]  # Define Y-axis limits

# Object positions and properties
objects = [
    {'start_position': -15, 'lane': 0, 'color': 'red', 'length': 2, 'direction': 'right'},  # Object moving right
    {'start_position': 20, 'lane': 1, 'color': 'green', 'length': 2, 'direction': 'left'},  # Object moving left
    {'start_position': 10, 'lane': 1, 'color': 'orange', 'length': 4, 'direction': 'left'},  # Longer object moving left
]

radar_position = (0, 0)  # Radar at the origin (0, 0)

# Visualization only
def visualize_radar_with_grid(objects, radar_position, radar_distance, lane_spacing):
    """
    Visualize radar field of view, detected objects, and draw a grid.

    Args:
    - objects (list): List of object dictionaries with properties.
    - radar_position (tuple): Radar position (x, y).
    - radar_distance (float): Distance from radar to the first lane.
    - lane_spacing (float): Vertical spacing between lanes.
    """
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(PLOT_X_LIMITS)
    ax.set_ylim(PLOT_Y_LIMITS)
    ax.set_aspect('equal')

    # Radar visualization
    radar = Rectangle((radar_position[0] - 1, radar_position[1] - 1), 2, 2, color='blue', label='Radar')
    radar_fov = Wedge((radar_position[0], radar_position[1]), 50, 30, 150, alpha=0.2, color='blue', label='Radar FOV')
    ax.add_patch(radar)
    ax.add_patch(radar_fov)

    # Draw grid with 1x1 meter squares
    for x in range(PLOT_X_LIMITS[0], PLOT_X_LIMITS[1] + 1, 1):
        ax.plot([x, x], PLOT_Y_LIMITS, linestyle='--', color='gray', linewidth=0.5)  # Vertical lines
    for y in range(PLOT_Y_LIMITS[0], PLOT_Y_LIMITS[1] + 1, 1):
        ax.plot(PLOT_X_LIMITS, [y, y], linestyle='--', color='gray', linewidth=0.5)  # Horizontal lines

    # Display objects
    for obj in objects:
        position = (obj['start_position'], radar_distance + obj['lane'] * lane_spacing)
        obj_box = Rectangle(
            (obj['start_position'], position[1]),
            obj['length'],
            2,
            color=obj['color']
        )
        ax.add_patch(obj_box)

    # Labels and legends
    ax.set_title("Radar Detection Visualization with Grid")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc='upper right')

    # Show plot
    plt.show()


# Call the function to visualize objects with grid
visualize_radar_with_grid(objects, radar_position, radar_distance=10, lane_spacing=2)
