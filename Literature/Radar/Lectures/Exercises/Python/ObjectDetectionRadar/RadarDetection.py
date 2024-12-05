import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
import numpy as np

# Constants
c = 3e8  # Speed of light in m/s
fc = 24e9  # Radar carrier frequency in Hz
speed_kph = 40  # Speed in km/h
speed_mps = speed_kph / 3.6  # Convert speed to m/s

# Object positions and properties
objects = [
    {'start_position': -15, 'lane': 0, 'color': 'red', 'length': 2, 'direction': 'right'},  # Object moving right
    {'start_position': 20, 'lane': 1, 'color': 'green', 'length': 2, 'direction': 'left'},  # Object moving left
    {'start_position': 10, 'lane': 1, 'color': 'orange', 'length': 4, 'direction': 'left'},  # Longer object moving left
]

radar_position = (0, 0)  # Radar at the origin (0, 0)

# Function to calculate radial velocity and Doppler shift
def calculate_doppler(object_position, object_direction, object_speed, radar_position, fc, c):
    """
    Calculate the radial velocity and Doppler shift for an object.

    Args:
    - object_position (tuple): (x, y) position of the object.
    - object_direction (str): Direction of object ('right' or 'left').
    - object_speed (float): Speed of the object in m/s.
    - radar_position (tuple): (x, y) position of the radar.
    - fc (float): Radar carrier frequency in Hz.
    - c (float): Speed of light in m/s.

    Returns:
    - radial_velocity (float): Radial velocity of the object in m/s.
    - doppler_shift (float): Doppler shift in Hz.
    """
    object_x, object_y = object_position
    radar_x, radar_y = radar_position

    # Vector from radar to object
    delta_x = object_x - radar_x
    delta_y = object_y - radar_y
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # Angle between the velocity and the radar line of sight
    alpha = np.arctan2(delta_y, delta_x)

    # Radial velocity (positive if approaching, negative if receding)
    if object_direction == 'right':
        vr = object_speed * np.cos(alpha)  # Moving to the right
    elif object_direction == 'left':
        vr = -object_speed * np.cos(alpha)  # Moving to the left

    # Doppler shift
    fd = (2 * vr * fc) / c
    return vr, fd


# Visualization and calculations
def visualize_radar_with_doppler_with_textbox(objects, radar_position, radar_distance, lane_spacing):
    """
    Visualize radar field of view, detected objects, and display Doppler calculations with a textbox below the graph.

    Args:
    - objects (list): List of object dictionaries with properties.
    - radar_position (tuple): Radar position (x, y).
    - radar_distance (float): Distance from radar to the first lane.
    - lane_spacing (float): Vertical spacing between lanes.
    """
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x_limits = [-70, 70]  # Highway bounds
    y_limits = [0, radar_distance + len(objects) * lane_spacing + 5]
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect('equal')

    # Radar visualization
    radar = Rectangle((radar_position[0] - 1, radar_position[1] - 1), 2, 2, color='blue', label='Radar')
    radar_fov = Wedge((radar_position[0], radar_position[1]), 50, 30, 150, alpha=0.2, color='blue', label='Radar FOV')
    ax.add_patch(radar)
    ax.add_patch(radar_fov)

    # Draw a single lane divider
    divider_y = radar_distance + (len(objects) - 1) * lane_spacing / 2
    ax.plot(x_limits, [divider_y, divider_y], linestyle='--', color='gray', label='Lane Divider')

    # Display objects and their Doppler calculations
    doppler_results = []
    for obj in objects:
        position = (obj['start_position'], radar_distance + obj['lane'] * lane_spacing)
        vr, fd = calculate_doppler(position, obj['direction'], speed_mps, radar_position, fc, c)
        obj_box = Rectangle(
            (obj['start_position'], position[1]),
            obj['length'],
            2,
            color=obj['color']
        )
        ax.add_patch(obj_box)

        # Annotate radial velocity and Doppler shift
        ax.text(
            obj['start_position'] + 3,
            position[1] + 1,
            f"v_r: {vr:.2f} m/s\nf_d: {fd:.2f} Hz",
            fontsize=8,
            color='black'
        )

        # Store results for textbox
        doppler_results.append(
            f"Object at {position}:\n"
            f"  Radial Velocity: {vr:.2f} m/s\n"
            f"  Doppler Shift: {fd:.2f} Hz"
        )

    # Add textbox with results below the plot
    textstr = "\n\n".join(doppler_results)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.1, 0.12, textstr, fontsize=10, bbox=props, ha='center')

    # Labels and legends
    ax.set_title("Radar Detection with Doppler Calculations")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc='upper right')

    # Adjust layout for textbox
    plt.subplots_adjust(bottom=0.05)

    # Show plot
    plt.show()


# Call the function to visualize and calculate Doppler shifts
visualize_radar_with_doppler_with_textbox(objects, radar_position, radar_distance=10, lane_spacing=2)
