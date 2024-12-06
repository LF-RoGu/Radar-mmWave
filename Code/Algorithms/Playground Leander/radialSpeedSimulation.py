import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation Parameters
dt = 0.1  # Time step (seconds)
car_speed = 5.0  # Car speed in m/s (constant)
simulation_time = 10  # Total simulation time in seconds
radar_range = 50.0  # Radar detection range in meters

# Initialize car position (x, y)
car_position = np.array([0.0, 0.0])

# Object positions (x, y)
objects = np.array([
    [20.0, 5.0],
    [20.0, -5.0],
    [30.0, 5.0],
    [30.0, -5.0],
    [40.0, 5.0],
    [40.0, -5.0],
    [55.0, 0.0],
    [55.0, 10.0],
    [55.0, -10.0]
])

# Initialize figure for visualization
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
ax1, ax2, ax3 = axes

# Dictionary to store radial speeds over time for each object
radial_speed_history = {i: [] for i in range(len(objects))}
time_history = []

def radar_detection(car_pos, objects):
    """
    Simulate radar sensor that detects objects ahead within range and outputs point cloud.
    Args:
        car_pos (ndarray): Car's current position [x, y].
        objects (ndarray): Positions of the objects [[x1, y1], [x2, y2], ...].
    Returns:
        point_cloud (ndarray): Detected objects ahead as [[x, y, radial_speed], ...].
        detected_indices (list): Indices of detected objects.
    """
    point_cloud = []
    detected_indices = []
    for i, obj in enumerate(objects):
        relative_pos = obj - car_pos
        distance = np.linalg.norm(relative_pos)
        # Check if the object is within radar range and ahead of the car
        if distance <= radar_range and relative_pos[0] > 0:
            radial_speed = -car_speed * (relative_pos[0] / distance)
            point_cloud.append([relative_pos[0], relative_pos[1], radial_speed])
            detected_indices.append(i)
    return np.array(point_cloud), detected_indices

def update_car_visualization(car_pos, objects, point_cloud):
    """
    Update the car visualization with position, objects, and radar detections.
    """
    ax1.clear()
    ax1.set_xlim(-10, 70)
    ax1.set_ylim(-20, 20)
    ax1.set_title("Car and Radar Simulation")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Plot car position
    ax1.plot(car_pos[0], car_pos[1], 'bo', label="Car")

    # Plot objects
    for obj in objects:
        ax1.plot(obj[0], obj[1], 'ro', label="Object" if obj is objects[0] else "")

    # Plot radar detections
    if point_cloud.size > 0:
        for p in point_cloud:
            ax1.plot(car_pos[0] + p[0], car_pos[1] + p[1], 'gx', label="Detection" if p is point_cloud[0] else "")

    ax1.legend()

def update_detection_visualization(point_cloud):
    """
    Update the visualization of detected points in a separate plot.
    """
    ax2.clear()
    ax2.set_xlim(-10, 70)
    ax2.set_ylim(-20, 20)
    ax2.set_title("Radar Detected Points")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Plot detected points
    if point_cloud.size > 0:
        for p in point_cloud:
            ax2.plot(p[0], p[1], 'gx', label="Detection" if p is point_cloud[0] else "")

    ax2.legend()

def update_radial_speed_plot(time_history, radial_speed_history):
    """
    Update the radial speed plot for detected objects over time.
    """
    ax3.clear()
    ax3.set_title("Radial Speed Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Radial Speed (m/s)")

    # Plot radial speeds
    for obj_idx, speeds in radial_speed_history.items():
        # Filter out None values for plotting
        valid_times = [time_history[i] for i in range(len(speeds)) if speeds[i] is not None]
        valid_speeds = [speed for speed in speeds if speed is not None]
        ax3.plot(valid_times, valid_speeds, label=f"Object {obj_idx + 1}")

    #ax3.legend()

# Simulation loop
for t in np.arange(0, simulation_time, dt):
    # Move the car forward
    car_position[0] += car_speed * dt

    # Simulate radar detection
    point_cloud, detected_indices = radar_detection(car_position, objects)

    # Update radial speed history
    for i in range(len(objects)):
        if i in detected_indices:
            radial_speed = point_cloud[detected_indices.index(i)][2]
            radial_speed_history[i].append(radial_speed)
        else:
            radial_speed_history[i].append(None)  # None for missed detection

    time_history.append(t)

    # Update visualizations
    update_car_visualization(car_position, objects, point_cloud)
    update_detection_visualization(point_cloud)
    update_radial_speed_plot(time_history, radial_speed_history)

    plt.pause(0.01)

plt.ioff()
plt.show()
