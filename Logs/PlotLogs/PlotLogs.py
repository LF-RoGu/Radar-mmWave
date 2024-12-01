import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to your log file
log_file = os.path.join(script_dir, '..', 'azim30_elev30_static_log_2024-11-28.csv')

# Load the data
data = pd.read_csv(log_file)

# Ensure timestamps are parsed as datetime objects
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Sort the data by timestamp
data = data.sort_values('Timestamp')

# Set up 3D plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set plot limits (adjust as needed)
axis_limit = 3  # Example: 3 meters in each direction
ax.set_xlim([-axis_limit, axis_limit])
ax.set_ylim([-axis_limit, axis_limit])
ax.set_zlim([-axis_limit, axis_limit])

ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Z Coordinate (m)')
ax.set_title("Dynamic Visualization of Object Movement")

# Initialize previous timestamp
prev_time = None

for index, row in data.iterrows():
    # Extract coordinates and timestamp
    x, y, z = row['x'], row['y'], row['z']
    current_time = row['Timestamp']

    # Clear and update the plot
    ax.cla()
    ax.scatter(x, y, z, c='red', marker='o')
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.set_title("Dynamic Visualization of Object Movement")

    # Pause for the duration between timestamps
    if prev_time:
        pause_duration = (current_time - prev_time).total_seconds()
        plt.pause(max(0.01, pause_duration))
    else:
        plt.pause(0.01)

    prev_time = current_time

# Keep the plot open
plt.show()
