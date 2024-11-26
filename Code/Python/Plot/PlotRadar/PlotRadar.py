import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Specify the file path (adjust for your setup)
file_path = r"\\wsl$\Ubuntu\root\.vs\Radar\out\build\linux-debug\Radar\OutputFile\detected_points.csv"

# Fixed range for the axes (3 meters in each direction)
axis_limit = 3 #mts

# Threshold for determining movement
doppler_threshold = 0.1  # Adjust based on your Doppler data's precision

# Initialize plots
fig = plt.figure(figsize=(12, 6))

# Create subplots
ax_stationary = fig.add_subplot(121, projection='3d', title="Stationary Objects")
ax_moving = fig.add_subplot(122, projection='3d', title="Moving Objects")

while True:
    try:
        # Read and process the file
        data = pd.read_csv(file_path)

        # Separate stationary and moving points
        stationary_points = data[abs(data['doppler']) <= doppler_threshold]
        moving_points = data[abs(data['doppler']) > doppler_threshold]

        # Clear previous plots
        ax_stationary.cla()
        ax_moving.cla()

        # Plot stationary objects
        if not stationary_points.empty:
            ax_stationary.scatter(
                stationary_points['x'], stationary_points['y'], stationary_points['z'], c='green', marker='o'
            )
        ax_stationary.set_xlabel('X Coordinate (m)')
        ax_stationary.set_ylabel('Y Coordinate (m)')
        ax_stationary.set_zlabel('Z Coordinate (m)')
        ax_stationary.set_xlim([-axis_limit, axis_limit])
        ax_stationary.set_ylim([-axis_limit, axis_limit])
        ax_stationary.set_zlim([-axis_limit, axis_limit])
        ax_stationary.set_title("Stationary Objects")

        # Plot moving objects
        if not moving_points.empty:
            ax_moving.scatter(
                moving_points['x'], moving_points['y'], moving_points['z'], c='red', marker='o'
            )
        ax_moving.set_xlabel('X Coordinate (m)')
        ax_moving.set_ylabel('Y Coordinate (m)')
        ax_moving.set_zlabel('Z Coordinate (m)')
        ax_moving.set_xlim([-axis_limit, axis_limit])
        ax_moving.set_ylim([-axis_limit, axis_limit])
        ax_moving.set_zlim([-axis_limit, axis_limit])
        ax_moving.set_title("Moving Objects")

        # Pause to update the plots
        plt.pause(0.001)  # Pause for 1ms to allow the plots to update

        # Print data for debugging
        #print("Stationary Points:")
        #print(stationary_points)
        #print("Moving Points:")
        #print(moving_points)

    except Exception as e:
        print(f"Error reading file: {e}")

    # Optional: Wait before next read (already handled by `plt.pause`)
