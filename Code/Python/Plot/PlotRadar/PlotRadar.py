import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Specify the file path (adjust for your setup)
file_path = r"\\wsl$\Ubuntu\root\.vs\Radar\out\build\linux-debug\Radar\OutputFile\detected_points.csv"

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    try:
        # Read and process the file
        data = pd.read_csv(file_path)
        
        # Clear the previous plot
        ax.cla()
        
        # Plot the points in 3D
        ax.scatter(data['x'], data['y'], data['z'], c='blue', marker='o')
        
        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Plot of Detected Points')
        
        # Pause to update the plot
        plt.pause(0.001)  # Pause for 10ms to allow the plot to update
        
        # Print data for debugging
        #print(data)
    except Exception as e:
        print(f"Error reading file: {e}")
    
    # Optional: Wait before next read (already handled by `plt.pause`)
