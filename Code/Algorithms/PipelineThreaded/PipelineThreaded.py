import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from dataDecoderTI import DataDecoderTI


# Shared data for 3D points
points = []
lock = threading.Lock()

# Thread 1: Generate random 3D points
def generate_points():
    decoder = DataDecoderTI()
    decoder.initIWR6843("COM6", "COM7", "profile_azim60_elev30_optimized.cfg")
    
    global points
    while True:
        numFrames = decoder.pollIWR6843()

        if numFrames == 0:
            continue

        newFrames = decoder.get_and_delete_decoded_frames(numFrames)
        
        newPoints = []
        for i in range(len(newFrames)):
            frame = newFrames[i]
            for p in range(len(frame["detectedPoints"])):
                point = frame["detectedPoints"][p]
                newPoints.append((point["x"], point["y"], point["z"]))

        with lock:
            points = newPoints

# Main function
if __name__ == "__main__":
    thread1 = threading.Thread(target=generate_points)

    # Start threads
    thread1.start()
    
    plt.ion()  # Interactive mode on
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        with lock:
            ax.clear()
            if points:
                xs, ys, zs = zip(*points)
                ax.scatter(xs, ys, zs, c='blue', marker='o')
            ax.set_xlim([-5, 5])
            ax.set_ylim([0, 10])
            ax.set_zlim([-1, 10])
            ax.set_title("3D Points Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
        plt.pause(0.025)  # Allow time for the plot to refresh
    

    # Wait for threads to finish
    thread1.join()
    thread2.join()
    plt.ioff()  # Turn off interactive mode
    plt.show()
