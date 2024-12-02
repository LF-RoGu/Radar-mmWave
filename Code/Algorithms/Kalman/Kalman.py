import numpy as np
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize parameters
time_steps = 50
car_position = [5, 5]  # Car's cluster center (stationary or slightly moving)
pedestrian_start = [15, 10]  # Starting position of the pedestrian cluster
pedestrian_velocity = [-0.5, 0]  # Pedestrian moving from right to left
pedestrian_points = 5  # Increased number of points in pedestrian cluster
car_points = 20  # Number of points in car cluster
noise_points = 100  # Number of random noise points

# Simulate the car cluster (stationary or slightly moving)
car_cluster = np.random.normal(loc=car_position, scale=0.2, size=(car_points, 2))

# Simulate pedestrian cluster movement over time
true_pedestrian_positions = []
pedestrian_clusters = []
for t in range(time_steps):
    pedestrian_position = pedestrian_start + np.array(pedestrian_velocity) * t
    pedestrian_cluster = np.random.normal(loc=pedestrian_position, scale=0.3, size=(pedestrian_points, 2))
    pedestrian_clusters.append(pedestrian_cluster)
    true_pedestrian_positions.append(np.mean(pedestrian_cluster, axis=0))  # Track the true center

# Add random noise to simulate radar clutter
noise = np.random.uniform(low=0, high=20, size=(time_steps, noise_points, 2))

# Apply DBSCAN clustering to filter noise
eps = 1.5  # Maximum distance between points in a cluster
min_samples = 5  # Minimum points to form a cluster
filtered_pedestrian_positions = []
for t in range(time_steps):
    combined_data = np.vstack([pedestrian_clusters[t], car_cluster, noise[t]])
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(combined_data)
    
    # Identify the pedestrian cluster (largest cluster after removing noise and car cluster)
    unique_labels, counts = np.unique(labels, return_counts=True)
    pedestrian_label = unique_labels[np.argmax(counts[unique_labels != -1])]  # Largest valid cluster
    pedestrian_cluster = combined_data[labels == pedestrian_label]
    
    # Store the pedestrian cluster's mean as the filtered measurement
    filtered_pedestrian_positions.append(np.mean(pedestrian_cluster, axis=0))

# Initialize Kalman filter for pedestrian tracking
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                 [0, 1, 0, 0]])
kf.R = np.eye(2) * 0.5  # Measurement noise
kf.Q = np.eye(4) * 0.01  # Process noise
kf.P = np.eye(4) * 10  # Initial uncertainty
kf.x = np.array([pedestrian_start[0], pedestrian_start[1], 0, 0])  # Initial state (x, y, vx, vy)

# Track pedestrian positions with Kalman filter
estimated_positions = []
for measurement in filtered_pedestrian_positions:
    kf.predict()
    kf.update(measurement)
    estimated_positions.append(kf.x[:2])  # Store estimated pedestrian position

estimated_positions = np.array(estimated_positions)

# Create an animated plot to visualize the movement
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.set_xlabel("X Coordinate (m)")
ax.set_ylabel("Y Coordinate (m)")
ax.set_title("Tracking a Moving Pedestrian with Kalman Filter and DBSCAN")

car_scatter = ax.scatter([], [], c="blue", label="Car Cluster")
noise_scatter = ax.scatter([], [], c="gray", label="Noise")
ped_scatter = ax.scatter([], [], c="orange", label="Filtered Pedestrian")
est_line, = ax.plot([], [], "r-", label="Kalman Filter Prediction")

def update(frame):
    # Update car cluster
    car_scatter.set_offsets(car_cluster)
    
    # Update noise points
    noise_scatter.set_offsets(noise[frame])
    
    # Update filtered pedestrian cluster
    ped_cluster = pedestrian_clusters[frame]
    filtered_pedestrian = filtered_pedestrian_positions[frame]
    ped_scatter.set_offsets(ped_cluster)
    
    # Update Kalman filter predictions
    est_line.set_data(estimated_positions[:frame, 0], estimated_positions[:frame, 1])
    
    return car_scatter, noise_scatter, ped_scatter, est_line

ani = FuncAnimation(fig, update, frames=time_steps, interval=200, blit=False)

plt.legend()
plt.grid()
plt.show()
