import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize parameters
time_steps = 50
car_position = [5, 5]  # Car's cluster center (stationary or slightly moving)
pedestrian_start = [15, 10]  # Starting position of the pedestrian cluster
pedestrian_velocity = [-0.5, 0]  # Pedestrian moving from right to left
pedestrian_points = 5  # Number of points in pedestrian cluster
car_points = 20  # Number of points in car cluster

# Simulate the car cluster (stationary or slightly moving)
car_cluster = np.random.normal(loc=car_position, scale=0.2, size=(car_points, 2))

# Simulate pedestrian cluster movement over time
true_pedestrian_positions = []
for t in range(time_steps):
    pedestrian_position = pedestrian_start + np.array(pedestrian_velocity) * t
    pedestrian_cluster = np.random.normal(loc=pedestrian_position, scale=0.1, size=(pedestrian_points, 2))
    true_pedestrian_positions.append(np.mean(pedestrian_cluster, axis=0))  # Track the true center

# Add noise to simulate radar measurements
noisy_pedestrian_positions = true_pedestrian_positions + np.random.normal(0, 0.3, (time_steps, 2))

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
for measurement in noisy_pedestrian_positions:
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
ax.set_title("Tracking a Moving Pedestrian with Kalman Filter")

car_scatter = ax.scatter([], [], c="blue", label="Car Cluster")
ped_scatter = ax.scatter([], [], c="orange", label="Noisy Pedestrian Measurements")
est_line, = ax.plot([], [], "r-", label="Kalman Filter Prediction")

def update(frame):
    # Update car cluster
    car_scatter.set_offsets(car_cluster)
    
    # Update pedestrian measurements
    noisy_pedestrian = noisy_pedestrian_positions[frame]
    ped_scatter.set_offsets(noisy_pedestrian)
    
    # Update Kalman filter predictions
    est_line.set_data(estimated_positions[:frame, 0], estimated_positions[:frame, 1])
    
    return car_scatter, ped_scatter, est_line

ani = FuncAnimation(fig, update, frames=time_steps, interval=200, blit=False)

plt.legend()
plt.grid()
plt.show()
