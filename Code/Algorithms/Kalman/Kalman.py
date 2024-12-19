import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# Simulated radar data for a moving target (e.g., a vehicle)
time_steps = 50
true_positions = np.array([[t, 5 + 0.1 * t] for t in range(time_steps)])  # Simulated trajectory (x, y)
measurements = true_positions + np.random.normal(0, 1, true_positions.shape)  # Add noise to simulate radar

# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                 [0, 1, 0, 0]])
kf.R = np.eye(2) * 2  # Measurement noise
kf.Q = np.eye(4) * 0.01  # Process noise
kf.P = np.eye(4) * 10  # Initial uncertainty
kf.x = np.array([0, 0, 0, 0])  # Initial state (x, y, vx, vy)

# Run the Kalman filter
estimated_positions = []
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    estimated_positions.append(kf.x[:2])  # Store the estimated position (x, y)

estimated_positions = np.array(estimated_positions)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], label="True Position", linestyle="dashed")
plt.scatter(measurements[:, 0], measurements[:, 1], label="Measurements", color="orange", alpha=0.7)
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="Kalman Filter Estimate", color="red")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("Tracking a Target with Kalman Filter")
plt.legend()
plt.grid()
plt.show()
