import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate synthetic 3D motion data
def generate_3d_motion_data(num_points=50, noise_std=1.0):
    t = np.linspace(0, 10, num_points)
    x = 10 * np.sin(t)
    y = 10 * np.cos(t)
    z = t
    # Add noise
    x_noisy = x + np.random.normal(0, noise_std, num_points)
    y_noisy = y + np.random.normal(0, noise_std, num_points)
    z_noisy = z + np.random.normal(0, noise_std, num_points)
    return np.stack([x, y, z], axis=1), np.stack([x_noisy, y_noisy, z_noisy], axis=1)

# Kalman Filter implementation for 3D motion tracking
class KalmanFilter3D:
    def __init__(self):
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        # State covariance matrix
        self.P = np.eye(6) * 500
        # Transition matrix (for constant velocity model)
        dt = 1.0
        self.F = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ])
        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        # Measurement noise covariance
        self.R = np.eye(3) * 4
        # Process noise covariance
        self.Q = np.eye(6) * 0.1

    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update the state with the new measurement
        y = measurement - self.H @ self.state
        self.state += K @ y
        # Update the state covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.state[:3]

# Generate synthetic motion data
true_positions, noisy_positions = generate_3d_motion_data()

# Apply Kalman Filter
kf = KalmanFilter3D()
filtered_positions = []

for measurement in noisy_positions:
    kf.predict()
    kf.update(measurement)
    filtered_positions.append(kf.get_state())

filtered_positions = np.array(filtered_positions)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='True Path', linestyle='dashed')
ax.scatter(noisy_positions[:, 0], noisy_positions[:, 1], noisy_positions[:, 2], label='Noisy Measurements', alpha=0.5)
ax.plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2], label='Kalman Filter Output')
ax.set_title('3D Motion Tracking with Kalman Filter')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()