import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate pedestrian cluster with 3D points
def simulate_pedestrian_with_z(num_samples=50, dt=0.1):
    """
    Simulates radar-like data for a moving pedestrian cluster with z-coordinate.
    :param num_samples: Number of samples to simulate.
    :param dt: Time step (sampling interval).
    :return: Time array, true positions, and radar cluster points.
    """
    t = np.arange(0, num_samples * dt, dt)  # Time array
    true_positions = np.array([5 + 0.5 * t, 10 - 0.3 * t]).T  # Linear motion (x, y)
    true_z = np.random.uniform(0.5, 1.8, size=num_samples)  # Z fluctuates between 0.5 and 1.8

    # Add noise for x, y, z
    cluster_noise = np.random.normal(0, 0.1, size=(num_samples, 10, 3))  # Noise for x, y, z
    radar_points = np.zeros((num_samples, 10, 3))  # Radar points with (x, y, z)

    for i in range(num_samples):
        cluster_center = np.append(true_positions[i], true_z[i])  # Combine x, y, z
        radar_points[i] = cluster_center + cluster_noise[i]  # Add noise to cluster center

    return t, np.hstack([true_positions, true_z[:, None]]), radar_points

# Kalman Filter for pedestrian tracking with z-coordinate
class PedestrianKalmanFilterWithZ:
    def __init__(self, dt=0.1):
        """
        Initialize Kalman filter for 3D pedestrian tracking.
        :param dt: Time step (sampling interval).
        """
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0, 0],  # State transition matrix
                           [0, 1, 0, dt, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])  # Z remains constant
        self.H = np.array([[1, 0, 0, 0, 0],  # Measurement matrix
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1]])  # Measurement includes x, y, z
        self.Q = np.eye(5) * 0.001  # Process noise covariance
        self.R = np.eye(3) * 0.25  # Measurement noise covariance
        self.x = np.zeros((5, 1))  # Initial state [x, y, vx, vy, z]
        self.P = np.eye(5) * 1000  # Initial state covariance

    def update(self, z):
        """
        Perform one step of Kalman filter update.
        :param z: Measurement (3D position: x, y, z).
        :return: Estimated state and covariance.
        """
        # Prediction
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)

        # Update state and covariance
        self.x = xp + K @ (z - self.H @ xp)
        self.P = Pp - K @ self.H @ Pp

        return self.x.flatten(), self.P

    def predict_future(self, steps_ahead):
        """
        Predict future positions based on the current state.
        :param steps_ahead: Number of steps ahead to predict.
        :return: Future positions and uncertainties.
        """
        future_positions = []
        future_uncertainties = []
        x_future = self.x.copy()
        P_future = self.P.copy()

        for _ in range(steps_ahead):
            # Predict next state and covariance
            x_future = self.A @ x_future
            P_future = self.A @ P_future @ self.A.T + self.Q

            future_positions.append(x_future[:3].flatten())  # Save position (x, y, z)
            future_uncertainties.append(np.diag(P_future)[:3])  # Save uncertainties for x, y, z

        return np.array(future_positions), np.array(future_uncertainties)

# Main function
def main():
    dt = 0.1
    num_samples = 50
    future_steps = 20  # Predict 20 steps into the future

    # Simulate pedestrian data
    t, true_positions, radar_points = simulate_pedestrian_with_z(num_samples, dt)

    # Initialize Kalman filter
    kalman = PedestrianKalmanFilterWithZ(dt)

    # Arrays to store results
    estimated_positions = []
    covariance_matrices = []

    for k in range(num_samples):
        # Extract pedestrian cluster
        cluster_points = radar_points[k]
        z = cluster_points.mean(axis=0)  # Cluster centroid as measurement

        # Update Kalman filter with measurement
        estimated_state, P = kalman.update(z)
        estimated_positions.append(estimated_state[:3])  # Save x, y, z
        covariance_matrices.append(np.diag(P))  # Save diagonal elements of covariance matrix

    estimated_positions = np.array(estimated_positions)
    covariance_matrices = np.array(covariance_matrices)

    # Predict future positions
    future_positions, future_uncertainties = kalman.predict_future(future_steps)
    t_future = np.arange(num_samples * dt, (num_samples + future_steps) * dt, dt)

    # 3D Plot of results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot radar points
    for i in range(num_samples):
        ax.scatter(radar_points[i, :, 0], radar_points[i, :, 1], radar_points[i, :, 2],
                   c='orange', s=10, alpha=0.5, label='Radar Points' if i == 0 else "")

    # Plot true positions
    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2],
            'g-', label="True Trajectory")

    # Plot Kalman filter estimates
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2],
            'r-', label="Kalman Estimate")

    # Plot future predictions
    ax.plot(future_positions[:, 0], future_positions[:, 1], future_positions[:, 2],
            'b--', label="Future Predictions")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("3D Pedestrian Tracking with Kalman Filter and Predictions")
    ax.legend()
    plt.show()

    # Plot error covariance
    plt.figure(figsize=(12, 6))
    plt.plot(t, covariance_matrices[:, 0], label="Position Covariance (X)", color="blue")
    plt.plot(t, covariance_matrices[:, 1], label="Position Covariance (Y)", color="green")
    plt.plot(t, covariance_matrices[:, 2], label="Position Covariance (Z)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Covariance")
    plt.title("Error Covariance Over Time")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
