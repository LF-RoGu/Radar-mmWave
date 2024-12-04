import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Simulate pedestrian cluster
def simulate_pedestrian(num_samples=50, dt=0.1):
    """
    Simulates radar-like data for a moving pedestrian cluster.
    :param num_samples: Number of samples to simulate.
    :param dt: Time step (sampling interval).
    :return: Time array, true positions, and radar cluster points.
    """
    t = np.arange(0, num_samples * dt, dt)  # Time array
    true_positions = np.array([5 + 0.5 * t, 10 - 0.3 * t]).T  # Linear motion (x, y)

    cluster_noise = np.random.normal(0, 0.2, size=(num_samples, 10, 2))  # Cluster noise
    radar_points = true_positions[:, np.newaxis, :] + cluster_noise  # Noisy radar points

    return t, true_positions, radar_points

# Kalman Filter for pedestrian tracking
class PedestrianKalmanFilter:
    def __init__(self, dt=0.1):
        """
        Initialize Kalman filter for 2D pedestrian tracking.
        :param dt: Time step (sampling interval).
        """
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],  # State transition matrix
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # Measurement matrix
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.001  # Process noise covariance
        self.R = np.eye(2) * 0.25  # Measurement noise covariance
        self.x = np.zeros((4, 1))  # Initial state [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Initial state covariance

    def update(self, z):
        """
        Perform one step of Kalman filter update.
        :param z: Measurement (2D position).
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
        :return: Future positions (x, y).
        """
        future_positions = []
        x_future = self.x.copy()
        for _ in range(steps_ahead):
            x_future = self.A @ x_future  # Predict next state
            future_positions.append(x_future[:2].flatten())  # Store [x, y]
        return np.array(future_positions)

# Main function
def main():
    dt = 0.1
    num_samples = 50
    future_steps = 20  # Predict 20 steps into the future

    # Simulate pedestrian data
    t, true_positions, radar_points = simulate_pedestrian(num_samples, dt)

    # Initialize Kalman filter
    kalman = PedestrianKalmanFilter(dt)

    # Arrays to store results
    estimated_positions = []
    predicted_trajectories = []

    for k in range(num_samples):
        # Extract pedestrian cluster
        cluster_points = radar_points[k]
        z = cluster_points.mean(axis=0)  # Cluster centroid as measurement

        # Update Kalman filter with measurement
        estimated_state, _ = kalman.update(z)
        estimated_positions.append(estimated_state[:2])

        # Predict future trajectory
        if k == num_samples - 1:  # Predict at the last sample
            predicted_trajectories = kalman.predict_future(future_steps)

    estimated_positions = np.array(estimated_positions)

    # Time array for future predictions
    t_future = np.arange(num_samples * dt, (num_samples + future_steps) * dt, dt)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label="True Position")
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r-', label="Kalman Estimate")
    plt.scatter(radar_points[:, :, 0].flatten(), radar_points[:, :, 1].flatten(),
                s=10, c='orange', label="Radar Points", alpha=0.5)
    plt.plot(predicted_trajectories[:, 0], predicted_trajectories[:, 1], 'b--', label="Predicted Trajectory")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Pedestrian Tracking and Future Trajectory Prediction")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
