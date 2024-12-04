import numpy as np
import matplotlib.pyplot as plt

# Simulate true trajectory
def simulate_radar_data(num_samples=1500, dt=0.02):
    """
    Simulates realistic radar data with noise.
    :param num_samples: Number of samples to simulate.
    :param dt: Time step (sampling interval).
    :return: True positions, noisy measurements, and time array.
    """
    t = np.arange(0, num_samples * dt, dt)  # Time array
    true_positions = 10 + 2 * np.sin(0.2 * t)  # True position (sinusoidal motion)
    velocity = 2 * 0.2 * np.cos(0.2 * t)  # True velocity (derivative of position)

    noise = np.random.normal(0, 0.5, size=num_samples)  # Gaussian noise (std = 0.5)
    noisy_measurements = true_positions + noise  # Noisy radar measurements

    return t, true_positions, noisy_measurements, velocity

# Kalman Filter implementation for position and velocity
class DvKalman:
    def __init__(self, dt=0.02):
        """
        Initialize Kalman filter for position and velocity estimation.
        :param dt: Time step (sampling interval).
        """
        self.dt = dt  # Sampling time
        self.A = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.Q = np.array([[0.001, 0], [0, 0.001]])  # Process noise covariance
        self.R = np.array([[0.25]])  # Measurement noise covariance
        self.x = np.array([[0], [0]])  # Initial state [position, velocity]
        self.P = np.eye(2) * 1000  # Initial state covariance (large uncertainty)

    def update(self, z):
        """
        Perform one step of Kalman filter update.
        :param z: Measurement
        :return: Estimated position, velocity, and state covariance
        """
        # Prediction
        xp = self.A @ self.x  # Predicted state
        Pp = self.A @ self.P @ self.A.T + self.Q  # Predicted covariance

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)

        # Update state and covariance with measurement
        self.x = xp + K @ (z - self.H @ xp)
        self.P = Pp - K @ self.H @ Pp

        return self.x.flatten(), self.P

# Main function to simulate and apply Kalman filter
def main():
    dt = 0.02  # Time step
    num_samples = 1500  # Number of samples

    # Simulate radar data
    t, true_positions, noisy_measurements, true_velocity = simulate_radar_data(num_samples, dt)

    # Initialize Kalman filter
    kalman = DvKalman(dt)

    # Initialize arrays to save data
    Xsaved = np.zeros((num_samples, 2))  # Estimated position and velocity
    Zsaved = noisy_measurements  # Save noisy measurements
    Psaved = np.zeros((num_samples, 2))  # State covariance

    # Run the Kalman filter
    for k in range(num_samples):
        pos_vel, P = kalman.update(Zsaved[k])
        Xsaved[k, :] = pos_vel  # Save position and velocity
        Psaved[k, :] = [P[0, 0], P[1, 1]]  # Save diagonal of covariance matrix

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot position
    plt.subplot(2, 1, 1)
    plt.plot(t, Zsaved, 'r.', markersize=10, label='Position, Noisy Measurements')
    plt.plot(t, Xsaved[:, 0], 'k-', linewidth=2, label='Position from Kalman Filter')
    plt.plot(t, true_positions, 'g-', linewidth=2, label='True Position')
    plt.ylabel("Position (m)")
    plt.title("Position and Velocity from Noisy Radar Measurements")
    plt.legend()
    plt.grid()

    # Plot velocity
    plt.subplot(2, 1, 2)
    plt.plot(t, Xsaved[:, 1], 'b-', linewidth=2, label='Velocity from Kalman Filter')
    plt.plot(t, true_velocity, 'g-', linewidth=2, label='True Velocity')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Additional plots for covariance
    plt.figure()
    plt.plot(t, Psaved[:, 0], 'k-', label='Position Covariance (P[0,0])')
    plt.xlabel("Time (s)")
    plt.ylabel("Covariance")
    plt.title("Position Covariance (P[0,0])")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t, Psaved[:, 1], 'b-', label='Velocity Covariance (P[1,1])')
    plt.xlabel("Time (s)")
    plt.ylabel("Covariance")
    plt.title("Velocity Covariance (P[1,1])")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
