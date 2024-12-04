import numpy as np

# Persistent variables replacement
class AdvancedKalmanFilter:
    def __init__(self):
        # Initialization
        self.A = 1
        self.H = 1
        self.Q = 0
        self.R = 4
        self.x = 14
        self.P = 6

    def update(self, z):
        # I. Prediction
        xp = self.A * self.x  # Predicted state
        Pp = self.A * self.P * self.A + self.Q  # Predicted error covariance

        # II. Compute Kalman gain
        K = Pp * self.H / (self.H * Pp * self.H + self.R)

        # III. Update state estimate
        self.x = xp + K * (z - self.H * xp)

        # IV. Update error covariance
        self.P = Pp - K * self.H * Pp

        return self.x, self.P, K  # Return voltage estimate, error covariance, and Kalman gain


# Simulated voltage generator
def get_voltage():
    noise = np.random.normal(0, 4)  # Random noise with standard deviation 4
    return 14.4 + noise  # Simulated voltage with noise


# Main function to test the Kalman filter
def main():
    kalman = AdvancedKalmanFilter()  # Initialize the Kalman filter
    num_samples = 500  # Number of voltage samples
    measured_voltages = []
    filtered_voltages = []
    error_covariances = []
    kalman_gains = []

    # Simulate and filter voltages
    for _ in range(num_samples):
        z = get_voltage()  # Simulated noisy voltage measurement
        measured_voltages.append(z)
        volt, Px, K = kalman.update(z)
        filtered_voltages.append(volt)
        error_covariances.append(Px)
        kalman_gains.append(K)

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Plot measured and filtered voltages
    plt.subplot(2, 1, 1)
    plt.plot(measured_voltages, label="Measured Voltage (Noisy)", color="orange", linestyle="dashed")
    plt.plot(filtered_voltages, label="Filtered Voltage (Kalman)", color="red")
    plt.axhline(y=14.4, color="green", linestyle="--", label="True Voltage")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.title("Kalman Filter for Voltage Measurement")
    plt.legend()
    plt.grid()

    # Plot Kalman gain and error covariance
    plt.subplot(2, 1, 2)
    plt.plot(error_covariances, label="Error Covariance (Px)", color="blue")
    plt.plot(kalman_gains, label="Kalman Gain (K)", color="purple")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Error Covariance and Kalman Gain")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
