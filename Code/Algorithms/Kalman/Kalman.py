import numpy as np

# Persistent variables replacement
class KalmanFilter:
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
        xp = self.A * self.x  # Prediction of state
        Pp = self.A * self.P * self.A + self.Q  # Prediction of error covariance

        # II. Compute Kalman gain
        K = Pp * self.H / (self.H * Pp * self.H + self.R)

        # III. Update state estimate
        self.x = xp + K * (z - self.H * xp)

        # IV. Update error covariance
        self.P = Pp - K * self.H * Pp

        return self.x


# Simulated voltage generator
def get_voltage():
    noise = np.random.normal(0, 4)  # Random noise with standard deviation 4
    return 14.4 + noise  # Simulated voltage with noise


# Main function to test the Kalman filter
def main():
    kalman = KalmanFilter()  # Initialize the Kalman filter
    num_samples = 20  # Number of voltage samples
    measured_voltages = []
    filtered_voltages = []

    # Simulate and filter voltages
    for _ in range(num_samples):
        z = get_voltage()  # Simulated noisy voltage measurement
        measured_voltages.append(z)
        filtered_voltages.append(kalman.update(z))

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(measured_voltages, label="Measured Voltage (Noisy)", color="orange", linestyle="dashed")
    plt.plot(filtered_voltages, label="Filtered Voltage (Kalman)", color="red")
    plt.axhline(y=14.4, color="green", linestyle="--", label="True Voltage")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.title("Kalman Filter for Voltage Measurement")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
