import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- Simulation Parameters ---
time_step = 0.1  # Time step in seconds
num_steps = 100  # Total simulation steps
car_velocity = 5  # Constant car velocity in m/s
target_distance = 50  # Initial distance to the target in meters

def simulate_car_motion():
    """
    Simulate the car moving towards a static target.
    Returns:
        positions (np.array): Positions of the car at each time step.
        distances (np.array): Distance between car and target at each time step.
    """
    positions = np.arange(0, car_velocity * num_steps * time_step, car_velocity * time_step)
    distances = target_distance - positions
    return positions, distances

# --- Filters (Based on Paper Page 3) ---
def butter_lowpass(cutoff, fs, order=5):
    """ Design a Butterworth low-pass filter """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff=1.0, fs=10.0, order=5):
    """ Apply low-pass filter to reduce noise as suggested in the paper """
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

# --- Velocity Estimation using Doppler Measurements ---
def estimate_velocity(doppler_speeds):
    """
    Estimate the velocity of the vehicle based on Doppler speeds.
    Args:
        doppler_speeds (np.array): Doppler speeds for each detected point.
    Returns:
        float: Estimated velocity of the vehicle.
    """
    # Use the median of Doppler speeds to reduce the effect of outliers
    return np.median(doppler_speeds)


# --- Main Simulation ---
positions, distances = simulate_car_motion()

doppler_speeds = car_velocity + np.random.normal(0, 0.5, len(positions))  # Simulated Doppler readings
filtered_doppler = lowpass_filter(doppler_speeds)
estimated_velocity = estimate_velocity(filtered_doppler)

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(positions, distances, label='Distance to Target (m)')
plt.xlabel('Position (m)')
plt.ylabel('Distance (m)')
plt.title('Car Motion Towards Static Target')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(positions, doppler_speeds, label='Doppler Speeds (Raw)')
plt.plot(positions, filtered_doppler, label='Doppler Speeds (Filtered)')
plt.xlabel('Position (m)')
plt.ylabel('Speed (m/s)')
plt.title('Doppler Speed Measurements')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Estimated Vehicle Velocity: {estimated_velocity:.2f} m/s")
