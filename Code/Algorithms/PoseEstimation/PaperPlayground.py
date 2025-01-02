import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- Simulation Parameters ---
time_step = 0.1  # Time step in seconds
num_steps = 100  # Total simulation steps
car_velocity = 5  # Constant car velocity in m/s
target_distance = 50  # Initial distance to the target in meters
frame_window = 5  # Number of frames to average (adjustable)

# --- Simulation of Car Motion ---
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


# --- Velocity Estimation with Frame Averaging ---
def estimate_velocity(doppler_speeds, frame_window):
    """
    Estimate the velocity of the vehicle based on Doppler speeds using frame averaging.
    Args:
        doppler_speeds (np.array): Doppler speeds for each detected point.
        frame_window (int): Number of frames to average.
    Returns:
        velocities (np.array): Estimated velocity at each time step.
    """
    velocities = []
    for i in range(len(doppler_speeds)):
        # Select up to 'frame_window' previous frames
        start_idx = max(0, i - frame_window + 1)
        frame_data = doppler_speeds[start_idx:i + 1]
        
        # Calculate median to reduce noise influence
        velocities.append(np.median(frame_data))
    return np.array(velocities)


# --- Main Simulation ---
positions, distances = simulate_car_motion()

# Generate noisy Doppler speeds
doppler_speeds = car_velocity + np.random.normal(0, 0.5, len(positions))

# Apply low-pass filter to Doppler speeds
filtered_doppler = lowpass_filter(doppler_speeds)

# Estimate velocity using frame averaging
estimated_velocity = estimate_velocity(filtered_doppler, frame_window)

# --- Plot Results ---
plt.figure(figsize=(10, 6))

# Distance to Target Plot
plt.subplot(2, 1, 1)
plt.plot(positions, distances, label='Distance to Target (m)')
plt.xlabel('Position (m)')
plt.ylabel('Distance (m)')
plt.title('Car Motion Towards Static Target')
plt.grid(True)
plt.legend()

# Doppler Speed Measurements Plot
plt.subplot(2, 1, 2)
plt.plot(positions, doppler_speeds, label='Doppler Speeds (Raw)')
plt.plot(positions, filtered_doppler, label='Doppler Speeds (Filtered)')
plt.plot(positions, estimated_velocity, label='Estimated Velocity (Frame Avg)')
plt.xlabel('Position (m)')
plt.ylabel('Speed (m/s)')
plt.title('Doppler Speed Measurements')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Estimated Vehicle Velocity (last frame): {estimated_velocity[-1]:.2f} m/s")
