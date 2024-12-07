import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0.0
        self.estimated_error = 1.0

    def update(self, measurement):
        # Kalman Gain
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        # Update the estimated value
        self.estimated_value = self.estimated_value + kalman_gain * (measurement - self.estimated_value)
        # Update the error covariance
        self.estimated_error = (1 - kalman_gain) * self.estimated_error + self.process_variance
        return self.estimated_value


# Simulation Parameters
dt = 0.1  # Time step (seconds)
car_speed = 5.0  # Car speed in m/s (constant)
simulation_time = 10  # Total simulation time in seconds
radar_range = 15.0  # Radar detection range in meters

# Initialize car position (x, y)
car_position = np.array([0.0, 0.0])

# Object positions (x, y)
objects = np.array([
    [10.0, 5.0],
    [10.0, -5.0],
    [12.5, 7.5],
    [12.5, -7.5],
    [15.0, 10.0],
    [15.0, -10.0],
    [17.5, 7.5],
    [17.5, -7.5],
    [20.0, 5.0],
    [20.0, -5.0],
    [22.5, 7.5],
    [22.5, -7.5],
    [25.0, 10.0],
    [25.0, -10.0],
    [27.5, 7.5],
    [27.5, -7.5],
    [30.0, 5.0],
    [30.0, -5.0],
    [32.5, 7.5],
    [32.5, -7.5],
    [35.0, 10.0],
    [35.0, -10.0],
    [37.5, 7.5],
    [37.5, -7.5],
    [40.0, 5.0],
    [40.0, -5.0],
    [42.5, 7.5],
    [42.5, -7.5],
    [45.0, 10.0],
    [45.0, -10.0],
    [47.5, 7.5],
    [47.5, -7.5],

    [50.0, 5.0],
    [50.0, -5.0],
    [55.0, 0.0],
    [55.0, 10.0],
    [55.0, -10.0],
])

# Initialize figure for visualization
plt.ion()
fig, axes = plt.subplots(4, 1, figsize=(10, 6))
ax1, ax2, ax3, ax4 = axes

# Dictionary to store radial speeds over time for each object
radial_speed_history = {i: [] for i in range(len(objects))}
time_history = []

#Adding everything for kalman filtering the self speed
self_speed_history = []
kf_self_speed_history = []
kf_self_speed = KalmanFilter(process_variance=0.01, measurement_variance=0.1)

def radar_detection_with_noise(car_pos, objects, noise_std=0.5):
    """
    Simulate radar sensor that detects objects ahead within range and outputs noisy point cloud.
    Args:
        car_pos (ndarray): Car's current position [x, y].
        objects (ndarray): Positions of the objects [[x1, y1], [x2, y2], ...].
        noise_std (float): Standard deviation of noise for x and y coordinates.
    Returns:
        point_cloud (ndarray): Detected objects as [[x, y, radial_speed], ...].
        point_cloud_noisy (ndarray): Noisy detected objects.
        detected_indices (list): Indices of detected objects.
    """
    point_cloud = []
    point_cloud_noisy = []
    detected_indices = []
    
    for i, obj in enumerate(objects):
        relative_pos = obj - car_pos
        distance = np.linalg.norm(relative_pos)
        
        # Check if the object is within radar range and ahead of the car
        if distance <= radar_range and relative_pos[0] > 0:
            radial_speed = -car_speed * (relative_pos[0] / distance)
            point_cloud.append([relative_pos[0], relative_pos[1], radial_speed])
            detected_indices.append(i)
            
            # Add noise to x and y coordinates
            noisy_x = relative_pos[0] + np.random.normal(0, noise_std)
            noisy_y = relative_pos[1] + np.random.normal(0, noise_std)
            noisy_radial_speed = -car_speed * (noisy_x / np.sqrt(noisy_x**2 + noisy_y**2))
            point_cloud_noisy.append([noisy_x, noisy_y, noisy_radial_speed])
    
    return np.array(point_cloud), np.array(point_cloud_noisy), detected_indices

def update_car_visualization(car_pos, objects, point_cloud):
    """
    Update the car visualization with position, objects, and radar detections.
    """
    ax1.clear()
    ax1.set_xlim(0, 60)
    ax1.set_ylim(-20, 20)
    ax1.set_title("Car and Radar Simulation")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Plot car position
    ax1.plot(car_pos[0], car_pos[1], 'bo', label="Car")

    # Plot objects
    for obj in objects:
        ax1.plot(obj[0], obj[1], 'ro', label="Object" if obj is objects[0] else "")

    # Plot radar detections
    if point_cloud.size > 0:
        for p in point_cloud:
            ax1.plot(car_pos[0] + p[0], car_pos[1] + p[1], 'gx', label="Detection" if p is point_cloud[0] else "")

    ax1.legend()

def update_detection_visualization(point_cloud):
    """
    Update the visualization of detected points in a separate plot.
    """
    ax2.clear()
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-20, 20)
    ax2.set_title("Radar Detected Points with noise")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Plot detected points
    if point_cloud.size > 0:
        for p in point_cloud:
            ax2.plot(p[0], p[1], 'gx', label="Detection" if p is point_cloud[0] else "")

    ax2.legend()



#Method 1: Fitting a curve into different angles and radial speeds
def estimating_self_speed(point_cloud):
    #Preparing an array to contain angle to target and radial speed
    phi_radspeed = []

    #Iterating over all points
    for i in range(len(point_cloud)):
        #Calculating the distance from car to target
        dist = np.sqrt(point_cloud[i][0]**2 + point_cloud[i][1]**2)

        #Calculating the angle to the target
        phi = np.rad2deg(np.arcsin(point_cloud[i][1]/dist))

        #Appending the angle and the radial speed 
        phi_radspeed.append([phi, point_cloud[i][2]])

    #Converting array of tuples to NumPy array
    phi_radspeed = np.array(phi_radspeed, dtype=float)

    #Fitting a first order polynominal into the points
    poly_coeff = np.polyfit(phi_radspeed[:,0], phi_radspeed[:,1], deg=2)  # Polynomial coefficients
    poly_model = np.poly1d(poly_coeff)  # Polynomial model
    phi_fit = np.linspace(-90, 90, 100)
    phi_radial_speed_fit = poly_model(phi_fit)

    #Preparing plot
    ax3.clear()
    ax3.set_xlim(-90, 90)
    ax3.set_ylim(-10, 1)
    ax3.set_title("Radial speeds vs angles")
    ax3.set_xlabel("phi (deg)")
    ax3.set_ylabel("Radial speed (m/s)")

    #Plotting all points
    for i in range(len(phi_radspeed)):
        ax3.plot(phi_radspeed[i][0], phi_radspeed[i][1], 'kx')

    #Plotting fitted curve
    ax3.plot(phi_fit, phi_radial_speed_fit)

    #Returning the self-speed after interpolating
    return poly_model(0)
    
def update_self_speed_plot(time_history, self_speed_history, kf_self_speed_history):
    ax4.clear()
    ax4.set_xlim(0, 10)
    ax4.set_ylim(-10, 0)
    ax4.set_title("Re-calculated self-speed over time")
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel("Self-speed (m/s)")

    # Plot raw self-speed
    ax4.plot(time_history, self_speed_history, label="Self-Speed 1 (Raw)", linestyle='--')

    # Plot filtered self-speed
    ax4.plot(time_history, kf_self_speed_history, label="Self-Speed 1 (Filtered)")

    #ax5.legend()


# Simulation loop
for t in np.arange(0, simulation_time, dt):
    # Move the car forward
    car_position[0] += car_speed * dt

    # Simulate radar detection
    point_cloud, point_cloud_noisy, detected_indices = radar_detection_with_noise(car_position, objects, 1.5)

    #Estimating the self_speed by both algorithms
    self_speed = estimating_self_speed(point_cloud_noisy)
    filtered_self_speed = kf_self_speed.update(self_speed)
    self_speed_history.append(self_speed)
    kf_self_speed_history.append(filtered_self_speed)

    time_history.append(t)

    # Update visualizations
    update_car_visualization(car_position, objects, point_cloud)
    update_detection_visualization(point_cloud_noisy)
    update_self_speed_plot(time_history, self_speed_history, kf_self_speed_history)

    plt.pause(0.01)

plt.ioff()
plt.show()