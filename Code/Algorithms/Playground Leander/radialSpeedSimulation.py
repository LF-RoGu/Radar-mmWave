import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulation Parameters
dt = 0.1  # Time step (seconds)
car_speed = 5.0  # Car speed in m/s (constant)
simulation_time = 10  # Total simulation time in seconds
radar_range = 50.0  # Radar detection range in meters

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
    #[55.0, 0.0],
    [55.0, 10.0],
    [55.0, -10.0]
])

# Initialize figure for visualization
plt.ion()
fig, axes = plt.subplots(5, 1, figsize=(10, 6))
ax1, ax2, ax3, ax4, ax5 = axes

# Dictionary to store radial speeds over time for each object
radial_speed_history = {i: [] for i in range(len(objects))}
time_history = []

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
    ax2.set_title("Radar Detected Points")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    # Plot detected points
    if point_cloud.size > 0:
        for p in point_cloud:
            ax2.plot(p[0], p[1], 'gx', label="Detection" if p is point_cloud[0] else "")

    ax2.legend()

def update_radial_speed_plot(time_history, radial_speed_history):
    """
    Update the radial speed plot for detected objects over time.
    """
    ax3.clear()
    ax3.set_title("Radial Speed Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Radial Speed (m/s)")

    # Plot radial speeds
    for obj_idx, speeds in radial_speed_history.items():
        # Filter out None values for plotting
        valid_times = [time_history[i] for i in range(len(speeds)) if speeds[i] is not None]
        valid_speeds = [speed for speed in speeds if speed is not None]
        ax3.plot(valid_times, valid_speeds, label=f"Object {obj_idx + 1}")

    #ax3.legend()



def model3(params, x):
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d

# Define the objective function (sum of absolute residuals)
def objective3(params, x, y):
    predictions = model3(params, x)
    residuals = np.abs(y - predictions)
    return np.sum(residuals)

#Method 1 begin: Fitting a curve into different angles and radial speeds
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

    #Fitting the a curve into the points
    initial_guess = [0.0, 0.0, 0.0, 0.0]
    result = minimize(objective3, initial_guess, args=(phi_radspeed[:,0], phi_radspeed[:,1]), method='Powell')
    optimized_params = result.x
    phi_fit = np.linspace(-90, 90, 100)
    radial_speed_fit = model3(optimized_params, phi_fit)

    #Preparing plot
    ax4.clear()
    ax4.set_xlim(-90, 90)
    ax4.set_ylim(-10, 1)
    ax4.set_title("Radial speeds vs angles")
    ax4.set_xlabel("phi (deg)")
    ax4.set_ylabel("Radial speed (m/s)")

    #Plotting all points
    for i in range(len(phi_radspeed)):
        ax4.plot(phi_radspeed[i][0], phi_radspeed[i][1], 'kx')

    #Plotting fitted curve
    ax4.plot(phi_fit, radial_speed_fit)

#Method 2: Recalculating self-speed from different angles before fitting
def estimating_self_speed2(point_cloud):
    #Preparing an array to contain phi and the re-calculated self-speed
    phi_selfspeed = []
    
    #Iterating over all points
    for i in range(len(point_cloud)):
        #Calculating the distance from car to target
        dist = np.sqrt(point_cloud[i][0]**2 + point_cloud[i][1]**2)

        #Calculating the angle to the target
        phi = np.rad2deg(np.arcsin(point_cloud[i][1]/dist))
        
        #Re-calculating the self-speed from phi and radial speed of target
        selfspeed = point_cloud[i][2]/np.cos(np.deg2rad(phi))
        phi_selfspeed.append([phi, selfspeed])
    
    #Converting array of tuples to NumPy array
    phi_selfspeed = np.array(phi_selfspeed)
    
    #Fitting the a curve into the points
    poly_coeff = np.polyfit(phi_selfspeed[:,0], phi_selfspeed[:,1], deg=1)  # Polynomial coefficients
    poly_model = np.poly1d(poly_coeff)  # Polynomial model
    phi_fit = np.linspace(-90, 90, 100)
    radial_speed_fit = poly_model(phi_fit)

    #Preparing plot
    ax5.clear()
    ax5.set_xlim(-90, 90)
    ax5.set_ylim(-10, 1)
    ax5.set_title("Re-calculated self-speed vs angles")
    ax5.set_xlabel("phi (deg)")
    ax5.set_ylabel("Self-speed(m/s)")

    #Plotting all points
    for i in range(len(phi_selfspeed)):
        ax5.plot(phi_selfspeed[i][0], phi_selfspeed[i][1], 'kx')

    #Plotting fitted curve
    ax5.plot(phi_fit, radial_speed_fit)
    


    


# Simulation loop
for t in np.arange(0, simulation_time, dt):
    # Move the car forward
    car_position[0] += car_speed * dt

    # Simulate radar detection
    point_cloud, point_cloud_noisy, detected_indices = radar_detection_with_noise(car_position, objects, 1)

    #Estimating the self_speed by both algorithms
    estimating_self_speed(point_cloud_noisy)
    estimating_self_speed2(point_cloud_noisy)


    # Update radial speed history
    for i in range(len(objects)):
        if i in detected_indices:
            radial_speed = point_cloud[detected_indices.index(i)][2]
            radial_speed_history[i].append(radial_speed)
        else:
            radial_speed_history[i].append(None)  # None for missed detection

    time_history.append(t)

    # Update visualizations
    update_car_visualization(car_position, objects, point_cloud)
    update_detection_visualization(point_cloud)
    update_radial_speed_plot(time_history, radial_speed_history)

    plt.pause(0.01)

plt.ioff()
plt.show()