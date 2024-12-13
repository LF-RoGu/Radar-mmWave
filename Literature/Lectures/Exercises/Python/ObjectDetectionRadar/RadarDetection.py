import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import numpy as np
from scipy.optimize import curve_fit

# Constants
c = 3e8  # Speed of light (m/s)
f = 24e9  # Radar carrier frequency (Hz)
v_s = 5  # Speed of the square (m/s)

# Dots configuration
dots_start_x = 20
num_dots = 2*dots_start_x

# Main configuration
plot_x_limits = [0, 60]
plot_y_limits = [0, 10]
grid_spacing = 1

# Square configuration
square_config = {
    'width': 3,
    'height': 1,
    'start_x': 0,
    'start_y': 5,
    'color': 'blue'
}

# Wedge configuration
wedge_config = {
    'radius': 12,
    'angle': 30,
    'start_x': square_config['start_x'] + square_config['width'],
    'start_y': square_config['start_y'] + square_config['height'] / 2,
    'color': 'red',
    'alpha': 0.5
}

NUM_SLICES = 500  # Number of angular slices
angular_slices = np.linspace(-wedge_config['angle'] / 2, wedge_config['angle'] / 2, NUM_SLICES)
slice_width = angular_slices[1] - angular_slices[0]  # Width of each slice

# Generate random dots
dots = [
    (random.uniform(dots_start_x, plot_x_limits[1]), random.uniform(plot_y_limits[0], plot_y_limits[1]))
    for _ in range(num_dots)
]

# Detected dots and their Kalman filters
detected_dots = []
displayed_dots = set()  # To track dots whose Doppler results have already been displayed
kalman_filters = {}
radial_speeds_over_time = {}  # Store radial speed history for each dot

# Create the figure and axes
fig, (ax_main, ax_curve, ax_time) = plt.subplots(3, 1, figsize=(10, 12))

ax_main.set_title("Car and Radar Simulation")
ax_main.set_xlim(plot_x_limits[0], plot_x_limits[1])
ax_main.set_ylim(plot_y_limits[0], plot_y_limits[1])

ax_curve.set_title("Radial Speed vs Angle")
ax_curve.set_xlabel("Angle (degrees)")
ax_curve.set_ylabel("Radial Speed (m/s)")
ax_curve.set_xlim(-90, 90)
ax_curve.set_ylim(-10, 10)

ax_time.set_title("Radial Speed vs Time")
ax_time.set_xlabel("Time (frames)")
ax_time.set_ylabel("Radial Speed (m/s)")

# Plot the dots
dot_plots = {}
for dot in dots:
    dot_plots[dot] = ax_main.plot(dot[0], dot[1], 'o', color='green')  # Static dots in green

# Add square and wedge
square = patches.Rectangle(
    (square_config['start_x'], square_config['start_y']),
    square_config['width'], square_config['height'],
    color=square_config['color']
)
ax_main.add_patch(square)

wedge = patches.Wedge(
    (wedge_config['start_x'], wedge_config['start_y']),
    wedge_config['radius'],
    -wedge_config['angle'] / 2,
    wedge_config['angle'] / 2,
    color=wedge_config['color'],
    alpha=wedge_config['alpha']
)
ax_main.add_patch(wedge)

def calculate_exponential_speed(frame, v_start=2, v_max=2, growth_rate=0.05):
    """
    Calculate the exponentially increasing speed for the car.

    Parameters:
    - frame: Current frame number.
    - v_start: Initial speed (m/s).
    - v_max: Maximum speed (m/s).
    - growth_rate: Exponential growth rate.

    Returns:
    - Current speed of the car (m/s).
    """
    return v_start + (v_max - v_start) * (1 - np.exp(-growth_rate * frame))


# Define the cosine model
def cosine_model(phi, A, phi0, B):
    """
    Cosine model for radial speed as a function of angle.

    Parameters:
    - phi: Angle in degrees.
    - A: Amplitude (max radial speed).
    - phi0: Phase shift (alignment offset in degrees).
    - B: Baseline radial speed.

    Returns:
    - Radial speed at the given angle.
    """
    phi_rad = np.radians(phi)  # Convert to radians for cosine
    return A * np.cos(phi_rad + np.radians(phi0)) + B

# Function to fit cosine model and estimate self-speed
def estimating_self_speed_cosine(detected_dots):
    """
    Fit a cosine model to the radial speed vs angle data from detected points.

    Parameters:
    - detected_dots: List of tuples (x, y, radial_speed).

    Returns:
    - phi_fit: Array of angles for the fitted curve.
    - radial_speed_fit: Array of radial speeds for the fitted curve.
    """
    if not detected_dots or len(detected_dots) < 3:  # Ensure enough points
        return None, None

    # Calculate angles (phi) and radial speeds
    phi_radspeed = []
    for dot in detected_dots:
        dist = np.sqrt(dot[0]**2 + dot[1]**2)  # Distance to the dot
        phi = np.rad2deg(np.arcsin(dot[1] / dist))  # Angle in degrees
        phi_radspeed.append([phi, dot[2]])  # Angle and radial speed

    phi_radspeed = np.array(phi_radspeed)

    # Perform curve fitting using the cosine model
    phi_data = phi_radspeed[:, 0]
    radial_speed_data = phi_radspeed[:, 1]
    initial_guess = [1.0, 0.0, 0.0]  # Initial guess for [A, phi0, B]

    try:
        params, _ = curve_fit(cosine_model, phi_data, radial_speed_data, p0=initial_guess)
    except RuntimeError:
        # Handle fitting errors gracefully
        return None, None

    # Generate the fitted curve
    phi_fit = np.linspace(-90, 90, 100)
    radial_speed_fit = cosine_model(phi_fit, *params)

    return phi_fit, radial_speed_fit


def create_kalman_filter(initial_x, initial_y):
    """Create and initialize a Kalman filter for a dot."""
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.P *= 10  # Initial covariance matrix
    kf.R = np.array([[0.1, 0],  # Measurement noise
                     [0, 0.1]])
    kf.Q = np.eye(4) * 0.01  # Process noise
    kf.x = np.array([[initial_x],  # Initial state
                     [initial_y],
                     [0],
                     [0]])
    return kf

def calculate_doppler_and_radial_speed(square_position, dot_position):
    """
    Calculate the radial speed and Doppler shift for a detected dot.

    Parameters:
    - square_position: Tuple (x, y) of the square's position.
    - dot_position: Tuple (x, y) of the dot's position.

    Returns:
    - radial_speed: Radial speed (m/s) of the dot relative to the square.
    - doppler_shift: Doppler frequency shift (Hz) due to the radial speed.
    """
    # Relative position vector
    dx = dot_position[0] - square_position[0]
    dy = dot_position[1] - square_position[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero for extremely close dots
    if distance == 0:
        return 0, 0

    # Angle between the radar's motion and the line to the dot
    theta = np.arctan2(dy, dx)
    
    # Radial speed (m/s)
    radial_speed = v_s * np.cos(theta)
    
    # Doppler frequency shift (Hz)
    doppler_shift = (2 * f * radial_speed) / c

    return radial_speed, doppler_shift

def is_point_in_wedge(point, wedge_center, wedge_radius, wedge_angle, wedge_direction):
    """Check if a point is within the wedge's detection area."""
    px, py = point
    cx, cy = wedge_center
    dx, dy = px - cx, py - cy
    distance = np.sqrt(dx**2 + dy**2)
    if distance > wedge_radius:
        return False

    angle_to_point = np.degrees(np.arctan2(dy, dx))
    relative_angle = (angle_to_point - wedge_direction) % 360
    if relative_angle > 180:
        relative_angle -= 360

    return -wedge_angle / 2 <= relative_angle <= wedge_angle / 2

def is_point_in_wedge_with_shadow(point, wedge_center, wedge_radius, wedge_angle, slice_width, closest_points):
    """
    Check if a point is in the wedge and simulate shadowing.

    Parameters:
    - point: (x, y) position of the object.
    - wedge_center: (x, y) center of the wedge.
    - wedge_radius: Maximum detection range of the wedge.
    - wedge_angle: Total angular range of the wedge (degrees).
    - slice_width: Angular width of each slice (degrees).
    - closest_points: Dictionary of the closest detected object for each angular slice.

    Returns:
    - True if the point is detected (not occluded); False otherwise.
    """
    px, py = point
    cx, cy = wedge_center

    # Calculate the relative position
    dx, dy = px - cx, py - cy
    distance = np.sqrt(dx**2 + dy**2)

    # Ignore points outside the wedge's radius
    if distance > wedge_radius:
        return False

    # Calculate the angle of the point relative to the wedge center
    angle_to_point = np.degrees(np.arctan2(dy, dx))

    # Check if the angle is within the wedge's range
    if not (-wedge_angle / 2 <= angle_to_point <= wedge_angle / 2):
        return False

    # Determine which slice the point belongs to
    slice_index = int((angle_to_point + wedge_angle / 2) / slice_width)

    # Simulate shadowing: Check if this point is closer than the current closest
    if slice_index in closest_points:
        if distance >= closest_points[slice_index]['distance']:
            return False  # Occluded by a closer point

    # Update the closest point for this slice
    closest_points[slice_index] = {'distance': distance, 'point': point}
    return True

def add_shadow(ax, detected_point, wedge_center, wedge_radius, slice_width, angle_to_point):
    """
    Add a shadow wedge to the plot to simulate occlusion.

    Parameters:
    - ax: The Matplotlib axis to draw on.
    - detected_point: (x, y) coordinates of the detected object.
    - wedge_center: (x, y) center of the wedge.
    - wedge_radius: Maximum radius of the wedge.
    - slice_width: Width of an angular slice (degrees).
    - angle_to_point: Angle of the detected object relative to the wedge center.

    Returns:
    - A Wedge patch representing the shadow.
    """
    # Calculate shadow start and end angles
    start_angle = angle_to_point - slice_width / 2
    end_angle = angle_to_point + slice_width / 2

    # Create a shadow wedge segment
    shadow_wedge = patches.Wedge(
        center=wedge_center,
        r=wedge_radius,
        theta1=start_angle,
        theta2=end_angle,
        color='gray',
        alpha=0.3  # Semi-transparent to show occlusion
    )
    ax.add_patch(shadow_wedge)
    return shadow_wedge

def update(frame):
    """
    Update function for the animation. Detects and tracks dots using Kalman Filter,
    calculates Doppler shift and radial speed for detected dots, and marks detected dots with crosses.
    """
    global detected_dots, displayed_dots, kalman_filters

    # Calculate the car's current speed
    v_s = calculate_exponential_speed(frame)

    # Move the square
    square.set_x(frame * v_s)  # Use speed to determine position

    # Move the wedge
    wedge_center = ((frame * v_s) + square_config['width'], wedge_config['start_y'])
    wedge.set_center(wedge_center)

    # Simulate shadowing
    closest_points = {}  # Track the closest points in each angular slice

    # Check for new detections
    new_detections = [
        dot for dot in dots if is_point_in_wedge_with_shadow(
            dot, wedge_center, wedge_config['radius'], wedge_config['angle'], slice_width, closest_points
        )
    ]

    for dot in new_detections:
        if dot not in detected_dots:
            detected_dots.append(dot)
            if dot not in radial_speeds_over_time:
                radial_speeds_over_time[dot] = []  # Initialize radial speed history

            # Calculate angle to the detected point
            dx, dy = dot[0] - wedge_center[0], dot[1] - wedge_center[1]
            angle_to_point = np.degrees(np.arctan2(dy, dx))

            # Add a shadow for this detection
            shadow_wedge = add_shadow(ax_main, dot, wedge_center, wedge_config['radius'], slice_width, angle_to_point)

            ax_main.plot(dot[0], dot[1], 'x', color='red')  # Mark as detected


    # Predict and update Kalman filters for tracked dots and calculate Doppler effect
    for dot, kf in kalman_filters.items():
        if dot not in displayed_dots:
            kf.predict()
            pred_x, pred_y = kf.x[0, 0], kf.x[1, 0]

            # Calculate Doppler effect and radial speed
            radial_speed, doppler_shift = calculate_doppler_and_radial_speed(wedge_center, (pred_x, pred_y))
            print(f"Dot at {dot}: Radial Speed = {radial_speed:.2f} m/s, Doppler Shift = {doppler_shift:.2f} Hz")

            # Add the dot to the displayed set
            displayed_dots.add(dot)

    # Calculate radial speeds for detected dots
    detected_with_radial_speeds = []
    for dot in detected_dots:
        dist = np.sqrt(dot[0]**2 + dot[1]**2)
        phi = np.rad2deg(np.arcsin(dot[1] / dist))
        radial_speed, _ = calculate_doppler_and_radial_speed(wedge_center, dot)
        detected_with_radial_speeds.append((dot[0], dot[1], radial_speed))
        # Update radial speed history
        if dot not in radial_speeds_over_time:
            radial_speeds_over_time[dot] = []  # Initialize for new dots
        radial_speeds_over_time[dot].append((frame, radial_speed))



    # Fit and plot the curve
    phi_fit, radial_speed_fit = estimating_self_speed_cosine(detected_with_radial_speeds)
    ax_curve.clear()
    ax_curve.set_title("Radial Speed vs Angle")
    ax_curve.set_xlabel("Angle (degrees)")
    ax_curve.set_ylabel("Radial Speed (m/s)")
    ax_curve.set_xlim(-90, 90)
    ax_curve.set_ylim(-10, 10)
    # Update radial speed vs time plot
    ax_time.clear()
    ax_time.set_title("Radial Speed vs Time")
    ax_time.set_xlabel("Time (frames)")
    ax_time.set_ylabel("Radial Speed (m/s)")

    if phi_fit is not None and radial_speed_fit is not None:
        for dot in detected_with_radial_speeds:
            dist = np.sqrt(dot[0]**2 + dot[1]**2)
            phi = np.rad2deg(np.arcsin(dot[1] / dist))
            ax_curve.plot(phi, dot[2], 'kx')  # Detected points
        ax_curve.plot(phi_fit, radial_speed_fit, 'b-', label="Fitted Curve")
        ax_curve.legend()
    
    # Plot radial speed history for each dot
    if frame % (plot_x_limits[1] - square_config['width']) == 0:
        # Clear and reset the plot periodically
        ax_time.clear()
        ax_time.set_title("Radial Speed vs Time (Reset)")
        ax_time.set_xlabel("Time (frames)")
        ax_time.set_ylabel("Radial Speed (m/s)")
        radial_speeds_over_time.clear()  # Reset data
    else:
        ax_time.clear()
        ax_time.set_title("Radial Speed vs Time")
        ax_time.set_xlabel("Time (frames)")
        ax_time.set_ylabel("Radial Speed (m/s)")

        # Plot radial speed history for each dot
        for dot, speeds in radial_speeds_over_time.items():
            if speeds:
                times, rspeeds = zip(*speeds)  # Separate times and speeds
                ax_time.plot(times, rspeeds, label=f"Dot {dot}")

        #ax_time.legend()

    #ax_time.legend()

    return square, wedge

# Create the animation
ani = FuncAnimation(fig, update, frames=range(plot_x_limits[0], plot_x_limits[1] - square_config['width']), interval=100, blit=False)

# Show the animation
plt.show()
