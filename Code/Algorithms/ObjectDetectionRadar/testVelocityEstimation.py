import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate points for a single frame
def generate_frame(num_points, frame_idx):
    # Generate random angles between 0 and 180 degrees (in radians)
    angles = np.linspace(0, np.pi, num_points) + np.random.uniform(-0.1, 0.1, num_points)
    # Fixed radius for simplicity
    radius = 5
    # Generate X and Y points
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    # Assign random Doppler speeds
    doppler = np.random.uniform(0.5, 1.5, num_points)
    return x, y, doppler

# Generate data for multiple frames
def generate_frames(num_frames, points_per_frame):
    frames = []
    for frame_idx in range(num_frames):
        x, y, doppler = generate_frame(points_per_frame, frame_idx)
        frames.append((x, y, doppler))
    return frames

# Fit points to a polynomial curve
def fit_curve(x, y, degree=2):
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    return polynomial

# Animation initialization
def init():
    sc.set_offsets(np.c_[[], []])  # Empty scatter plot initially
    line.set_data([], [])  # Empty line plot initially
    return sc, line

# Animation update function
def update(frame_idx):
    x, y, doppler = frames[frame_idx]
    
    # Update scatter plot points
    sc.set_offsets(np.c_[x, y])
    sc.set_array(doppler)  # Update Doppler color values
    
    # Fit and update the curve
    polynomial = fit_curve(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    line.set_data(x_fit, y_fit)
    
    # Update the title
    ax.set_title(f"Frame {frame_idx + 1}")
    return sc, line

# Parameters
num_frames = 10
points_per_frame = 20

# Generate frames
frames = generate_frames(num_frames, points_per_frame)

# Setup the plot
fig, ax = plt.subplots(figsize=(8, 6))
x, y, doppler = frames[0]  # Initial frame data
sc = ax.scatter(x, y, c=doppler, cmap='viridis', s=50, label="Points")
line, = ax.plot([], [], 'r-', label="Fitted Curve")

# Add colorbar (only once)
cbar = plt.colorbar(sc, ax=ax, label="Doppler (m/s)")

# Set plot limits and labels
ax.set_xlim(-6, 6)
ax.set_ylim(-1, 6)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=1000)

# Display the animation
plt.show()
