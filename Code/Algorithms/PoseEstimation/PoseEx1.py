import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# --------------------------
# 1. Define a rotation function using Axis-Angle or Exponential Map
# --------------------------
def axis_angle_to_matrix(axis, angle):
    """
    Convert axis-angle to a 3x3 rotation matrix using Rodrigues' formula.
    axis: (3,) array, the rotation axis (must be normalized).
    angle: scalar, the rotation angle in radians.
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    R = np.array([
        [c + x**2 * C,     x*y*C - z*s,   x*z*C + y*s],
        [y*x*C + z*s,     c + y**2 * C,   y*z*C - x*s],
        [z*x*C - y*s,     z*y*C + x*s,   c + z**2 * C]
    ])
    return R

# --------------------------
# 2. Create an example "vehicle" as a cube or set of 3D points
# --------------------------
def make_cube(side=1.0):
    """
    Generate an 8-point cube centered at origin.
    side: length of a cube side
    """
    d = side / 2.0
    # corners of a cube
    corners = np.array([
        [-d, -d, -d],
        [-d, -d,  d],
        [-d,  d, -d],
        [-d,  d,  d],
        [ d, -d, -d],
        [ d, -d,  d],
        [ d,  d, -d],
        [ d,  d,  d],
    ])
    return corners

# --------------------------
# 3. Animation setup
# --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create a cube representing our vehicle
cube_points = make_cube(side=2.0)
scat = ax.scatter([], [], [], s=40, c='b')

# set up axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ego Vehicle Pose Rotation Demo')

# initial axis-angle
rotation_axis = np.array([0.0, 1.0, 0.0])  # rotate about Y
angle_increment = 0.1  # rotate by 0.1 rad (~5.7 degrees) each frame
current_angle = 0.0

def init():
    scat._offsets3d = ([], [], [])
    return (scat,)

def update(frame):
    global current_angle
    # 4. Compute the current rotation matrix
    R = axis_angle_to_matrix(rotation_axis, current_angle)
    
    # 5. Apply the rotation to our cube_points
    rotated_pts = (R @ cube_points.T).T
    
    # 6. Update the scatter plot data
    xdata = rotated_pts[:,0]
    ydata = rotated_pts[:,1]
    zdata = rotated_pts[:,2]
    scat._offsets3d = (xdata, ydata, zdata)
    
    # increment angle for the next frame
    current_angle += angle_increment
    
    return (scat,)

ani = FuncAnimation(fig, update, frames=50, init_func=init, blit=False, interval=300)
plt.show()
