import math

VEHICLE_SPEED = 1.5  # m/s
FPS = 1  # Animation frames per second

def adjust_positions_with_ego_motion(objects, ego_motion_y):
    """Adjust object positions based on ego motion."""
    adjusted_objects = []
    for obj in objects:
        adjusted_objects.append({
            "X [m]": obj["X [m]"],
            "Y [m]": obj["Y [m]"] - ego_motion_y,  # Adjust for ego motion
            "Z [m]": obj["Z [m]"]
        })
    return adjusted_objects

def group_frames(frames_data, group_size=10):
    """Group frames into sets of `group_size`."""
    grouped_data = []
    keys = sorted(frames_data.keys())
    for i in range(0, len(keys), group_size):
        grouped_data.append(keys[i:i + group_size])
    return grouped_data

def extract_objects_from_group(frames_data, group):
    """Extract and adjust objects for a group of frames."""
    ego_motion_y = 0
    all_objects = []
    for frame_num in group:
        frame = frames_data[frame_num]
        objects = frame["TLVs"][0]["Type 1 Data"]  # Extract Type 1 Data
        adjusted_objects = adjust_positions_with_ego_motion(objects, ego_motion_y)
        all_objects.extend(adjusted_objects)
        ego_motion_y += VEHICLE_SPEED / FPS  # Update ego motion
    return all_objects

def animate_3d_plot(frames_data, grouped_frames):
    """Animate the 3D plot with grouped frames."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index):
        ax.clear()
        ax.set_title(f"Frame Group {frame_index + 1}")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_xlim([-10, 10])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-10, 10])

        # Extract objects for the current group
        current_group = grouped_frames[frame_index]
        objects = extract_objects_from_group(frames_data, current_group)
        xs = [obj["X [m]"] for obj in objects]
        ys = [obj["Y [m]"] for obj in objects]
        zs = [obj["Z [m]"] for obj in objects]

        # Plot the objects
        ax.scatter(xs, ys, zs, c="blue", marker="o", label="Detected Objects")

        # Plot the vehicle as a red circle at Z = 0
        ax.scatter(0, -(frame_index * VEHICLE_SPEED), 0, c="red", s=100, marker="o", label="Vehicle (Z=0)")

        # Add legend
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(grouped_frames), interval=1000 // FPS)
    plt.show()

