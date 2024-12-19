from radar_utilsProcessing import process_log_file
from radar_utilsPlot import group_frames, animate_3d_plot
import os

# Main script
if __name__ == "__main__":
    # File path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "StaticMonitoring", "Test_30fps_dist15mts_vehicleLog_5mps_d3x3wall_att1_log.csv")
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    print(f"Processing file: {file_path}")
    # Process radar log file to extract frames and detected objects
    frames_data = process_log_file(file_path)

    # Count total rows in the file (excluding header)
    total_rows = sum(1 for _ in open(file_path)) - 1
    print(f"\nParsed {len(frames_data)} frames successfully out of {total_rows} total rows.")

    # Group frames into sets of 10 for animation
    grouped_frames = group_frames(frames_data, group_size=10)

    # Animate the 3D plot
    animate_3d_plot(frames_data, grouped_frames)
