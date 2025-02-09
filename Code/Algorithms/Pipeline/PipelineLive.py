import serial
import time
import threading
import queue
from threading import Lock
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import dataDecoder
from frameAggregator import FrameAggregator
import pointFilter
import selfSpeedEstimator
from kalmanFilter import KalmanFilter
import veSpeedFilter
import dbCluster
import occupancyGrid

# -------------------------------
# Configuration Commands
# -------------------------------
CONFIG_COMMANDS = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 60 46 7 18.24 0 0 82.237 1 128 12499 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "frameCfg 0 1 128 0 33.333 1 0",
    "lowPower 0 0",
    "guiMonitor -1 1 0 0 0 0 0",
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 8 4 4 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "lvdsStreamCfg -1 0 0 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 4 31 0",
    "CQSigImgMonitor 0 63 4",
    "analogMonitor 0 0",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0 18.23",
    "cfarFovCfg -1 1 -9.72 9.72",
    "calibData 0 0 0",
    "sensorStart"
]

#Defining the number of how many frames from the past should be used in the frame aggregator
# 0 = only current frame
# n = current frame + n previous frames
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9

#Defining a minimum SNR for the filter stage
FILTER_SNR_MIN = 12

#Defining minimum and maximum z for the filter stage
FILTER_Z_MIN = -0.3
FILTER_Z_MAX = 2

#Defining minimum and maximum phi for the filter stage
FILTER_PHI_MIN = -85
FILTER_PHI_MAX = 85

#Defining the self-speed's Kalman filter process variance and measurement variance
KALMAN_FILTER_PROCESS_VARIANCE = 0.01
KALMAN_FILTER_MEASUREMENT_VARIANCE = 0.1

#Defining dbClustering stages
cluster_processor_stage1 = dbCluster.ClusterProcessor(eps=2.0, min_samples=2)
cluster_processor_stage2 = dbCluster.ClusterProcessor(eps=1.0, min_samples=4)

##Creating the pipeline's objects
#Creating the frame aggregator
frame_aggregator = FrameAggregator(FRAME_AGGREGATOR_NUM_PAST_FRAMES)

#Creating the Kalman filter for the self-speed esimation
self_speed_kf = KalmanFilter(process_variance=KALMAN_FILTER_PROCESS_VARIANCE, measurement_variance=KALMAN_FILTER_MEASUREMENT_VARIANCE)

#
grid_processor = occupancyGrid.OccupancyGridProcessor(grid_spacing=0.5)

#
frame_queue = queue.Queue()

#
write_lock = threading.Lock()
read_lock = threading.Lock()
plot_data_lock = threading.Lock()

#
latest_point_cloud_raw = []
latest_point_cloud_filtered = []
latest_self_speed_raw = []
latest_self_speed_filtered = []
latest_dbscan_clusters = []
latest_occupancy_grid = []

# -------------------------------
# Send Configuration to Sensor
# -------------------------------
def send_configuration(port='COM4', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)

    for command in CONFIG_COMMANDS:
        ser.write((command + '\n').encode())
        print(f"Sent: {command}")
        time.sleep(0.1)
    ser.close()

# -------------------------------
# Sensor Reading Thread
# -------------------------------
def sensor_thread(port='COM6', baudrate=921600):
    ser = serial.Serial(port, baudrate, timeout=1)
    magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    buffer = bytearray()

    while True:
        if ser.in_waiting:
            buffer.extend(ser.read(ser.in_waiting))

            while magic_word in buffer:
                start_idx = buffer.find(magic_word)
                end_idx = buffer.find(magic_word, start_idx + 8)

                if end_idx == -1:
                    break

                frame = buffer[start_idx:end_idx]

                # Thread-safe write operation
                with write_lock:
                    frame_queue.put(frame)

                buffer = buffer[end_idx:]


# -------------------------------
# Data Processing Thread
# -------------------------------
def processing_thread():
    global latest_point_cloud_raw, latest_point_cloud_filtered
    global latest_self_speed_raw, latest_self_speed_filtered
    global latest_dbscan_clusters, latest_occupancy_grid

    while True:
        frame = None

        # Thread-safe read operation
        with read_lock:
            if not frame_queue.empty():
                frame = frame_queue.get()

        if frame:
            try:
                # Convert bytearray to list for decoding
                frame_list = list(frame)

                # Decode the frame correctly
                decoded_frames = dataDecoder.dataToFrames(frame_list)

                for decoded in decoded_frames:
                    # Updating the frame aggregator
                    frame_aggregator.updateBuffer(decoded)

                    # Getting the current point cloud from the frame aggregator
                    point_cloud = frame_aggregator.getPoints()

                    # Filtering by SNR
                    point_cloud_filtered = pointFilter.filterSNRmin(point_cloud, FILTER_SNR_MIN)
                    # Filtering by z
                    point_cloud_filtered = pointFilter.filterCartesianZ(point_cloud_filtered, FILTER_Z_MIN, FILTER_Z_MAX)
                    # Filtering by phi
                    point_cloud_filtered = pointFilter.filterSphericalPhi(point_cloud_filtered, FILTER_PHI_MIN, FILTER_PHI_MAX)

                    # Estimating the self-speed
                    self_speed_raw = selfSpeedEstimator.estimate_self_speed(point_cloud_filtered)
                    # Kalman filtering the self-speed
                    self_speed_filtered = self_speed_kf.update(self_speed_raw)

                    # Filtering point cloud by Ve
                    point_cloud_ve_filtered = pointFilter.filter_by_speed(point_cloud_filtered, self_speed_filtered, 1.0)

                    # First Clustering Stage
                    point_cloud_clustering_stage1 = pointFilter.extract_points(point_cloud_ve_filtered)
                    clusters_stage1, _ = cluster_processor_stage1.cluster_points(point_cloud_clustering_stage1)
                    point_cloud_clustering_stage2 = pointFilter.extract_points(clusters_stage1)
                    clusters_stage2, _ = cluster_processor_stage2.cluster_points(point_cloud_clustering_stage2)
                    point_cloud_clustered = clusters_stage2

                    # OccupancyGrid
                    point_cloud_clustered = pointFilter.extract_points(point_cloud_clustered)


                    # Thread-safe data update for plotting
                    with plot_data_lock:
                        latest_point_cloud_raw
                        latest_point_cloud_filtered = point_cloud_ve_filtered
                        latest_dbscan_clusters = point_cloud_clustered
                        latest_occupancy_grid = grid_processor.calculate_cartesian_grid(point_cloud_clustered[:, :2], x_limits=(-10, 10), y_limits=(0, 15))
                        #latest_occupancy_grid = grid_processor.calculate_polar_grid(point_cloud_clustered[:, :2], range_max , range_bins, angle_bins)
                        latest_self_speed_raw.append(self_speed_raw)
                        latest_self_speed_filtered.append(self_speed_filtered)

            except Exception as e:
                print(f"Error decoding frame: {e}")


# -------------------------------
# Plotting Thread
# -------------------------------
def plotting_thread():
    global latest_point_cloud_raw, latest_point_cloud_filtered
    global latest_self_speed_raw, latest_self_speed_filtered
    global latest_dbscan_clusters, latest_occupancy_grid

    plt.ion()  # Enable interactive mode

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    plot_raw_data =         fig.add_subplot(gs[0, 0], projection='3d')
    plot_filtered_data =    fig.add_subplot(gs[1, 0], projection='3d')
    plot_Ve =               fig.add_subplot(gs[1, 1])
    plot_dbCluster =        fig.add_subplot(gs[0, 2], projection='3d')
    plot_occupancyGrid =    fig.add_subplot(gs[1, 2])
    #plot_polarGrid =        fig.add_subplot(gs[1, 2], polar = True)


    while True:
        with plot_data_lock:
            # Safely copy data for plotting
            point_cloud_raw = latest_point_cloud_raw.copy()
            point_cloud_filtered = latest_point_cloud_filtered.copy()
            self_speed_raw = latest_self_speed_raw.copy()
            self_speed_filtered = latest_self_speed_filtered.copy()
            dbscan_clusters = latest_dbscan_clusters.copy()
            occupancy_grid = latest_occupancy_grid.copy()

        # --- Plot Raw Point Cloud ---
        plot_raw_data.clear()
        if point_cloud_raw:
            try:
                points_x = np.array([point["x"] for point in point_cloud_raw])
                points_y = np.array([point["y"] for point in point_cloud_raw])
                points_z = np.array([point["z"] for point in point_cloud_raw])

                plot_raw_data.set_title('Physical Filters')
                plot_raw_data.set_xlabel('X [m]')
                plot_raw_data.set_ylabel('Y [m]')
                plot_raw_data.set_zlabel('Z [m]')
                plot_raw_data.set_xlim(-10, 10)
                plot_raw_data.set_ylim(0, 15)
                plot_raw_data.set_zlim(-0.30, 10)
                plot_raw_data.scatter(points_x, points_y, points_z)
            except Exception as e:
                print(f"Error plotting raw data: {e}")
        

        # --- Plot Filtered Point Cloud ---
        plot_filtered_data.clear()
        if point_cloud_filtered:
            try:
                points_x = np.array([point["x"] for point in point_cloud_filtered])
                points_y = np.array([point["y"] for point in point_cloud_filtered])
                points_z = np.array([point["z"] for point in point_cloud_filtered])

                plot_filtered_data.set_title('Ve Filters')
                plot_filtered_data.set_xlabel('X [m]')
                plot_filtered_data.set_ylabel('Y [m]')
                plot_filtered_data.set_zlabel('Z [m]')
                plot_filtered_data.set_xlim(-10, 10)
                plot_filtered_data.set_ylim(0, 15)
                plot_filtered_data.set_zlim(-0.30, 10)
                plot_filtered_data.scatter(points_x, points_y, points_z)
            except Exception as e:
                print(f"Error plotting raw data: {e}")

        # --- Plot Self-Speed (Raw vs Filtered) ---
        plot_Ve.clear()
        if self_speed_raw and self_speed_filtered:
            try:
                plot_Ve.plot(self_speed_raw, linestyle='--', label='Raw Speed')
                plot_Ve.plot(self_speed_filtered, label='Filtered Speed')
                plot_Ve.legend()
                plot_Ve.set_title("Self-Speed Estimation")
                plot_Ve.set_ylim(-3, 3)
            except Exception as e:
                print(f"Error plotting self-speed: {e}")
        

        # --- Plot DBSCAN Clusters ---
        priority_colors = {1: 'red', 2: 'orange', 3: 'green'}
        
        plot_dbCluster.clear()
        if dbscan_clusters:
            try:
                for i, cluster in enumerate(dbscan_clusters):
                    cluster_points = pointFilter.extract_points(cluster)
                    if cluster_points.size > 0:
                        color = np.random.rand(3,)
                        plot_dbCluster.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                                               s=10, c=[color], label=f'Cluster {i + 1}')
                plot_dbCluster.legend()
                plot_dbCluster.set_title("DBSCAN Clusters")
            except Exception as e:
                print(f"Error plotting DBSCAN clusters: {e}")
        plot_dbCluster.set_xlim(-10, 10)
        plot_dbCluster.set_ylim(0, 15)
        plot_dbCluster.set_zlim(-0.3, 2)

        # --- Plot Occupancy Grid ---
        plot_occupancyGrid.clear()
        if occupancy_grid is not None:
            try:
                plot_occupancyGrid.imshow(occupancy_grid, cmap='gray', origin='lower')
                plot_occupancyGrid.set_title("Occupancy Grid")
            except Exception as e:
                print(f"Error plotting occupancy grid: {e}")


        plt.pause(0.1)  # Smooth real-time updates

# -------------------------------
# Start Threads
# -------------------------------
if __name__ == "__main__":
    send_configuration(port='COM6')
    
    threading.Thread(target=sensor_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    threading.Thread(target=plotting_thread, daemon=True).start()

    while True:
        time.sleep(1)
