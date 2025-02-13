#!/usr/bin/env python3
"""! @brief Pipeline for object detection and prevention for static objects using mmWave sensor."""
##
# @mainpage PipelineLive
#
# @section description_main This project aims to develop a real-time object detection and collision avoidance system using the IWR6843AOPEVM mmWave radar sensor. The system processes raw radar data to extract meaningful information about surroundings, estimate self-speed, detect obstacles, and trigger a braking mechanism when necessary.
# 
#
# @section notes_main Notes
# - Add special project notes here that you want to communicate to the user.
#
# @section authors_main Author(s)
# - Luis Fernando Rodriguez Gutierrez
# - Leander Hackmann

# Imports
import serial
import time
import threading
import queue
from threading import Lock
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings

# Local Imports
import radarSensor
import dataDecoder
from frameAggregator import FrameAggregator
import pointFilter
import selfSpeedEstimator
from kalmanFilter import KalmanFilter
import veSpeedFilter
import dbCluster
import occupancyGrid

# List of configuration commands for initializing the mmWave sensor.
## This configuration commands is using the following presets:
## - 30 FPS
## - 60° Azimuth
## - 30° Elevation 
SENSOR_CONFIG_COMMANDS = [
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

# Port Configuration, depending on the information from the device manager on how your device treats the UART sensor. Since we have a USB-PORT to a USB-BRIDGE (Sensor side). Assigned the number of the comm ports depending on your device.
## CONFIG_PORT -> Enhanced Port
## DATA_PORT   -> Standard Port
SENSOR_CONFIG_PORT = "COM9"
SENSOR_DATA_PORT = "COM8"

# Defining the number of how many frames from the past should be used in the frame aggregator
## 0 = only current frame
## n = current frame + n previous frames
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9

## Defining the minimum value of SNR so a point can be consider as a valid point.
FILTER_SNR_MIN = 12

## Defining minimum and maximum z for the filter stage
FILTER_Z_MIN = -0.3
FILTER_Z_MAX = 2

## Defining minimum and maximum phi for the filter stage
FILTER_PHI_MIN = -85
FILTER_PHI_MAX = 85

## Defining the self-speed's Kalman filter process variance and measurement variance
KALMAN_FILTER_PROCESS_VARIANCE = 0.01
KALMAN_FILTER_MEASUREMENT_VARIANCE = 0.1


# Creating the pipeline's objects
## Creating the frame aggregator
frame_aggregator = FrameAggregator(FRAME_AGGREGATOR_NUM_PAST_FRAMES)

## Creating the Kalman filter for the self-speed esimation
self_speed_kf = KalmanFilter(process_variance=KALMAN_FILTER_PROCESS_VARIANCE, measurement_variance=KALMAN_FILTER_MEASUREMENT_VARIANCE)

## Defining dbClustering stages
cluster_processor_stage1 = dbCluster.ClusterProcessor(eps=2.0, min_samples=2)
cluster_processor_stage2 = dbCluster.ClusterProcessor(eps=1.0, min_samples=4)

## Currently not in use: occupancy grid for processing the clusters
#grid_processor = occupancyGrid.OccupancyGridProcessor(grid_spacing=0.5)

# Thread locks
## Setting up a queue together with a lock for passing the data from the sensor thread to the processing thread
frame_queue = queue.Queue()
frame_lock = threading.Lock()

# Setting up buffers together with a lock for storing data for plotting 
latest_point_cloud_raw = []
latest_point_cloud_filtered = []
latest_self_speed_raw = []
latest_occupancy_grid = []
latest_self_speed_filtered = []
latest_dbscan_clusters = []
processed_data_lock = threading.Lock()



# Functions
def sensor_thread():
    """!
    Reads data from the UART from the mmWave sensor, detects frames using a predefined MAGIC WORD,
    and stores valid frames in a thread-safe queue for further processing.

    @return Buffer with current frame obtained from the sensor.

    @note Uses `frame_lock` to prevent race conditions when accessing global data.
    """
    ser = serial.Serial(SENSOR_DATA_PORT, 921600, timeout=1)
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
                with frame_lock:
                    frame_queue.put(frame)

                buffer = buffer[end_idx:]


def processing_thread():
    """!
    This function continuously retrieves sensor data frames using the following process:
    - Obtain information from a shared queue.
    - Decodes them.
    - Applies filtering and clustering techniques.
    - Estimates self-speed. 
    The processed data is then stored in shared global variables for visualization and further analysis.

    @param in latest_point_cloud_raw         Raw point cloud data from the sensor.
    @param in latest_point_cloud_filtered    Filtered point cloud after SNR, Z, and Phi filtering.
    @param in latest_self_speed_raw          Unfiltered self-speed estimation from the point cloud.
    @param in latest_self_speed_filtered     Kalman-filtered self-speed estimation.
    @param in latest_dbscan_clusters         Clusters detected in the point cloud.
    @param in latest_occupancy_grid          Occupancy grid representation of the scene.

    @return latest_dbscan_clusters         Updated with the clustered point cloud after processing.
    @return latest_self_speed_filtered     Updated with the latest Kalman-filtered self-speed estimation.
    
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.
    """
    global latest_point_cloud_raw, latest_point_cloud_filtered
    global latest_self_speed_raw, latest_self_speed_filtered
    global latest_dbscan_clusters, latest_occupancy_grid

    while True:
        frame = None

        #Trying to get the next frame from the queue; continuing if there was no new frame
        try:
            with frame_lock:
                frame = frame_queue.get_nowait()
        except:
            continue
        
        try:
            # Converting the bytearray to a list for decoding
            frame_list = list(frame)

            # Decode the frame correctly
            decoded_frames = dataDecoder.dataToFrames(frame_list)

            for decoded in decoded_frames:
                #Updating the frame aggregator
                frame_aggregator.updateBuffer(decoded)

                #Getting the current point cloud from the frame aggregator
                point_cloud = frame_aggregator.getPoints()

                #Filtering by SNR
                point_cloud_filtered = pointFilter.filterSNRmin(point_cloud, FILTER_SNR_MIN)
                #Filtering by z
                point_cloud_filtered = pointFilter.filterCartesianZ(point_cloud_filtered, FILTER_Z_MIN, FILTER_Z_MAX)
                #Filtering by phi
                point_cloud_filtered = pointFilter.filterSphericalPhi(point_cloud_filtered, FILTER_PHI_MIN, FILTER_PHI_MAX)

                #Estimating the self-speed
                self_speed_raw = selfSpeedEstimator.estimate_self_speed(point_cloud_filtered)
                #Kalman filtering the self-speed
                self_speed_filtered = self_speed_kf.update(self_speed_raw)

                #Filtering point cloud by Ve
                point_cloud_ve = veSpeedFilter.calculateVe(point_cloud_filtered)
                point_cloud_ve_filtered = veSpeedFilter.filterPointsWithVe(point_cloud_ve, self_speed_filtered, 0.5)
                #print(f"\n[!]Filtered points: {len(point_cloud_filtered) - len(point_cloud_ve_filtered)}")

                #Clustering the points (stage 1)
                point_cloud_clustering_stage1 = pointFilter.extract_points(point_cloud_ve_filtered)
                clusters_stage1, _ = cluster_processor_stage1.cluster_points(point_cloud_clustering_stage1)
                
                #Clustering the points (stage 2)
                point_cloud_clustering_stage2 = pointFilter.extract_points(clusters_stage1)
                clusters_stage2, _ = cluster_processor_stage2.cluster_points(point_cloud_clustering_stage2)


                # Thread-safe data update for plotting
                with processed_data_lock:
                    latest_dbscan_clusters = clusters_stage2
                    latest_self_speed_filtered.append(self_speed_filtered)

        except Exception as e:
            print(f"Error decoding frame: {e}")

def data_monitor():
    """!
    This function retrieves self-speed estimations and clustered radar detections from shared
    global variables, displaying real-time updates. It runs as a monitoring thread
    and checks for potential obstacles within a predefined range and azimuth.

    @param in latest_dbscan_clusters      Dictionary containing the most recent detected radar clusters.
    @param in latest_self_speed_filtered  List storing the most recent Kalman-filtered self-speed estimations.
    
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.
    """

    # Continuously prints the latest processed data, including self-speed estimation and cluster warnings.
    offset = -90  # Adjusts the reference for azimuth
    brake_range = 4

    while True:
        with processed_data_lock:
            local_clusters = latest_dbscan_clusters.copy()  # ✅ Use full cluster data instead of centroids
            local_self_speed = latest_self_speed_filtered.copy()  # Copy self-speed data

        # --- Print Self-Speed Estimation ---
        if local_self_speed:
            latest_speed = local_self_speed[-1]  # Get the most recent self-speed estimation
            print(f"\n🚗 Self-Speed Estimation: {latest_speed:.2f} m/s")

        # --- Check for empty clusters ---
        if len(local_clusters) == 0:
            print("No clusters detected.")
            time.sleep(0.5)
            continue

        print("\n📡 Latest DBSCAN Clusters:")
        for cluster_id, cluster in local_clusters.items():

            # Extract cluster information
            centroid = cluster.get('centroid', np.array([0, 0, 0]))  # Default to [0,0,0] if missing
            priority = cluster.get('priority', 'N/A')
            doppler_avg = cluster.get('doppler_avg', 0.0)  # Default to 0.0 if missing

            # Convert to polar coordinates
            r = np.linalg.norm(centroid[:2])  # Compute range (distance from origin)
            azimuth = (np.degrees(np.arctan2(centroid[1], centroid[0])) + offset) % 360  # Compute azimuth

            print(f"[!] Cluster {cluster_id}: Centroid={centroid[:2]}, Range={r:.2f}m, Azimuth={azimuth:.2f}°, "
                  f"Priority={priority}, Doppler Avg={doppler_avg:.2f}")

            # Check if the cluster is within the specified range and angle
            if (r <= brake_range) and (azimuth >= 330 or azimuth <= 30):
                print(f"[!] Warning: Cluster {cluster_id} is at ~{r:.2f}m and {azimuth:.2f}°!")
                # Activate break if object is in range and azimuth
                detection_triggered = True  # Object detected


        time.sleep(0.5)  # Print updates every 0.5 seconds

# -------------------------------
# Start Threads
# -------------------------------
if __name__ == "__main__":
    """! 
    Main program entry point.

    This script initializes and starts multiple background threads for handling sensor data acquisition,
    processing, and real-time monitoring. It ensures proper sensor configuration before launching threads.

    @section Threads Started:
      - `sensor_thread`: Reads raw sensor data from the UART.
      - `processing_thread`: Decodes and processes sensor frames.
      - `data_monitor`: Monitors and prints processed data.

    @note The program runs indefinitely, with daemon threads ensuring automatic cleanup on exit.
    
    @pre The mmWave sensor must be properly connected and configured.
    @post Sensor data is continuously collected, processed, and monitored in concurrent threads.
    """
    warnings.filterwarnings('ignore')


    # Send configuration commands to the radar sensor before starting the threads
    radarSensor.send_configuration(SENSOR_CONFIG_COMMANDS, SENSOR_CONFIG_PORT)
    
    # -------------------------------
    # Start Background Threads
    # -------------------------------
    threading.Thread(target=sensor_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    threading.Thread(target=data_monitor, daemon=True).start()

    # -------------------------------
    # Main Loop
    # -------------------------------
    while True:
        time.sleep(0.1)
