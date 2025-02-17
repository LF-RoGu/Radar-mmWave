"""""! 
    @file PipelineLive.py
    @brief Pipeline for real-time object detection and collision prevention using mmWave sensor.
    @details This script processes radar sensor data to detect static obstacles, estimate velocity,
    and trigger emergency braking when necessary.

    @defgroup Pipeline_V2 Pipeline Live
    @brief Real-time processing pipeline.
    @{
"""

## @mainpage PipelineLive
#
# @section description_main Description
# This project aims to develop a real-time object detection and collision avoidance system
# using the IWR6843AOPEVM mmWave radar sensor. The system processes raw radar data
# to extract meaningful information about surroundings, estimate self-speed, detect obstacles,
# and trigger a braking mechanism when necessary.
#
# @section notes_main Notes
# - Add any special project notes here.
#
# @section authors_main Author(s)
# - Luis Fernando Rodriguez Gutierrez
# - Leander Hackmann

# Imports
import serial
import time
import threading
import queue
import warnings
import logging
import numpy as np

# Local Imports
import radarSensor
import dataDecoder
from frameAggregator import FrameAggregator
import pointFilter
import selfSpeedEstimator
from kalmanFilter import KalmanFilter
import veSpeedFilter
import dbCluster

## @defgroup Global Constants
## @{

## @brief Set logging level
LOGGING_LEVEL = logging.DEBUG
## @brief Setting the distance (m) for the emergency brake to activate
EMERGENCY_BRAKE_RANGE = 4

## @brief List of configuration commands for initializing the mmWave sensor.
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

## @brief UART port used for sensor configuration.
## @note CONFIG_PORT -> Enhanced Port
SENSOR_CONFIG_PORT = "COM9"
## @brief UART port used for receiving sensor data.
## @note DATA_PORT   -> Standard Port
SENSOR_DATA_PORT = "COM8"

## @brief Number of past frames to store in the frame aggregator.
## 0 = only current frame, n = current frame + n previous frames
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9

## @brief Minimum SNR value required for a point to be considered valid.
FILTER_SNR_MIN = 12

## @brief Minimum Z-coordinate threshold for filtering points (meters).
FILTER_Z_MIN = -0.3
## @brief Maximum Z-coordinate threshold for filtering points (meters).
FILTER_Z_MAX = 2.0

## @brief Minimum Phi angle threshold for filtering points (degrees).
FILTER_PHI_MIN = -85
## @brief Maximum Phi angle threshold for filtering points (degrees).
FILTER_PHI_MAX = 85

## @brief Process variance for the Kalman filter (affects smoothness of estimates).
KALMAN_FILTER_PROCESS_VARIANCE = 0.01
## @brief Measurement variance for the Kalman filter (accounts for sensor noise).
KALMAN_FILTER_MEASUREMENT_VARIANCE = 0.1
## @}  # End of Constants


## @defgroup Pipeline Constructors
## @brief Initializes core objects for the pipeline.
## @{

## @brief Creates the frame aggregator to store past frames.
frame_aggregator = FrameAggregator(FRAME_AGGREGATOR_NUM_PAST_FRAMES)
## @brief Initializes the Kalman filter for self-speed estimation.
self_speed_kf = KalmanFilter(process_variance=KALMAN_FILTER_PROCESS_VARIANCE, measurement_variance=KALMAN_FILTER_MEASUREMENT_VARIANCE)
## @brief Defines the first-stage DBSCAN clustering processor.
cluster_processor_stage1 = dbCluster.ClusterProcessor(eps=2.0, min_samples=2)
## @brief Defines the second-stage DBSCAN clustering processor.
cluster_processor_stage2 = dbCluster.ClusterProcessor(eps=1.0, min_samples=4)
## @} # End of Pipeline Constructors


## @defgroup Thread locks
## @{

## @brief Queue for passing sensor data from the sensor thread to the processing thread.
frame_queue = queue.Queue()
## @brief Lock to ensure safe access to `frame_queue` between threads.
frame_lock = threading.Lock()
## @brief Lock to synchronize access to processed data before plotting.
processed_data_lock = threading.Lock()
## @} # End of Thread locks

## @defgroup Global variables
## @{
## @brief Stores the latest raw point cloud data from the sensor.
latest_point_cloud_raw = []
## @brief Stores the latest point cloud data after filtering.
latest_point_cloud_filtered = []
## @brief Stores the latest unfiltered self-speed estimations.
latest_self_speed_raw = []
## @brief Stores the latest Kalman-filtered self-speed estimations.
latest_self_speed_filtered = 0
## @brief Stores the most recent detected DBSCAN clusters.
latest_dbscan_clusters = 0
## @brief Stores the latest occupancy grid representation of the environment.
latest_occupancy_grid = []
## @}

## @defgroup threadFunctions Thread Functions
## @brief Functions that run in separate threads to handle sensor data acquisition, processing, and monitoring.
##
## These functions are executed in parallel using Python's `threading` module.
## Each function runs indefinitely in its own thread, ensuring real-time data handling.
## 
## @note These functions rely on global variables and require thread-safe mechanisms such as locks.
## @{

def sensor_thread():
    """!
    Reads data from the UART from the mmWave sensor, detects frames using a predefined MAGIC WORD,
    and stores valid frames in a thread-safe queue for further processing.

    @note This function runs indefinitely in a separate thread.
    @note Uses `frame_lock` to prevent race conditions when accessing global data.

    @param in SENSOR_DATA_PORT  The UART port from which sensor data is read.
    @param out frame_queue  Thread-safe queue where valid frames are stored.
    @param inout buffer  Internal buffer that accumulates incoming bytes before processing.

    @ingroup threadFunctions
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
    - Decodes incoming frames into usable point clouds.
    - Applies filtering to remove noise and irrelevant data.
    - Estimates self-speed using Doppler velocity data.
    - Uses DBSCAN clustering to group detected objects.
    - Stores processed results in global variables for visualization.
    The processed data is then stored in shared global variables for visualization and further analysis.

    @note This function runs indefinitely in a separate thread.
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.

    @param in latest_point_cloud_raw         Raw point cloud data from the sensor.
    @param in latest_point_cloud_filtered    Filtered point cloud after SNR, Z, and Phi filtering.
    @param in latest_self_speed_raw          Unfiltered self-speed estimation from the point cloud.
    @param in latest_self_speed_filtered     Kalman-filtered self-speed estimation.
    @param in latest_dbscan_clusters         Clusters detected in the point cloud.
    @param in latest_occupancy_grid          Occupancy grid representation of the scene.

    @param out latest_dbscan_clusters         Updated with the clustered point cloud after processing.
    @param out latest_self_speed_filtered     Updated with the latest Kalman-filtered self-speed estimation.
    
    @ingroup threadFunctions
    """
    global latest_point_cloud_raw, latest_point_cloud_filtered
    global latest_self_speed_raw, latest_self_speed_filtered
    global latest_dbscan_clusters, latest_occupancy_grid

    while True:
        frame = None

        # Trying to get the next frame from the queue; continuing if there was no new frame
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
                point_cloud_ve = veSpeedFilter.calculateVe(point_cloud_filtered)
                point_cloud_ve_filtered = veSpeedFilter.filterPointsWithVe(point_cloud_ve, self_speed_filtered, 0.5)
                # print(f"\n[!]Filtered points: {len(point_cloud_filtered) - len(point_cloud_ve_filtered)}")

                # Clustering the points (stage 1)
                point_cloud_clustering_stage1 = pointFilter.extract_points(point_cloud_ve_filtered)
                clusters_stage1, _ = cluster_processor_stage1.cluster_points(point_cloud_clustering_stage1)
                
                # Clustering the points (stage 2)
                point_cloud_clustering_stage2 = pointFilter.extract_points(clusters_stage1)
                clusters_stage2, _ = cluster_processor_stage2.cluster_points(point_cloud_clustering_stage2)


                # Thread-safe data update for plotting
                with processed_data_lock:
                    latest_dbscan_clusters = clusters_stage2
                    latest_self_speed_filtered = self_speed_filtered

        except Exception as e:
            logging.error(f"Error decoding frame: {e}")

def data_monitor():
    """!
    This function retrieves self-speed estimations and clustered radar detections from shared
    global variables, displaying real-time updates. It runs as a monitoring thread
    and checks for potential obstacles within a predefined range and azimuth.

    @note This function runs indefinitely in a separate thread.
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.

    @param in latest_dbscan_clusters      Dictionary containing the most recent detected radar clusters.
    @param in latest_self_speed_filtered  List storing the most recent Kalman-filtered self-speed estimations.

    @ingroup threadFunctions
    """

    # Continuously prints the latest processed data, including self-speed estimation and cluster warnings.
    offset = -90  # Adjusts the reference for azimuth

    while True:
        # Copying the most recent data thread-safe
        with processed_data_lock:
            local_clusters = latest_dbscan_clusters.copy()
            local_self_speed = latest_self_speed_filtered.copy()

        # Printing the latest self-speed estimation
        if local_self_speed:
            latest_speed = local_self_speed[-1]  # Get the most recent self-speed estimation
            logging.debug(f"Self-Speed Estimation: {latest_speed:.2f} m/s")

        # Sleeping if there are no new clusters
        if len(local_clusters) == 0:
            logging.debug("No clusters detected.")
            time.sleep(0.5)
            continue

        logging.debug("Latest DBSCAN Clusters:")
        for cluster_id, cluster in local_clusters.items():

            # Extract cluster information
            centroid = cluster.get('centroid', np.array([0, 0, 0]))  # Default to [0,0,0] if missing
            priority = cluster.get('priority', 'N/A')
            doppler_avg = cluster.get('doppler_avg', 0.0)  # Default to 0.0 if missing

            # Convert to polar coordinates
            r = np.linalg.norm(centroid[:2])  # Compute range (distance from origin)
            azimuth = (np.degrees(np.arctan2(centroid[1], centroid[0])) + offset) % 360  # Compute azimuth

            logging.debug(f"Cluster {cluster_id}: Centroid={centroid[:2]}, Range={r:.2f}m, Azimuth={azimuth:.2f}°, "
                  f"Priority={priority}, Doppler Avg={doppler_avg:.2f}")

            # Check if the cluster is within the specified range and angle
            if (r <= EMERGENCY_BRAKE_RANGE) and (azimuth >= 330 or azimuth <= 30):
                logging.warning(f"Cluster {cluster_id} is at ~{r:.2f}m and {azimuth:.2f}°!")
                # Activate break if object is in range and azimuth
                detection_triggered = True  # Object detected


        time.sleep(0.5)  # Printing updates every 0.5 seconds


# Main program entry point
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
    # Disabling warnings (of numpy)
    warnings.filterwarnings('ignore')

    # Setting the logging level
    logging.basicConfig(level=LOGGING_LEVEL)


    # Sending the configuration commands to the radar sensor before starting the threads
    radarSensor.send_configuration(SENSOR_CONFIG_COMMANDS, SENSOR_CONFIG_PORT)
    
    # Starting all background threads
    threading.Thread(target=sensor_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    threading.Thread(target=data_monitor, daemon=True).start()

    # Doing something
    while True:
        time.sleep(0.1)

## @}

## @}  # End of Pipeline_V2 group