"""""! 
    @file PipelineLive.py
    @brief Pipeline for real-time object detection and collision prevention using mmWave sensor.
    @details This script processes radar sensor data to detect static obstacles, estimate velocity,
    and trigger emergency braking when necessary.

    @defgroup Pipeline_V2 Pipeline Live
    @brief Real-time processing pipeline.
    @{
"""

## @mainpage Pipeline
#
# @section description_main Description
# This project aims to develop a real-time object detection and collision avoidance system
# using the IWR6843AOPEVM mmWave radar sensor. The system processes raw radar data
# to extract meaningful information about surroundings, estimate self-speed, detect obstacles,
# and trigger a braking mechanism when necessary.
#
# @section notes_main Notes
# - For this project the bare minimum to be able to replicated is an IWR6843AOPEVM mmWave sensor from TI.
#
# @section hardware_sec Hardware Used
# - Radar Sensor: IWR6843AOPEVM
# - Processing Unit: Raspberry Pi
# - Communication: UART interface
#
# @section notes_sec Special Considerations
# - The radar operates at 30 FPS or 12 FPS, configurable via UART commands.
# - The field of view is adjustable between 30° and 60° azimuth.
#
# @section authors_main Author(s)
# - Luis Fernando Rodriguez Gutierrez
# - Leander Hackmann

# Imports
import time
import threading
import warnings
import logging
import numpy as np
from gpiozero import DigitalOutputDevice

# Local Imports
from dataDecoderTI import DataDecoderTI
from frameAggregator import FrameAggregator
import pointFilter
import selfSpeedEstimator
from kalmanFilter import KalmanFilter
import veSpeedFilter
import dbCluster
from linearController import LinearSpeedController

## @defgroup Global Constants
## @{
## @brief Set platform (embedded device w Linux / Windows PC)
PLATFORM_EMBEDDED = False
## @brief Set logging level
LOGGING_LEVEL = logging.DEBUG
## @brief Setting the distance (m) for the emergency brake to activate
EMERGENCY_BRAKE_RANGE = 3
## @brief Setting the angle (+- from 0°) for the emergency brake to activate
EMERGENCY_BRAKE_PHI = 10
## @brief Setting the minimal self-speed (m/s) for the emergency brake to activate
EMERGENCY_BRAKE_MIN_SELFSPEED = -0.75
## @brief Setting the timeout (s) that needs to pass after an activation for the brake to be deactivated
EMERGENCY_BRAKE_TIMEOUT = 5

## @brief Seting the sensor's config file filename
SENSOR_CONFIG_FILE = "profile_azim60_elev30_optimized.cfg"
## @brief UART port used for sensor configuration on the embedded linux device.
## @note CONFIG_PORT -> Enhanced Port
SENSOR_CONFIG_PORT_EMBEDDED = "/dev/ttyUSB0"
## @brief UART port used for receiving sensor data on the embedded linux device.
## @note DATA_PORT   -> Standard Port
SENSOR_DATA_PORT_EMBEDDED = "/dev/ttyUSB1"
## @brief UART port used for sensor configuration on the Windows PC.
## @note CONFIG_PORT -> Enhanced Port
SENSOR_CONFIG_PORT_PC = "COM6"
## @brief UART port used for receiving sensor data Windows PC.
## @note DATA_PORT   -> Standard Port
SENSOR_DATA_PORT_PC = "COM7"

## @brief Seting the GPIO-Pin of the braking signal on the embedded linux device
PLATFORM_EMBEDDED_BRAKE_GPIO_PIN = 26

## @brief Number of past frames to store in the frame aggregator.
## 0 = only current frame, n = current frame + n previous frames
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9

## @brief Minimum SNR value required for a point to be considered valid.
FILTER_SNR_MIN = 12

## @brief Minimum Z-coordinate threshold for filtering points (meters).
FILTER_Z_MIN = 0.0
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
## @brief Creates the sensor object
radarSensor = DataDecoderTI()
## @brief Creates the frame aggregator to store past frames.
frame_aggregator = FrameAggregator(FRAME_AGGREGATOR_NUM_PAST_FRAMES)
## @brief Initializes the Kalman filter for self-speed estimation.
self_speed_kf = KalmanFilter(process_variance=KALMAN_FILTER_PROCESS_VARIANCE, measurement_variance=KALMAN_FILTER_MEASUREMENT_VARIANCE)
## @brief Defines the first-stage DBSCAN clustering processor.
cluster_processor_stage1 = dbCluster.ClusterProcessor(eps=2.0, min_samples=2)
## @brief Defines the second-stage DBSCAN clustering processor.
cluster_processor_stage2 = dbCluster.ClusterProcessor(eps=1.0, min_samples=4)
## @brief Defines a DigitalOutputDevice for the brake signal if on an embedded linux platform
if PLATFORM_EMBEDDED:
    brakeSignal = DigitalOutputDevice(PLATFORM_EMBEDDED_BRAKE_GPIO_PIN)
    speedController = LinearSpeedController()
## @} # End of Pipeline Constructors


## @defgroup Thread locks
## @{

## @brief List for passing sensor data from the sensor thread to the processing thread.
frame_list = []
## @brief Lock to ensure safe access to `frame_list` between threads.
frame_lock = threading.Lock()
## @brief Lock to synchronize access to processed data before plotting.
processed_data_lock = threading.Lock()
## @} # End of Thread locks

## @defgroup Global variables
## @{
## @brief Stores the latest Kalman-filtered self-speed estimation.
latest_self_speed_filtered = 0
## @brief Stores the most recent detected DBSCAN clusters.
latest_dbscan_clusters = []
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
    and stores valid frames in a thread-safe list for further processing.

    @note This function runs indefinitely in a separate thread.
    @note Uses `frame_lock` to prevent race conditions when accessing global data.

    @param in radarSensor       The radar sensor object.
    @param inout frame_lock     Lock for thread-safe access to 'frame_list'
    @param out frame_list       List where valid frames are stored.

    @ingroup threadFunctions
    """
    global radarSensor
    global frame_list
    global frame_lock

    while True:
        # Polling the IWR6843 sensor and getting the number of decoded frames
        numFrames = radarSensor.pollIWR6843()

        # Continuing if there are no new frames
        if numFrames == 0:
            continue

        # Getting the new frames and deleting them from the sensor's internal buffer
        newFrames = radarSensor.get_and_delete_decoded_frames(numFrames)

        # Appending the new frames to the global list of decoded frames in a thread-safe way
        with frame_lock:
            frame_list += newFrames


def processing_thread():
    """!
    This function continuously retrieves sensor data frames using the following process:
    - Decodes incoming frames into usable point clouds.
    - Applies filtering to remove noise and irrelevant data.
    - Estimates self-speed using Doppler velocity data.
    - Uses DBSCAN clustering to group detected objects.
    - Stores processed results in global variables for monitoring and processing.
    The processed data is then stored in shared global variables for monitoring and further analysis.

    @note This function runs indefinitely in a separate thread.
    @note Uses `frame_lock` and `processed_data_lock` to prevent race conditions when accessing global data.

    @param inout frame_lock     Lock for thread-safe access to `frame_list`.
    @param inout frame_list     List where valid frames are stored.

    @param inout processed_data_lock        Lock for thread-safe access to `latest_dbscan_clusters` and `latest_self_speed_filtered`
    @param out latest_dbscan_clusters       Updated with the latest clusters after processing.
    @param out latest_self_speed_filtered   Updated with the latest Kalman-filtered self-speed estimation.
    
    @ingroup threadFunctions
    """

    global frame_lock
    global frame_list

    global processed_data_lock
    global latest_self_speed_filtered
    global latest_dbscan_clusters

    while True:
        frames_to_process = []

        # Trying to get the all new frame from the list; continuing if there was no new frame
        with frame_lock:
            if len(frame_list) == 0:
                continue

            frames_to_process = frame_list
            frame_list = []
        
        try:
            # Processing frame-by-frame
            for frame in frames_to_process:
                # Updating the frame aggregator
                frame_aggregator.updateBuffer(frame)

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

def braking_system():
    """!
    This function retrieves self-speed estimations and clustered radar detections from shared
    global variables, processes the clusters and initiates the braking event if needed.
    After an initiated braking event, the brake is released.

    @note This function runs indefinitely in a separate thread.
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.

    @param inout processed_data_lock        Lock for thread-safe access to `latest_dbscan_clusters` and `latest_self_speed_filtered`
    @param in latest_dbscan_clusters        Dictionary containing the most recent detected radar clusters.
    @param in latest_self_speed_filtered    List storing the most recent Kalman-filtered self-speed estimation.

    @ingroup threadFunctions
    """

    global processed_data_lock
    global latest_dbscan_clusters
    global latest_self_speed_filtered
    
    brake_activated = False
    brake_activated_timestamp = None
    while True:
        # Copying the most recent data thread-safe
        with processed_data_lock:
            local_clusters = latest_dbscan_clusters.copy()
            local_self_speed = latest_self_speed_filtered

        # Continuing if there are no clusters
        if not local_clusters:
            continue

        # Flag for storing if the brake needs to be activated    
        brake_activation_trigger = False
        # Calculate current brake distance depending on the current self speed estimation.
        speedController.stopping_distance(local_self_speed)

        # Iterating over all clusters to check if the brake needs to be activated
        for cluster_id, cluster in local_clusters.items():
            centroid = cluster.get('centroid', np.array([0, 0, 0]))
            
            # Calculating the distance to the cluster's centroid
            r = np.linalg.norm(centroid[:2])
            # Checking distance of the cluster's centroid and continuing if too far away
            if speedController.control(local_self_speed, r) is not 0:
                continue
            
            # Calculating the angle to the cluster's centroid
            phi = np.rad2deg(np.arctan(centroid[0]/centroid[1]))
            # Checking if the cluster's centroid is inside the safety zone and if we are moving
            if abs(phi) <= EMERGENCY_BRAKE_PHI and local_self_speed < EMERGENCY_BRAKE_MIN_SELFSPEED:
                # Setting the brake activation trigger, logging and breaking out of the loop
                brake_activation_trigger = True
                break
        
        # Activating the brake and storing the timestamp for timeout if the brake is not already activated
        if brake_activation_trigger and not brake_activated:
            if PLATFORM_EMBEDDED:
                brakeSignal.on()
            
            brake_activated = True
            brake_activated_timestamp = time.time()
            logging.warning("Brake is now activated")
            continue
        
        # Deactivating the brake if it was activated and the timeout is already over
        if brake_activated and (time.time() - brake_activated_timestamp) >= EMERGENCY_BRAKE_TIMEOUT:
            if PLATFORM_EMBEDDED:
                brakeSignal.off()

            brake_activated = False
            brake_activated_timestamp = None
            logging.warning("Brake is now deactivated")
            continue


def data_monitor():
    """!
    This function retrieves self-speed estimations and clustered radar detections from shared
    global variables, displaying real-time updates. It runs as a monitoring thread
    and checks for potential obstacles within a predefined range and azimuth.

    @note This function runs indefinitely in a separate thread.
    @note Uses `processed_data_lock` to prevent race conditions when accessing global data.

    @param inout processed_data_lock        Lock for thread-safe access to `latest_dbscan_clusters` and `latest_self_speed_filtered`
    @param in latest_dbscan_clusters        Dictionary containing the most recent detected radar clusters.
    @param in latest_self_speed_filtered    List storing the most recent Kalman-filtered self-speed estimation.

    @ingroup threadFunctions
    """

    global processed_data_lock
    global latest_dbscan_clusters
    global latest_self_speed_filtered

    # Continuously prints the latest processed data, including self-speed estimation and cluster warnings.
    offset = -90  # Adjusts the reference for azimuth

    while True:
        # Copying the most recent data thread-safe
        with processed_data_lock:
            local_clusters = latest_dbscan_clusters.copy()
            local_self_speed = latest_self_speed_filtered

        # Printing the latest self-speed estimation
        logging.debug(f"Self-Speed Estimation: {local_self_speed:.2f} m/s")

        # Sleeping if there are no new clusters
        if not local_clusters or len(local_clusters) == 0:
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
    if PLATFORM_EMBEDDED:
        radarSensor.initIWR6843(SENSOR_CONFIG_PORT_EMBEDDED, SENSOR_DATA_PORT_EMBEDDED, SENSOR_CONFIG_FILE)
    else:
        radarSensor.initIWR6843(SENSOR_CONFIG_PORT_PC, SENSOR_DATA_PORT_PC, SENSOR_CONFIG_FILE)
    
    # Starting all background threads
    threading.Thread(target=sensor_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    threading.Thread(target=data_monitor, daemon=True).start()
    #threading.Thread(target=braking_system, daemon=True).start()

    # Doing something
    while True:
        time.sleep(0.1)

## @}

## @}  # End of Pipeline_V2 group