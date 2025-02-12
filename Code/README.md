# Algorithms

This folder contains Python scripts that facilitate radar data processing, visualization, and analysis for dynamic and static monitoring scenarios. The scripts implement radar point cloud processing, filtering, and submap visualization to assist in understanding the environment and estimating object movements.

## Data Processing
This folder contains Python scripts for processing and visualizing radar sensor data. fileRead.py extracts and filters radar log data, while filePlot.py visualizes it using 3D scatter plots. radar_utilsProcessing.py handles data decoding, computing metrics like range and azimuth, and applying filters. radar_utilsPlot.py provides visualization tools for aggregating and analyzing multiple frames. Together, these scripts enable radar-based perception and motion estimation.

## Object Detection Radar
This folder contains Python scripts for processing and visualizing radar sensor data, specially point clouds obtain from the mmWave sensor. This scripts help visualize how the point cloud is processed and how it works. By visualizing the clusters update them self frame by frame using the slider. Also how the occupancy grid works depending on the amount of 'hits' or points are in a certain grid, in which depending on the amount of presence an object has the occupancy grid of this cluster in this grid will have a higher value.
## Self Speed Estimation

## Pipeline

## Pipeline Threaded

-------------------------------------------------------------------------------------------------------------

## Theoretical Background
These scripts are based on radar signal processing concepts from **Chapter 3: Point-to-Point Radar ICP and Ego-Motion Estimation**. The core topics covered include:

### Radar Data Representation
- Radar sensors detect objects using time-of-flight and Doppler effects.
- The data consists of X, Y, Z coordinates, velocity, and intensity values.
- Signal processing involves filtering based on signal strength (SNR) and Doppler thresholds.

### Radar Point Cloud Processing
- Extracting individual detections (targets) per frame.
- Aggregating multiple frames into a submap for better environmental understanding.
- Applying geometric transformations to track moving objects.

### Object Tracking and Ego-Velocity Estimation
- Using Doppler velocity data to estimate object motion relative to the radar.
- Filtering noisy detections using predefined thresholds.
- Applying least squares optimization and Jacobian-based corrections for improved motion estimation.

### 3D Radar Data Visualization
- Rendering radar detections in 3D to analyze spatial relationships.
- Combining multiple frames into a single view for better object tracking.
- Identifying obstacles and static/moving objects using point cloud clustering.

This theoretical foundation underpins the functionality of the scripts, providing a framework for radar-based perception in automotive and robotic applications.

