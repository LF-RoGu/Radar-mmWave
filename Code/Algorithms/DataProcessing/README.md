# Data processing obtained from the point cloud.
This subsection of the repository contains multiple scripts to help processing the data obtained from the radar sensor, such data is stored in a .csv file.
And such scrips will help us to process these and to have a simple visualization, to realize if the data is just compromised with noise, or actual valuable data.

## fileRead.py
This script works as the main file, to be run to processes the radar logs, extracts and filters detected points, and prints frame-by-frame summaries of the parsed data, this data corresponding to the TLV 1 and 7. 
All according with the set filters, that in this point will only be physical filters, such as SNR levels, z-values and doppler speed.

Some keypoints from this script:
- Reads radar logs and parses frames, extracting TLVs and detected points.
- Filters data based on SNR, Z-axis, and Doppler thresholds.
- Displays detailed summaries of frames, including filtered points and their attributes. 
- Provides quick validation of radar logs for debugging and inspection.

## radar_utilsPlot.py
This utility script visualizes radar data by creating 3D plots of aggregated points from multiple frames.

- Aggregates detected points over multiple frames to create submaps.
- Generates 3D scatter plots for visualizing spatial distributions of detected points.
- Highlights clusters and point densities in 3D space.

## radar_utilsProcessing.py
This script handles parsing and processing radar logs, extracting useful data such as detected points, side info, and headers.

- Parses frame headers and TLV data, including detected points and Side Info for Detected Points.
- Calculates additional attributes such as range, azimuth, and elevation angles.

## filePlot.py
Combines data processing and visualization, providing end-to-end analysis from parsing logs to generating 3D plots.

- Reads radar logs and processes frames with filters for SNR, Z-axis, and Doppler thresholds.
- Calls plotting utilities to generate 3D visualizations of point clouds.
