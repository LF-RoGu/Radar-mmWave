# Data processing obtained from the point cloud.
This repository contains scripts for processing radar data, visualizing point clouds, and clustering detected objects. 
The tools assist in parsing logs, filtering data, and plotting aggregated results.

## fileRead.py
This script processes radar logs, extracts and filters detected points, and prints frame-by-frame summaries of the parsed data.

- Reads radar logs and parses frames, extracting TLVs and detected points.
- Filters data based on SNR, Z-axis, and Doppler thresholds.
- Displays detailed summaries of frames, including filtered points and their attributes.
- Provides quick validation of radar logs for debugging and inspection.

## radar_utilsPlot.py
This utility script visualizes radar data by creating 3D plots of aggregated points from multiple frames.

- Aggregates detected points over multiple frames to create submaps.
- Generates 3D scatter plots for visualizing spatial distributions of detected points.
- Highlights clusters and point densities in 3D space.
- Provides tools for visually inspecting radar data patterns.

## radar_utilsProcessing.py
This script handles parsing and processing radar logs, extracting useful data such as detected points, side info, and headers.

- Parses frame headers and TLV data, including detected points and metadata.
- Calculates additional attributes such as range, azimuth, and elevation angles.
- Filters points based on thresholds for SNR, Z-axis values, and Doppler speeds.
- Supports processing raw binary data and converting it into structured dictionaries.

## filePlot.py
Combines data processing and visualization, providing end-to-end analysis from parsing logs to generating 3D plots.

- Reads radar logs and processes frames with filters for SNR, Z-axis, and Doppler thresholds.
- Calls plotting utilities to generate 3D visualizations of point clouds.
- Allows quick inspection of radar data through visual feedback.
- Outputs parsed frame summaries for debugging and verification.
