# Pose Estimation

This subcection of the repository contains multiple scripts focused on motion analysis, vehicle pose estimation, and velocity estimation using 3D point cloud data and Doppler measurements.

## Main Scripts

### PoseEx1.py
Simulates and visualizes the rotation of an object (such as a vehicle) using an axis-angle representation. Key features:
- Implements Rodrigues' formula to compute rotation matrices.
- Creates a 3D cube representation of an object.
- Animates rotation over time around a specified axis.

### VelocityEstimatino.py
Estimates linear velocity from two consecutive frames of 3D point cloud data using a least-squares approach. Key features:
- Computes velocity by comparing point correspondences between frames.
- Uses synthetic data to validate the estimation method.
- Provides visualization of point cloud displacement over time.

### PaperPlayground.py
Simulates a vehicle moving toward a static target and estimates velocity using Doppler speeds and filtering techniques. Key features:
- Simulates vehicle motion with predefined parameters.
- Applies low-pass filtering to reduce noise in Doppler speed measurements.
- Uses frame averaging for velocity estimation.
- Visualizes vehicle motion, Doppler speeds, and estimated velocity.

---
This repository provides scripts for motion modeling, velocity estimation, and filtering techniques to analyze vehicle dynamics based on simulated and real-world data.
