# Detecting objects in front of the car.
This subsection of the repository focuses on the goal of detecting objects using a radar mmWave sensor. 
The code processes radar data to first apply processing algorithms and then to clustering the data.

## ObjectDetectionSubMaps3D.py
This script processes te point cloud to detect valid points and detect cluster that will be then processed to be considered object to lastly be represented in a 3D space. 
It visualizes the clustered points, calculates average Doppler speeds, and monitors a safety box around the vehicle to warn about potential collisions. 

### Coordinate Extraction with Doppler Information:
- Filters radar data based on thresholds for Y, Z coordinates, and Doppler velocity.
- Stores valid points with X, Y, Z coordinates and Doppler speed in a dictionary indexed by frame numbers.

### Submap Aggregation:
- Enhances spatial consistency by combining data over a configurable number of frames. This enhancement comes from agregating multiple frames into one sub-set of data, that will be used to process at a time.

### Clustering Using DBSCAN:
- Clusters points based on spatial proximity using DBSCAN.
- Ignores small clusters with fewer than 'n' points.
- Assigns priorities to clusters based on their size and marks them with different colors.

### 3D Visualization:
- Visualizes clusters with different colors based on priority levels.
- Displays bounding boxes around clusters and marks centroids.
- Shows average Doppler speed and priority labels for each cluster.

### Interactive Visualization with Sliders:
- Allows navigation through frames using a slider widget.
- Updates visualization dynamically for better analysis.

## ObjectDetectionRadar3D.py
This script parses and visualizes 3D radar data, extracting detected points and their Doppler speeds. \
It applies filters, clusters data with DBSCAN, and provides interactive visualizations.

### Frame Parsing and Filtering:
- Parses raw radar data logs and extracts frame headers and TLV payloads.
- Filters points based on thresholds for Y, Z coordinates, and Doppler velocity.

### Clustering and Visualization:
- Uses DBSCAN for spatial grouping and highlights clusters with color-coded priorities.
- Visualizes data in 3D with wedges indicating sensor detection zones and grids for density mapping.

### Interactive Analysis:
- Allows navigation through frames using a slider widget.
- Updates visualization dynamically for better analysis.

## ObjectDetectionSubMaps.py
This script focuses on 2D clustering and visualization, detecting objects using DBSCAN and aggregating frames into submaps to improve detection accuracy.

### Data Processing and Filtering:
- Extracts and filters radar points based on thresholds for Y, Z coordinates, and Doppler velocity.
- Aggregates multiple frames into submaps for denser point clouds.

### Clustering and Visualization:
- Clusters points based on spatial proximity using DBSCAN.
- Ignores small clusters with fewer than 'n' points.
- Assigns priorities to clusters based on their size and marks them with different colors.

### Interactive Analysis:
- Allows navigation through frames using a slider widget.
- Updates visualization dynamically for better analysis.

## ObjectDetectionRadar.py
This script generates 2D occupancy grids and clusters objects based on Doppler speeds and spatial positions, visualizing movement patterns interactively.

### Coordinate Extraction and Filtering:
- Filters radar data based on thresholds for Y, Z coordinates, and Doppler velocity.
- Maps filtered data to a grid for occupancy visualization.

### Clustering and Grid Mapping:
- Clusters points based on spatial proximity using DBSCAN.
- Ignores small clusters with fewer than 'n' points.
- Generates occupancy grids to highlight density patterns and tracks history for cumulative visualization.

### Interactive Visualization:
- Allows navigation through frames using a slider widget.
- Updates visualization dynamically for better analysis.
