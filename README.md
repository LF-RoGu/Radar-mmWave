# Radar-SiCo2

This repository contains materials and projects for the Radar and SiCo2 classes.

## Repository Structure

- [Code](Code/): Contains algorithms created in python for testing.
  - [Algorithms](Code/Algorithms):
    - [ObjectDetectionRadarCompareData](Code/Algorithms/ObjectDetectionRadarCompareData.py)
  - [Radar](Code/Radar):
    - [Radar](Code/Radar/Radar):
- [Literature](Literature/): Contains literature resources.
  - [Notes](Literature/Notes):
  - [Radar](Literature/Radar):
  - [PDF Resources](Literature/PDF Resources):
- [Radar](Literature/Radar/): Contains radar-related literature.


## Subjects to Cover
- [x] Obtain and read documentation about sensor [IWR6843AOPEVM](https://www.ti.com/tool/IWR6843AOPEVM)
- [x] Set UART communication using sensor web demo [IWR6843AOPEVM](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/)
- [x] Set UART communication using personal laptop [IWR6843AOPEVM](Code/Radar/Radar)
  - [x] Update code using classes.
  - [x] Update code using threads.
  - [x] Set logic to print Frame by Frame.
- [x] Store logs obtained from the sensor.
  - [x] Decode UART Frame.
  - [x] Decode each TLV from each Frame.
- [x] Create algorithm to visualize the obtained data.
- [ ] Create algorithm to detect Objects. [Object detection for automotive radar point clouds – a comparison](https://aiperspectives.springeropen.com/articles/10.1186/s42467-021-00012-z)
  - [x] Apply physical filters.
  - [ ] Apply PointNet++.
  - [ ] Apply Clustering.
    - [ ] Apply Stage one Clustering.
    - [ ] Apply Stage two Clustering.
  - [x] Apply Grid mapping.
- [ ] Create algorithm to estimate vehicle velocity.

## Authors

- [@LF-RoGu](https://github.com/LF-RoGu)
- Name: Luis Fernando Rodriguez Gutierrez.
- Matriculation Number: 7219085.

- [@LF-RoGu](https://github.com/lhckmn)
- Name: Leander Hackmann.
- Matriculation Number: 7217912.
