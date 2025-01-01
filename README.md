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
- [Logs](Logs/): Contain logs captured in the vehicle.
  - [LogsPart1](Logs/LogsPart1):
  - [LogsPart2](Logs/LogsPart2):
  - [PlotLogs](Logs/PlotLogs):
    - [PlotLogs.py](Logs/PlotLogs/PlotLogs.py): Containg the file to visualize.
      - In line 397.
      - For the variable 'log_file', modify the folder "LogsPart2" to access the folder that you whish to obtain the data.
      - For the variable 'log_file', modify the folder "3_Target_straightLine_attempt1_log_2024-12-09.csv" to access the file that you whish to visualze
      - In line 407 & 410. Modify the parameter "doppler_threshold" for any threshold that you want. It will mean that points below that threshold will appear in one plot or the other.
        - The program will provide 2 option.
          - The data will be able to be seen in 2 different plots. One for stationary data (All data below a Doppler speed of 0.1 mps). The later is data that is considered as actual data.
          - Visualze line [1]. This one as the name states, will use the timestamp stored in the files to show a real life data-motion as it was recorded.
          - Visualize all data [2]. This one as the name states, will only put in a 3d plot all the recorded data.


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
  - [x] Apply Clustering.
    - [x] Apply Stage one Clustering.
    - [x] Apply Stage two Clustering.
  - [x] Apply Grid mapping.
- [x] Create algorithm to estimate vehicle velocity.

## Authors

- [@LF-RoGu](https://github.com/LF-RoGu)
- Name: Luis Fernando Rodriguez Gutierrez.
- Matriculation Number: 7219085.

- [@lhckmn](https://github.com/lhckmn)
- Name: Leander Hackmann.
- Matriculation Number: 7217912.
