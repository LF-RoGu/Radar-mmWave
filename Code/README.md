# Code


## Algorithms
In this folder of the repository you are going to find different types of algorithms. From the code that will run in a raspberrypi, to codes that are used for debugging. In which a file is read to obtain the point cloud or any other relevant information from the sensor.
## Modules
In this folder of the repository you are going to find the submodules used for the project. This modules work in a class based system, as the goal is for this system to be closer to c++ using and easier to pass in the future.

Modules that you will find:
### Data decoder
This modules obtaines the fames from the UART port. This frames are obtained from the UART port, as this will be out mean of communicating with the sensor. 
### Frame Aggregator
This module obtains the point cloud from the sensor Payload, the goal of this module is to aggregate by an "N" factor, meaning that we simply aggregate point clouds from multiple frames to obtain a more accurate and detectable objects since we might not obtain enough points in a single frame. With this method and the high frame rate of the sensor, we ensure enough points are obtained for processing.
### Point Filter
This module obtains the point cloud, and the process that will be done in here is the phyisical filter of such points, in which we consider mainly the Z axis location of the point and its SNR obtained from the sensor, since this information is provided we will be using it.

Options are also added to filter depending on the X and Y axis, as well as treating the whole points in polar coordinates using "theta" and "r" as a parameter to filter points that are not to be considered.
### Self Speed Estimator

### Kalman Filter
This module is used to process the information regarding the self speed estimation values. As this information that we obtain is depending on the radial speed from the Doppler Effect, then we are bound to have certain relation to noise or uncertainties. This filter help us obtain a more "smooth" result for the estimation of speed.
### Ve Speed Filter
This module is obtaines the speed of the vehicle from "Self Speed Estimator" module, in which depending on this we filter out points that are we not interessted in. In this case since we are only focusing in the points that are static, this means only points that has a similar radial speed that the moving vehicle, STATIC objects. The rest are just filtered out.
## PlotLogs
This module contains the most simple scripts to plot the logged files from the sensor.
## Radar
This module is for the mmWave sensor to be functioning in C++. This code what it is doing right now is obtain the information from the sensor, for an "N" amount of frames, and stored it in a .csv file. All this needs to be ran through a computer.