# Code


## Algorithms
In this folder of the repository you are going to find different types of algorithms. From the code that will run in a raspberrypi, to codes that are used for debugging. In which a file is read to obtain the point cloud or any other relevant information from the sensor.
## Modules
In this folder of the repository you are going to find the submodules used for the project. This modules work in a class based system, as the goal is for this system to be closer to c++ using and easier to pass in the future.

Modules that you will find:
### Data decoder
This modules obtaines the fames from the UART port. This frames are obtained from the UART port, as this will be out mean of communicating with the sensor. 
### Frame Aggregator
### Point Filter
### Self Speed Estimator
### Kalman Filter
### Ve Speed Filter
This module is obtaines the speed of the vehicle from "Self Speed Estimator" module, in which depending on this we filter out points that are we not interessted in. In this case since we are only focusing in the points that are static, this means only points that has a similar radial speed that the moving vehicle, STATIC objects. The rest are just filtered out.
## PlotLogs

## Radar