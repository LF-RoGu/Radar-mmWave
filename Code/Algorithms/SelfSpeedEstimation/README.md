# Estimating the self-speed of the car
This code tries to estimate the self-speed of the car by using the detected points and their radial speed.

## radialSpeedSimulation.py
This is a simulation of the self-speed estimation algorithm.\
A car drives through a static scenario and recognizes all targets (red spheres) that are within the radar's range (red spheres marked with
a green x).\
All targets produce a point containing the target's x and y coordinate and its radial speed.
The points are stored in two lists: the first list contains all points without noise and the second list contains all points with gaussian
noise added to the x and y coordinates. The radial speed is influenced indirectly by the noise of the x and y coordinates.\
The function "estimating_self_speed" uses a point cloud to estimate the car's self speed by calculating the angle (-90 to 90 deg) of each
point, "plotting" the points with the angle on the x-axis and the radial speed on the y-axis, and fitting a quadratic function through
these points.
The self-speed is then retrieved by evaluating the function's value at 0 deg.\
This value is returned and passed through a kalman filter.

## radialSpeedSimulationRealData.py
This is the the self-speed estimation algorithm running with real recorded data.\
The raw data is decoded and the algorithm is executed frame by frame to simulate the 
real scenario.\
A possibility to include a number of past frames when calling "estimating_self_speed"
added as it was found to reduce ripples significantly.