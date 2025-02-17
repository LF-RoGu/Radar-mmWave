"""!
@file selfSpeedEstimator.py
@brief Estimates the vehicle's self-speed based on radar Doppler measurements.

@details This module processes radar point cloud data to estimate the vehicle's self-speed
by fitting a polynomial model to the Doppler velocities of detected objects. The method
calculates the angle to each target and determines the self-speed using polynomial regression.

@defgroup Self_Speed_Estimator Self-Speed Estimator
@brief Provides functionality for estimating vehicle speed using radar data.
@{
"""
import numpy as np

__all__ = ['estimate_self_speed']

def estimate_self_speed(pointCloud):
    """!
    @brief Estimates the vehicle's self-speed using Doppler velocity data.

    @param in pointCloud List of dictionaries containing detected radar points, each with `x`, `y`, and `doppler` values.
    
    @return Estimated self-speed based on polynomial regression.
    
    @ingroup Self_Speed_Estimator
    """
    #Returning zero if there are no points to process
    if len(pointCloud) < 1:
        return 0

    #Preparing an array to contain angle to target and radial speed
    phi_radspeed = []

    #Iterating over all points
    for i in range(len(pointCloud)):
        #Calculating the angle to target
        phi = np.rad2deg(np.arctan(pointCloud[i]["x"]/pointCloud[i]["y"]))

        #Appending the angle and the radial speed 
        phi_radspeed.append([phi, pointCloud[i]["doppler"]])

    #Converting array of tuples to NumPy array
    phi_radspeed = np.array(phi_radspeed, dtype=float)

    #Fitting a first order polynominal into the points
    poly_coeff = np.polyfit(phi_radspeed[:,0], phi_radspeed[:,1], deg=2)  # Polynomial coefficients
    poly_model = np.poly1d(poly_coeff)  # Polynomial model

    #Returning the self-speed after interpolating
    return poly_model(0)

## @}  # End of Self_Speed_Estimator group