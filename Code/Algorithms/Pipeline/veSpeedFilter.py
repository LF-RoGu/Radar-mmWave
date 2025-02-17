"""!
@file veSpeedFilter.py
@brief Computes and filters effective velocity (Ve) for radar-detected objects.

@details This module calculates the effective velocity (Ve) of detected objects
by using Doppler velocity and the object's azimuth angle. It also filters points
based on a threshold relative to the vehicle's self-speed.

@defgroup Ve_Speed_Filter Velocity Filtering
@brief Provides functions for calculating and filtering effective velocity (Ve).
@{
"""

import numpy as np

__all__ = ['calculateVe']

def calculateVe(point_cloud):
    """!
    @brief Computes the effective velocity (Ve) of detected objects.

    @param in point_cloud List of dictionaries containing detected radar points, each with `x`, `y`, and `doppler` values.
    
    @return List of points with computed `ve` values.
    
    @ingroup Ve_Speed_Filter
    """
    point_cloud_ve = []
    
    #Iterating over all points for calculaing ve
    for i in range(len(point_cloud)):
        point = point_cloud[i]
        
        #Calculating phi
        phi = np.rad2deg(np.arctan(point["x"]/point["y"]))
        
        #Calculating ve
        point["ve"] = point["doppler"] / np.cos(np.deg2rad(phi))

        point_cloud_ve.append(point)

    return point_cloud_ve


def filterPointsWithVe(point_cloud, self_speed_filtered, abs_threshold):
    """!
    @brief Filters points based on effective velocity relative to self-speed.
    
    @param in point_cloud List of dictionaries containing detected radar points, each with `ve` values.
    @param in self_speed_filtered Filtered self-speed estimation of the vehicle.
    @param in abs_threshold Maximum allowable deviation from self-speed.

    @return Filtered list of points.
    
    @ingroup Ve_Speed_Filter
    """
    point_cloud_ve_filtered = []

    #Calculating the difference of the total speed and comparing against the threshold
    for i in range(len(point_cloud)):
        if abs(point_cloud[i]["ve"] - self_speed_filtered) <= abs_threshold:
            point_cloud_ve_filtered.append(point_cloud[i])

    return point_cloud_ve_filtered

## @}  # End of Ve_Speed_Filter group