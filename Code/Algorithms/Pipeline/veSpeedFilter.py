import numpy as np

__all__ = ['calculateVe']

def calculateVe(point_cloud):
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
    point_cloud_ve_filtered = []

    #Calculating the difference of the total speed and comparing against the threshold
    for i in range(len(point_cloud)):
        if abs(point_cloud[i]["ve"] - self_speed_filtered) <= abs_threshold:
            point_cloud_ve_filtered.append(point_cloud[i])

    return point_cloud_ve_filtered