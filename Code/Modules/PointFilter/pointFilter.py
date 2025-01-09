import numpy as np

__all__ = ['filterSNR', 'filterCartesianX', 'filterCartesianY', 'filterCartesianZ', 'filterSphericalR', 'filterSphericalTheta', 'filterSphericalPhi']

def filterSNRmin(inputPoints, snr_min):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_x = inputPoints[i]["snr"]
            if point_x >= snr_min:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
        print(f"Error filtering points: {e}")
        return None
    return filteredPoints

def filterCartesianX(inputPoints, x_min, x_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_x = inputPoints[i]["x"]
            if point_x >= x_min and point_x <= x_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
        print(f"Error filtering points: {e}")
        return None
    return filteredPoints

def filterCartesianY(inputPoints, y_min, y_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_y = inputPoints[i]["y"]
            if point_y >= y_min and point_y <= y_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
        print(f"Error filtering points: {e}")
        return None
    return filteredPoints

def filterCartesianZ(inputPoints, z_min, z_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_z = inputPoints[i]["z"]
            if point_z >= z_min and point_z <= z_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
        print(f"Error filtering points: {e}")
        return None
    return filteredPoints

def filterDoppler(inputPoints, doppler_min, doppler_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_doppler = inputPoints[i]["doppler"]
            if point_doppler >= doppler_min and point_doppler <= doppler_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
        print(f"Error filtering points: {e}")
        return None
    return filteredPoints

def filterSphericalR(inputPoints, r_min, r_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_r = np.sqrt(inputPoints[i]["x"]**2 + inputPoints[i]["y"]**2 + inputPoints[i]["z"]**2)
            if point_r >= r_min and point_r <= r_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
                print(f"Error filtering points: {e}")
                return None
    return filteredPoints

def filterSphericalTheta(inputPoints, theta_min, theta_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_r = np.sqrt(inputPoints[i]["x"]**2 + inputPoints[i]["y"]**2 + inputPoints[i]["z"]**2)
            point_theta = np.rad2deg(np.arccos(inputPoints[i]["z"] / point_r))
            if point_theta >= theta_min and point_theta <= theta_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
                print(f"Error filtering points: {e}")
                return None
    return filteredPoints

def filterSphericalPhi(inputPoints, phi_min, phi_max):
    filteredPoints = []
    try:
        for i in range(len(inputPoints)):
            point_phi = np.rad2deg(np.arctan(inputPoints[i]["x"]/inputPoints[i]["y"]))
            
            if point_phi >= phi_min and point_phi <= phi_max:
                filteredPoints.append(inputPoints[i])
    except (ValueError, IndexError) as e:
                print(f"Error filtering points: {e}")
                return None
    return filteredPoints