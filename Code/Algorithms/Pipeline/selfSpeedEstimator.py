import numpy as np

__all__ = ['estimate_self_speed']

def estimate_self_speed(pointCloud):
    """Estimate self-speed using a weighted mean of Doppler velocities."""
    
    # Return 0 if there are not enough points
    if len(pointCloud) < 3:
        return 0

    # Store Doppler velocities and their weights
    doppler_values = []
    weights = []

    for point in pointCloud:
        if point["y"] == 0:  # Avoid division by zero
            continue

        # Compute angle phi in degrees
        phi = np.rad2deg(np.arctan(point["x"] / point["y"]))

        # Compute weight using cos(phi) to give more importance to direct observations
        weight = abs(np.cos(np.radians(phi)))  # Avoid negative scaling

        doppler_values.append(point["doppler"] * weight)
        weights.append(weight)

    # If no valid data, return 0
    if not doppler_values or not weights:
        return 0

    # Compute weighted average of Doppler velocities
    self_speed_estimate = np.sum(doppler_values) / np.sum(weights)

    return self_speed_estimate
