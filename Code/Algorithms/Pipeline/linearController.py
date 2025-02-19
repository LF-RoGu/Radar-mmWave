"""!
@file linearController.py
@brief Implements linear brake controller, depending on the speed is the actual distance that will be required.

@details This module provides tools for generating Cartesian and Polar occupancy grids
from point cloud data, allowing real-time environment mapping. The occupancy grid
provides spatial representation for detected objects, making it useful for
navigation and collision avoidance applications.

@defgroup Occupancy_Grid Occupancy Grid Processor
@brief Provides functionality for creating occupancy grids from radar point clouds.
@{
"""

# CLASS: LinearSpeedController
class LinearSpeedController:
    """!
    @class LinearSpeedController
    @ingroup Speed_Control
    @brief Implements a linear speed controller to determine stopping distances.

    @details This class models a simple linear control system that calculates the stopping
    distance based on the vehicle's speed. It also provides a control mechanism to adjust
    braking force to ensure the vehicle stops within a target distance.
    """
    def __init__(self, max_speed=20, ref_speed=40, ref_distance=6):
        """!
        @brief Initializes the braking controller with speed and distance parameters.

        @param in max_speed Maximum vehicle speed in kph (default: 20 kph).
        @param in ref_speed Reference speed for calibration (default: 40 kph).
        @param in ref_distance Stopping distance at reference speed (default: 6 meters).

        @ingroup Speed_Control
        """
        self.max_speed = max_speed  # Max speed in kph
        self.ref_speed = ref_speed  # Reference speed in kph
        self.ref_distance = ref_distance  # Stopping distance at ref_speed

    def stopping_distance(self, current_speed):
        """!
        @brief Computes the estimated stopping distance based on the current speed.

        @param in speed The current speed of the vehicle in kph.

        @return The calculated stopping distance in meters.
        
        @ingroup Speed_Control
        """
        return (current_speed / self.ref_speed) * self.ref_distance

    def control(self, current_speed, target_distance):
        """!
        @brief Computes the braking control signal to regulate stopping distance.

        @param in current_speed The current vehicle speed in kph.
        @param in target_distance The target stopping distance in meters.

        @return brake_signal A control signal between 0 and 1.
        
        @ingroup Speed_Control
        """
        required_distance = self.stopping_distance(current_speed)
        
        if target_distance <= required_distance:
            brake_signal = 1
        else:
            brake_signal = 0
        return brake_signal