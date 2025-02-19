"""!
@file kalmanFilter.py
@brief Implements a Kalman filter for estimating self-speed.

@details This module provides a simple implementation of a 1D Kalman filter to smooth noisy
radar measurements and provide a more stable estimation of self-speed. The filter
incorporates process and measurement variances to estimate the true velocity over time.

@defgroup Kalman_Filter
@brief Implements a 1D Kalman filter for self-speed estimation.
@{
"""

class KalmanFilter:
    """!
    @class KalmanFilter
    @ingroup Kalman_Filter
    @brief Implements a simple Kalman filter for 1D estimation.
    """
    def __init__(self, process_variance, measurement_variance):
        """!
        @brief Initializes the Kalman filter with process and measurement variances.
        
        @param in process_variance The expected variance in the system process.
        @param in measurement_variance The expected variance in the measurements.
        
        @ingroup Kalman_Filter
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0.0
        self.estimated_error = 1.0

    def clear(self):
        """!
        @brief Resets the Kalman filter state.
        
        @ingroup Kalman_Filter
        """
        self.estimated_value = 0.0
        self.estimated_error = 1.0

    def update(self, measurement):
        """!
        @brief Updates the Kalman filter estimate based on a new measurement.

        @param in measurement The new measurement input from the radar sensor.
        @return The updated estimated value after applying the Kalman filter.
        
        @ingroup Kalman_Filter
        """
        # @brief Kalman Gain
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        # @brief Update the estimated value
        self.estimated_value = self.estimated_value + kalman_gain * (measurement - self.estimated_value)
        # @brief Updates the error covariance.
        self.estimated_error = (1 - kalman_gain) * self.estimated_error + self.process_variance
        return self.estimated_value
    
## @}  # End of Kalman_Filter group