import numpy as np
import matplotlib.pyplot as plt
import dataDecoder
import os

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0.0
        self.estimated_error = 1.0

    def update(self, measurement):
        # Kalman Gain
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        # Update the estimated value
        self.estimated_value = self.estimated_value + kalman_gain * (measurement - self.estimated_value)
        # Update the error covariance
        self.estimated_error = (1 - kalman_gain) * self.estimated_error + self.process_variance
        return self.estimated_value


#Initializing a figure for visualization
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
ax1, ax2, ax3 = axes

#Adding everything for kalman filtering the self speed
self_speed_history = []
kf_self_speed_history = []
kf_self_speed = KalmanFilter(process_variance=0.01, measurement_variance=0.1)


#Method 1: Fitting a curve into different angles and radial speeds
def estimating_self_speed(point_cloud):
    if len(point_cloud) < 1:
        return 0
    
    #Preparing an array to contain angle to target and radial speed
    phi_radspeed = []

    #Iterating over all points
    for i in range(len(point_cloud)):
        #Calculating the angle to target
        phi = np.rad2deg(np.arctan(point_cloud[i]["x"]/point_cloud[i]["y"]))

        #Appending the angle and the radial speed 
        phi_radspeed.append([phi, point_cloud[i]["doppler"]])

    #Converting array of tuples to NumPy array
    phi_radspeed = np.array(phi_radspeed, dtype=float)

    #Fitting a first order polynominal into the points
    poly_coeff = np.polyfit(phi_radspeed[:,0], phi_radspeed[:,1], deg=2)  # Polynomial coefficients
    poly_model = np.poly1d(poly_coeff)  # Polynomial model
    phi_fit = np.linspace(-90, 90, 100)
    phi_radial_speed_fit = poly_model(phi_fit)

    #Preparing plot
    ax2.clear()
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-5, 5)
    ax2.set_title("Radial speeds vs angles")
    ax2.set_xlabel("phi (deg)")
    ax2.set_ylabel("Radial speed (m/s)")

    #Plotting all points
    for i in range(len(phi_radspeed)):
        ax2.plot(phi_radspeed[i][0], phi_radspeed[i][1], 'kx')

    #Plotting fitted curve
    ax2.plot(phi_fit, phi_radial_speed_fit)

    #Returning the self-speed after interpolating
    return poly_model(0)
    
def update_self_speed_plot(self_speed_history, kf_self_speed_history):
    ax3.clear()
    ax3.set_xlim(0, 200)
    ax3.set_ylim(-5, 1)
    ax3.set_title("Re-calculated self-speed over time")
    ax3.set_xlabel("frame")
    ax3.set_ylabel("Self-speed (m/s)")

    #Creating a vector for the x axis
    x_vec = np.arange(len(self_speed_history))

    #Plotting raw self-speed
    ax3.plot(x_vec, self_speed_history, linestyle='--')

    #Plotting filtered self-speed
    ax3.plot(x_vec, kf_self_speed_history)


def update_point_plot(point_cloud):
    ax1.clear()
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(0, 15)
    ax1.set_title("Points of frame")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    for i in range(len(point_cloud)):
        ax1.plot(point_cloud[i]["x"], point_cloud[i]["y"], 'bx')


#Getting the data
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, '3_Targets_MoveAround_attempt4_log_2024-12-09.csv')
frames = dataDecoder.decodeData(log_file)

#Processing frame by frame
for frm in range(len(frames)):
    #Getting the point cloud of the frame
    point_cloud = frames[frm][1]
    
    #Calculating the self speed
    self_speed = estimating_self_speed(point_cloud)
    self_speed_history.append(self_speed)

    #Kalman filtering the self speed
    filtered_self_speed = kf_self_speed.update(self_speed)
    kf_self_speed_history.append(filtered_self_speed)

    #Updating the plots
    update_point_plot(point_cloud)
    update_self_speed_plot(self_speed_history, kf_self_speed_history)
    
    #Waiting
    plt.pause(0.01)

plt.ioff()
plt.show()