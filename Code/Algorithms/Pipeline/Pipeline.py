import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from functools import partial

import dataDecoderBrokenTimestamp
from frameAggregator import FrameAggregator
import pointFilter
import selfSpeedEstimator


# -------------------------------
# Simulation parameters
# -------------------------------
#Defining the number of how many frames from the past should be used in the frame aggregator
# 0 = only current frame
# n = current frame + n previous frames
FRAME_AGGREGATOR_NUM_PAST_FRAMES = 9

#Defining a minimum SNR for the filter stage
FILTER_SNR_MIN = 0

#Defining minimum and maximum z for the filter stage
FILTER_Z_MIN = 0.3
FILTER_Z_MAX = 100

#Defining minimum and maximum phi for the filter stage
FILTER_PHI_MIN = -85
FILTER_PHI_MAX = 85


# -------------------------------
# FUNCTION: Updating the simulation when the value of the slider has changed
# -------------------------------
def update_sim(new_num_frame, curr_num_frame):
    #Checking if new frame is earlier than the current processed frame (--> simulation needs to be rebuild until this particular frame)
    if new_num_frame < curr_num_frame:
            ##Clearing the pipeline
            #Clearing the frame aggregator
            frame_aggregator.clearBuffer()

            
            #Setting the current frame to -1 to start feeding at index 0
            curr_num_frame = -1
    
    #Simulating the necessary frames
    for num_frame in range(curr_num_frame + 1, new_num_frame + 1, 1):
        ##Feeding the pipeline
        #Getting the current frame
        frame = frames[num_frame]

        #Updating the frame aggregator
        frame_aggregator.updateBuffer(frame)

        #Getting the current point cloud frum the frame aggregator
        point_cloud = frame_aggregator.getPoints()

        #Filtering by SNR
        point_cloud_filtered = pointFilter.filterSNRmin(point_cloud, FILTER_SNR_MIN)

        #Filtering by z
        point_cloud_filtered = pointFilter.filterCartesianZ(point_cloud_filtered, FILTER_Z_MIN, FILTER_Z_MAX)

        #Filtering by phi
        point_cloud_filtered = pointFilter.filterSphericalPhi(point_cloud_filtered, FILTER_PHI_MIN, FILTER_PHI_MAX)

        #Estimating the self-speed
        self_speed_raw = selfSpeedEstimator.estimate_self_speed(point_cloud_filtered)

    #Updating the current frame number to the new last processed frame
    curr_num_frame = new_num_frame


# -------------------------------
# Program entry point
# -------------------------------

##Getting the data
#Creating an absolute path to the raw data from a relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.abspath(os.path.join(script_dir, "../../../Logs/LogsPart3/DynamicMonitoring/30fps_straight_3x3_log_2024-12-16.csv"))

#Reading in the frames
frames = dataDecoderBrokenTimestamp.decodeData(log_file)


##Creating the pipeline's objects
#Creating the frame aggregator
frame_aggregator = FrameAggregator(FRAME_AGGREGATOR_NUM_PAST_FRAMES)


##Setting up the visualization and starting the simulation
#Creating a figure of size 10x10
fig = plt.figure(figsize=(10, 10))

#Defining a 2x1 grid layout
gs = GridSpec(2, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0], projection='3d')
ay = fig.add_subplot(gs[1, 0])

#Setting the initial view angle of the 3D-plot to top-down
ax.view_init(elev=90, azim=-90)

curr_num_frame = -1

#Creating a slider for frame selection and attaching a handler to the on_changed event
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)
slider.on_changed(partial(update_sim, additional_parameter=curr_num_frame))

#Starting the simulation with the first frame and showing the plot
update_sim(0, -1)
plt.show()